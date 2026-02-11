# %% [markdown]
# # Backward toy: Bayesian inversion with Active-MCMC (documentation notebook)
#
# This notebook-style tutorial demonstrates *how to use* `gp_active_mcmc` on a small toy
# inverse problem. The goal is clarity and API usage, not performance.
#
# We show **two inference modes** supported by the library:
#
# 1. **MCMC-guided active learning (single posterior)**
#    The sampler uses only the *coarse* model (`ActiveMCMCModel.coarse`). The model decides
#    internally when to fall back to the high-fidelity (HF) forward model based on a simple
#    uncertainty trigger.
#
# 2. **DA-MCMC guided active learning with an adaptive subchain (two posteriors, recommended)**
#    The sampler uses a *coarse* posterior and a *fine* posterior (delayed acceptance).
#    An `AdaptiveSubchain` monitors LF-HF discrepancy during HF corrections and adjusts
#    how often the fine level is called. Because `tinyDA` uses a fixed subsampling rate per
#    call, the run is performed in *chunks*.
#
# Along the way we:
#
# - define a Gaussian prior and generate a synthetic observation,
# - train a POD-GP surrogate on an initial design,
# - wrap surrogate + HF model into an `ActiveMCMCModel`,
# - run short chains and plot lightweight diagnostics.
#
# The budgets are intentionally small so the tutorial executes quickly in documentation builds.

# %% Imports
from __future__ import annotations

import copy

import numpy as np
import tinyDA as tda
from scipy.stats import multivariate_normal

# For documentation builds, import the toy helpers from the tutorial package.
# If you place this file at `docs/tutorials/backward_toy_notebook.py`, also place
# `toy.py` in the same directory so this import works.
from toy_tutorial import make_forward_model, make_observation, make_timeline

# Diagnostics return (fig, ax) and never force `plt.show()` unless `show=True`.
from gp_active_mcmc.diagnostics import (
    plot_chain_2d,
    plot_cumulative_hf_fraction,
    plot_prediction_at_theta,
    plot_subchain_length_history,
)
from gp_active_mcmc.inference import (
    ActiveGPLogLike,
    ActiveMCMCModel,
    AdaptiveMetropolisShared,
    AdaptiveSubchain,
    AdaptiveSubchainControl,
    AdaptiveSubchainState,
    ChunkedMCMCConfig,
    sample_active_chain,
    sample_adaptive_active_chain,
)
from gp_active_mcmc.surrogates import POD, MultiOutputGP, PODGPSurrogate
from gp_active_mcmc.utils.rng import set_seed

# %% [markdown]
# ## Configuration
#
# - `t` defines the toy trajectory grid (the observation space).
# - The prior is a multivariate Normal over three parameters.
# - The surrogate is trained on `n_init` HF snapshots drawn from the prior.
# - `gamma_threshold` controls when `ActiveMCMCModel.coarse` triggers HF:
#   it compares the *mean surrogate variance* to `gamma_threshold**2`.

# %%
rng = set_seed(2)

# Time grid for the toy trajectory
t = make_timeline(T=500, t_end=0.05)

# High-fidelity forward model: theta -> y(t)
hf_forward = make_forward_model(t)

# Prior over theta = [A, f, tau]
prior_mean = np.array([0.8, 150.0, 0.01])
prior_cov = np.diag([0.5**2, 40.0**2, 0.01**2])
prior = multivariate_normal(mean=prior_mean, cov=prior_cov)

# Observation noise (standard deviation)
sigma_obs = 0.1

# Surrogate configuration (small budgets for docs)
n_init = 50
pod_rank = 20
gp_kernel = "matern52"
gp_ard = True
noise_variance_gp = 1e-6

# Active coupling: trigger HF when average predictive std exceeds gamma_threshold
gamma_threshold = 0.10

# MCMC budget (expressed as coarse evaluations)
n_coarse_evals = 1000
burn_in = 100

# Chunking for the adaptive workflow
chunk_size = 250

# %% [markdown]
# ## Synthetic observation
#
# We generate a synthetic "truth" parameter `theta_true`, evaluate the HF model, and
# corrupt with Gaussian noise. This is what the likelihood will condition on.

# %%
theta_true = prior.rvs(random_state=rng)
y_obs = make_observation(rng, theta_true, t, sigma_obs)

# %% [markdown]
# ## Initial surrogate training set
#
# We build an initial design by sampling parameters from the prior and evaluating the HF model.
# This yields training pairs `(theta_i, y_i(t))` for POD and GP fitting.

# %%
theta_train = prior.rvs(size=n_init, random_state=rng)
y_train = np.asarray([hf_forward(th) for th in np.atleast_2d(theta_train)], dtype=float)

# %% [markdown]
# ## Fit a POD-GP surrogate
#
# 1. Fit POD on snapshot matrix `y_train` to obtain a reduced basis.
# 2. Project snapshots to POD coefficients.
# 3. Fit a multi-output GP mapping `theta -> coefficients`.

# %%
pod = POD(rank=pod_rank).fit(y_train)
coeffs = pod.transform(y_train)  # (n_init, pod_rank)

gp = MultiOutputGP(
    theta_train,
    coeffs,
    kernel=gp_kernel,
    ard=gp_ard,
    noise_variance=noise_variance_gp,
    update_every=100,
    n_retrain_max=0,
)

# We create two surrogate *instances* so the two runs below do not share state.
# This keeps the tutorial results easier to interpret.
lf_surrogate_single = PODGPSurrogate(pod=copy.deepcopy(pod), gp=copy.deepcopy(gp))
lf_surrogate_adapt = PODGPSurrogate(pod=copy.deepcopy(pod), gp=copy.deepcopy(gp))

# %% [markdown]
# ## Wrap LF + HF in an ActiveMCMCModel
#
# `ActiveMCMCModel` is the heart of the inference API. It couples:
# - an LF surrogate (`predict`, `update`)
# - an HF forward model (`theta -> y`)
#
# It exposes two callables intended for MCMC:
#
# - `coarse(theta)`: LF-first; triggers HF if uncertainty is too large.
# - `fine(theta)`: always HF; updates the surrogate.
#
# The **choice of posteriors** determines the inference mode:
#
# - **single posterior** (`Posterior(..., model.coarse)`) → MCMC-guided active learning
# - **two posteriors** (`[Posterior(..., model.coarse), Posterior(..., model.fine)]`) → DA-MCMC guided AL
#
# In the adaptive subchain workflow, DA-MCMC is mandatory because subchains describe
# how often we attempt fine (HF) corrections relative to coarse steps.

# %%
model_single = ActiveMCMCModel(
    lf_model=lf_surrogate_single,
    hf_model=hf_forward,
    gamma_threshold=gamma_threshold,
)

adaptive_policy = AdaptiveSubchain(
    state=AdaptiveSubchainState(subchain_length=20),
    control=AdaptiveSubchainControl(
        update_every=5,
        target_error=0.05,
        min_subchain=1,
        max_subchain=500,
    ),
)

model_adapt = ActiveMCMCModel(
    lf_model=lf_surrogate_adapt,
    hf_model=hf_forward,
    gamma_threshold=gamma_threshold,
    adaptive=adaptive_policy,
)

# %% [markdown]
# ## Likelihood and posterior objects
#
# `ActiveGPLogLike` inflates the observation covariance when the model output carries a
# predictive variance (the LF surrogate case).
#
# For the fine level we use `tinyDA.AdaptiveGaussianLogLike`, which expects a plain mean vector.

# %%
cov = (sigma_obs**2) * np.eye(len(y_obs))

loglike_coarse = ActiveGPLogLike(data=y_obs, covariance=cov)
loglike_fine = tda.AdaptiveGaussianLogLike(data=y_obs, covariance=cov)

# Single-level posterior (MCMC-guided active learning)
posterior_single = tda.Posterior(prior, loglike_coarse, model_single.coarse)

# Two-level posterior list (DA-MCMC)
posterior_adapt = [
    tda.Posterior(prior, loglike_coarse, model_adapt.coarse),
    tda.Posterior(prior, loglike_fine, model_adapt.fine),
]

# %% [markdown]
# ## Proposal
#
# We use an adaptive Metropolis proposal. The `AdaptiveMetropolisShared` wrapper ensures
# proposal adaptation is not accidentally reset by deepcopies performed in chunked sampling.

# %%
theta0 = prior_mean.copy()

proposal = AdaptiveMetropolisShared(
    C0=0.1 * prior_cov,
    period=100,
    share_across_deepcopy=True,
    adaptive=True,
    sd=1,
)

# %% [markdown]
# ## Sanity check: surrogate prediction before sampling
#
# This plot is only meant to provide intuition: the surrogate has been trained on a small
# initial design and may be inaccurate away from the training region.

# %%
plot_prediction_at_theta(
    model=lf_surrogate_single,
    theta=theta_true,
    t=t,
    y_obs=y_obs,
    y_true=hf_forward(theta_true),
    title="Surrogate prediction (before sampling)",
    show=True,
)

# %% [markdown]
# # Part 1 — MCMC-guided active learning (single posterior)
#
# Here we pass a *single* posterior using `model.coarse`. The chain is driven by the coarse model.
# HF calls occur only when the uncertainty trigger fires inside `model.coarse`.

# %%
result_single = sample_active_chain(
    model=model_single,
    posterior=posterior_single,
    proposal=proposal,
    subsampling_rate=1,
    iterations=n_coarse_evals,
    initial_parameters=theta0,
    chain_key="chain_0",
)

chain_single = result_single.chain
chain_single.summary(theta_true=theta_true, burn_in=burn_in)


# %%
samples_single = chain_single.samples
used_hf_single = chain_single.extras.used_hf

fig1, ax1 = plot_chain_2d(
    samples_single[:, :2],
    used_hf=used_hf_single,
    theta_true=theta_true[:2],
    title="Single posterior: samples (first two parameters)",
    show=True,
)

fig2, ax2 = plot_cumulative_hf_fraction(
    used_hf_single,
    title="Single posterior: cumulative HF fraction",
    show=True,
)

fig_s1, ax_s1 = plot_prediction_at_theta(
    model_single.lf_model,
    theta=theta_true,
    y_obs=y_obs,
    t=t,
    y_true=hf_forward(theta_true),
    title="Surrogate prediction (after MCMC sampling)",
)
# %% [markdown]
# # Part 2 — DA-MCMC guided active learning with an adaptive subchain (recommended)
#
# Now we pass *two* posteriors `[coarse, fine]` and sample with `sample_adaptive_active_chain`.
#
# The adaptive policy updates `subchain_length` online using LF-HF discrepancy computed at HF calls.
# Because `tinyDA` uses a fixed `subsampling_rate` per call, we run sampling in chunks and update the
# subsampling rate between chunks.

# %%
theta0 = prior_mean.copy()

proposal = AdaptiveMetropolisShared(
    C0=0.1 * prior_cov,
    period=100,
    share_across_deepcopy=True,
    adaptive=True,
    sd=1,
)

result_adapt = sample_adaptive_active_chain(
    model=model_adapt,
    posterior=posterior_adapt,
    proposal=proposal,
    n_coarse_evals=n_coarse_evals,
    initial_parameters=theta0,
    chain_key="chain_coarse_0",
    config=ChunkedMCMCConfig(chain_key="chain_coarse_0", chunk_size=chunk_size),
    store_coarse_chain=True,
)

chain_adapt = result_adapt.chain
chain_adapt.summary(theta_true=theta_true, burn_in=burn_in)

# %%
samples_adapt = chain_adapt.samples
used_hf_adapt = chain_adapt.extras.used_hf

fig3, ax3 = plot_chain_2d(
    samples_adapt[:, :2],
    used_hf=used_hf_adapt,
    theta_true=theta_true[:2],
    title="Adaptive DA-MCMC: samples (first two parameters)",
    show=True,
)

fig4, ax4 = plot_cumulative_hf_fraction(
    used_hf_adapt,
    title="Adaptive DA-MCMC: cumulative HF fraction",
    show=True,
)

if chain_adapt.extras.subchain_length is not None:
    fig5, ax5 = plot_subchain_length_history(chain_adapt.extras.subchain_length, show=True)

fig_s2, ax_s2 = plot_prediction_at_theta(
    model_adapt.lf_model,
    theta=theta_true,
    y_obs=y_obs,
    y_true=hf_forward(theta_true),
    t=t,
    title="Surrogate prediction (after DA-MCMC sampling)",
)
