# %% [markdown]
# # Backward Navier-Stokes inverse problem (Active / Adaptive DA-MCMC)
#
# Bayesian inversion for a Navier-Stokes outlet-profile QoI using:
#
# - **LF model**: POD-GP surrogate trained on HF snapshots (fast, uncertain)
# - **HF model**: Navier-Stokes forward solver (expensive, accurate)
#
# The LF and HF models are coupled through `gp_active_mcmc.inference.ActiveMCMCModel`.
#
# Two strategies are demonstrated:
#
# 1. **Active MCMC (single posterior)**
#    Uses `model.coarse` only. HF is triggered *internally* when surrogate uncertainty is large.
#
# 2. **Adaptive DA-MCMC (recommended)**
#    Uses two posteriors `[coarse, fine]` and an adaptive subchain policy controlling how often
#    HF corrections are applied. Sampling is executed in chunks so the subsampling rate can
#    change between chunks.
#
# Notes
# -----
# - Parameters: `theta = [h1, U_in, L_down]` (adjust to your needs)
# - Observation: outlet profile `u_x(y)` resampled to `T` points
# - Likelihood: Gaussian with covariance `sigma_obs^2 I`
#
# How to run
# ----------
# - VSCode: run cells (`# %%`)
# - Terminal: `python run_backward_ns_hf_notebook.py`
#
# Requirements
# ------------
# This example requires the Navier-Stokes/FEniCS stack and the helper utilities in
# `examples/navier-stokes/utils/` (or your chosen path).

# %% Imports
from __future__ import annotations

import copy
from collections.abc import Callable

import numpy as np
import tinyDA as tda
from scipy.stats import multivariate_normal

# Local utilities (Navier-Stokes example)
from examples.navier_stokes.resample import resample_profile
from examples.navier_stokes.solver_hf import forward_model as hf_solver
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

# %%
SEED = 123
rng = set_seed(SEED)

# QoI
T = 100  # outlet profile length after resampling

# MCMC budgets (measured in coarse-evaluation units)
N_COARSE_EVALS = 1_000
BURN_IN = 100

# Surrogate training
N_INIT = 25  # number of HF snapshots for initial surrogate
POD_RANK = 5
GP_KERNEL = "matern52"
USE_ARD = True
NOISE_VARIANCE_GP = 1e-6
UPDATE_EVERY = 100
N_RETRAIN_MAX = 0

# Observation noise and active coupling
SIGMA_OBS = 0.01
GAMMA_THRESHOLD = 0.01  # triggers HF in coarse() when mean(var) > gamma_threshold**2

# Adaptive subchains (recommended strategy)
INIT_SUBCHAIN = 5  # initial subchain length
CHUNK_SIZE = 200  # coarse-evaluation units per chunk
TARGET_ERROR = 0.005  # RMSE target between LF mean and HF output at fine calls
UPDATE_EVERY_HF = 5  # update control every N HF evaluations
MIN_SUBCHAIN = 1
MAX_SUBCHAIN = 500

# Prior support (informative but not strictly enforced by MVN prior)
H1_MIN, H1_MAX = 0.05, 0.15
U_MIN, U_MAX = 0.5, 1.50
L_MIN, L_MAX = 0.30, 0.50

# %% [markdown]
# ## HF forward model wrapper
#
# We assume the HF solver returns an outlet profile `(y, u_x)` which we resample_ to length `T`.
# The active-learning / inference stack expects model outputs as 1D arrays.


# %%
def make_forward_model(*, T: int) -> Callable[[np.ndarray], np.ndarray]:
    def f(theta: np.ndarray) -> np.ndarray:
        th = np.asarray(theta, dtype=float).ravel()
        if th.shape[0] != 3:
            raise ValueError("Expected theta = [h1, U_in, L_down].")
        y, u = hf_solver(float(th[0]), U_in=float(th[1]), L_down=float(th[2]))
        return resample_profile(y, u, T=T)

    return f


hf_forward = make_forward_model(T=T)

# %% [markdown]
# ## Prior and synthetic observation

# %%
prior_mean = np.array([0.10, 1.00, 0.40])
prior_sig = np.array(
    [
        0.25 * (H1_MAX - H1_MIN),
        0.25 * (U_MAX - U_MIN),
        0.25 * (L_MAX - L_MIN),
    ]
)
prior_cov = np.diag(prior_sig**2)
prior = multivariate_normal(mean=prior_mean, cov=prior_cov)

theta_true = prior.rvs(random_state=rng)
y_clean = hf_forward(theta_true)
y_obs = y_clean + SIGMA_OBS * rng.standard_normal(size=T)

t_plot = np.arange(T)  # abscissa for plotting (index along resampled outlet line)

# %% [markdown]
# ## Build initial POD-GP surrogate (LF model)

# %%
X_init = prior.rvs(size=N_INIT, random_state=rng)
X_init = np.asarray(X_init, dtype=float)
if X_init.ndim == 1:
    X_init = X_init[None, :]

Y_init = np.asarray([hf_forward(th) for th in X_init], dtype=float)  # (N_INIT, T)

pod = POD(rank=POD_RANK).fit(Y_init)
A_init = pod.transform(Y_init)  # (N_INIT, POD_RANK)

gp = MultiOutputGP(
    X_train=X_init,
    Y_train=A_init,
    kernel=GP_KERNEL,
    ard=USE_ARD,
    noise_variance=NOISE_VARIANCE_GP,
    update_every=UPDATE_EVERY,
    n_retrain_max=N_RETRAIN_MAX,
)

lf_surrogate_base = PODGPSurrogate(pod=pod, gp=gp)

# %% [markdown]
# ## Likelihoods and posteriors
#
# - `ActiveGPLogLike` inflates the observation covariance using surrogate variance if the
#   forward call returns a `CoarseOutput` (LF case).
# - The fine posterior uses the standard tinyDA Gaussian likelihood, since `model.fine`
#   always returns HF output (plain array).

# %%
cov_obs = (SIGMA_OBS**2) * np.eye(T)

loglike_coarse = ActiveGPLogLike(data=y_obs, covariance=cov_obs)
loglike_fine = tda.AdaptiveGaussianLogLike(data=y_obs, covariance=cov_obs)

# %% [markdown]
# ## Proposal
#
# We use a shared Adaptive Metropolis proposal so that, in chunked sampling, proposal
# adaptation is preserved across chunks.

# %%
proposal = AdaptiveMetropolisShared(
    C0=0.1 * prior_cov,
    sd=1.0,
    adaptive=True,
    period=100,
    share_across_deepcopy=True,
)

theta0 = prior_mean.copy()

# %% [markdown]
# ## Strategy 1 — Active MCMC (single posterior, LF-first)
#
# HF is called only when `model.coarse` decides the surrogate uncertainty is too large.

# %%
model_active = ActiveMCMCModel(
    lf_model=copy.deepcopy(lf_surrogate_base),
    hf_model=hf_forward,
    gamma_threshold=GAMMA_THRESHOLD,
)

post_active = tda.Posterior(prior, loglike_coarse, model_active.coarse)

# Inspect surrogate prediction at the true parameter before sampling
plot_prediction_at_theta(
    model=model_active.lf_model,
    theta=theta_true,
    t=t_plot,
    y_obs=y_obs,
    y_true=y_clean,
    title=r"LF surrogate at $\theta_\mathrm{true}$ (before Active MCMC)",
    show=True,
)

result_active = sample_active_chain(
    model=model_active,
    posterior=post_active,
    proposal=proposal,
    iterations=N_COARSE_EVALS,
    initial_parameters=theta0,
    subsampling_rate=1,
    chain_key="chain_0",
    n_chains=1,
    force_sequential=True,
    store_coarse_chain=True,
)

chain_active = result_active.chain.burn_in(burn_in=BURN_IN)
samples_active = chain_active.samples
used_hf_active = chain_active.extras.used_hf

# Diagnostics
plot_prediction_at_theta(
    model=model_active.lf_model,
    theta=theta_true,
    t=t_plot,
    y_obs=y_obs,
    y_true=y_clean,
    title=r"LF surrogate at $\theta_\mathrm{true}$ (after Active MCMC)",
    show=True,
)

if used_hf_active is not None:
    plot_cumulative_hf_fraction(used_hf_active, burn_in=0, show=True)

# 2D view of (h1, U_in)
fig_a, ax_a = plot_chain_2d(
    samples_active[:, :2],
    used_hf=used_hf_active,
    theta_true=np.asarray(theta_true)[:2],
    names=("h1", "U_in"),
    title="Active MCMC samples (h1, U_in)",
    show=True,
)

# %% [markdown]
# ## Strategy 2 — Adaptive DA-MCMC (recommended)
#
# Uses two posteriors `[coarse, fine]` and an adaptive subchain policy, executed via chunked
# sampling (`sample_adaptive_active_chain`).

# %%
adaptive_policy = AdaptiveSubchain(
    state=AdaptiveSubchainState(subchain_length=INIT_SUBCHAIN),
    control=AdaptiveSubchainControl(
        update_every=UPDATE_EVERY_HF,
        target_error=TARGET_ERROR,
        min_subchain=MIN_SUBCHAIN,
        max_subchain=MAX_SUBCHAIN,
    ),
)

model_adaptive = ActiveMCMCModel(
    lf_model=copy.deepcopy(lf_surrogate_base),
    hf_model=hf_forward,
    gamma_threshold=GAMMA_THRESHOLD,
    adaptive=adaptive_policy,
)

post_coarse = tda.Posterior(prior, loglike_coarse, model_adaptive.coarse)
post_fine = tda.Posterior(prior, loglike_fine, model_adaptive.fine)
posterior_da = [post_coarse, post_fine]

plot_prediction_at_theta(
    model=model_adaptive.lf_model,
    theta=theta_true,
    t=t_plot,
    y_obs=y_obs,
    y_true=y_clean,
    title=r"LF surrogate at $\theta_\mathrm{true}$ (before Adaptive DA-MCMC)",
    show=True,
)

proposal = AdaptiveMetropolisShared(
    C0=0.1 * prior_cov,
    sd=1.0,
    adaptive=True,
    period=100,
    share_across_deepcopy=True,
)

theta0 = prior_mean.copy()

result_adapt = sample_adaptive_active_chain(
    model=model_adaptive,
    posterior=posterior_da,
    proposal=proposal,
    n_coarse_evals=N_COARSE_EVALS,
    initial_parameters=theta0,
    chain_key="chain_coarse_0",
    config=ChunkedMCMCConfig(chain_key="chain_coarse_0", chunk_size=CHUNK_SIZE),
    n_chains=1,
    force_sequential=True,
    store_coarse_chain=True,
)

chain_adapt = result_adapt.chain.burn_in(burn_in=BURN_IN)
samples_adapt = chain_adapt.samples
used_hf_adapt = chain_adapt.extras.used_hf
subchain_hist = chain_adapt.extras.subchain_length

plot_prediction_at_theta(
    model=model_adaptive.lf_model,
    theta=theta_true,
    t=t_plot,
    y_obs=y_obs,
    y_true=y_clean,
    title=r"LF surrogate at $\theta_\mathrm{true}$ (after Adaptive DA-MCMC)",
    show=True,
)

if used_hf_adapt is not None:
    plot_cumulative_hf_fraction(used_hf_adapt, burn_in=0, show=True)

fig_b, ax_b = plot_chain_2d(
    samples_adapt[:, :2],
    used_hf=used_hf_adapt,
    theta_true=np.asarray(theta_true)[:2],
    names=("h1", "U_in"),
    title="Adaptive DA-MCMC samples (h1, U_in)",
    show=True,
)

if subchain_hist is not None:
    plot_subchain_length_history(subchain_hist, show=True)

# %% [markdown]
# ## Summaries

# %%
print("Active MCMC summary:", chain_active.summary(theta_true=theta_true))
print("Adaptive DA-MCMC summary:", chain_adapt.summary(theta_true=theta_true))
