# %% [markdown]
# # Backward tritium: Bayesian inversion with Active-MCMC
#
# Same workflow as ``run_backward_toy.py``, applied to the Achlys tritium diffusion
# benchmark.  The HF model runs via UM-Bridge (Docker container).
#
# **Before running**, start the Docker server:
#
# ```bash
# docker run -it -p 4243:4243 linusseelinger/benchmark-achlys:latest
# ```
#
# Two inference modes:
#
# 1. **MCMC-guided active learning (single posterior)**
# 2. **DA-MCMC guided active learning with adaptive subchain (recommended)**

# %% Imports
from __future__ import annotations

import copy

import numpy as np
import tinyDA as tda
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, uniform

from tritium import (
    make_forward_model,
    make_observation,
    make_time_grid,
    sample_prior,
    PARAM_NAMES,
    PRIOR_BOUNDS,
    N_OUTPUT,
)
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


# =====================================================================
#  Configuration
# =====================================================================

import os
import time

rng = set_seed(2)

# Time grid (for plotting)
t = make_time_grid(n_pts=N_OUTPUT)

# HF forward model via UM-Bridge
hf_forward = make_forward_model(url="http://localhost:4243", model_name="forward")

# Timed wrapper to track HF cost
_hf_count = 0
_hf_time = 0.0

def hf_forward_timed(theta):
    global _hf_count, _hf_time
    t0 = time.perf_counter()
    y = hf_forward(theta)
    dt = time.perf_counter() - t0
    _hf_count += 1
    _hf_time += dt
    if _hf_count % 10 == 0:
        print(f"  [HF] {_hf_count} calls, total {_hf_time:.1f}s, last {dt:.1f}s")
    return y

# ---- Prior ----
# tinyDA expects a scipy-style prior with .rvs() and .logpdf().
# We build an independent uniform prior from the benchmark bounds.
lo = PRIOR_BOUNDS[:, 0]
hi = PRIOR_BOUNDS[:, 1]


class UniformBoxPrior:
    """Independent uniform prior matching tinyDA's interface."""

    def __init__(self, lo: np.ndarray, hi: np.ndarray):
        self.lo = np.asarray(lo, dtype=float)
        self.hi = np.asarray(hi, dtype=float)
        self.d = len(lo)
        self._marginals = [
            uniform(loc=lo[i], scale=hi[i] - lo[i]) for i in range(self.d)
        ]

    def rvs(self, size: int = 1, random_state=None) -> np.ndarray:
        gen = np.random.default_rng(random_state) if not isinstance(
            random_state, np.random.Generator
        ) else random_state
        samples = gen.uniform(self.lo, self.hi, size=(size, self.d))
        return samples[0] if size == 1 else samples

    def logpdf(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float).ravel()
        if np.any(x < self.lo) or np.any(x > self.hi):
            return -np.inf
        return float(-np.sum(np.log(self.hi - self.lo)))


prior = UniformBoxPrior(lo, hi)

# ---- Observation noise ----
# Use cached reference eval to avoid an extra HF call
REF_CACHE = os.path.join(os.path.dirname(__file__) or ".", "ref_eval.npz")
theta_centre = 0.5 * (lo + hi)
if os.path.exists(REF_CACHE):
    _ref = np.load(REF_CACHE)
    y_ref = _ref["y_ref"]
    print("Loaded cached reference evaluation")
else:
    y_ref = hf_forward(theta_centre)
    np.savez(REF_CACHE, y_ref=y_ref, theta_centre=theta_centre)
    print("Computed and cached reference evaluation")
signal_scale = float(np.max(np.abs(y_ref)))
sigma_obs = 0.02 * signal_scale          # 2% relative noise

# ---- Surrogate configuration ----
n_init = 200
pod_rank = 20
gp_kernel = "matern52"
gp_ard = True
noise_variance_gp = 1e-6

# ---- Active coupling thresholds ----
# With a slow HF model, keep thresholds high to minimize HF calls.
# The surrogato does most of the work; HF fires only when really needed.
gamma_threshold_single = 200.0 * sigma_obs
gamma_threshold_da = 100.0 * sigma_obs

# ---- MCMC budget ----
# Reduced budgets for a slow HF model
n_coarse_evals = 5_000
n_coarse_evals_da = 3_000
burn_in = 500
chunk_size = 500

print(f"signal_scale             = {signal_scale:.3e}")
print(f"sigma_obs                = {sigma_obs:.3e}")
print(f"gamma_threshold (single) = {gamma_threshold_single:.3e}")
print(f"gamma_threshold (DA)     = {gamma_threshold_da:.3e}")


# %% [markdown]
# ## Synthetic observation

# %%
OBS_CACHE = os.path.join(os.path.dirname(__file__) or ".", "observation.npz")
if os.path.exists(OBS_CACHE):
    _obs = np.load(OBS_CACHE)
    theta_true, y_obs = _obs["theta_true"], _obs["y_obs"]
    print("Loaded cached observation")
else:
    theta_true = sample_prior(rng, n=1)
    y_obs = make_observation(rng, theta_true, hf_forward, sigma_obs)
    np.savez(OBS_CACHE, theta_true=theta_true, y_obs=y_obs)
    print("Computed and cached observation")

print(f"theta_true = {theta_true}")
print(f"y_obs shape = {y_obs.shape}")


# %% [markdown]
# ## Initial surrogate training set

# %%
TRAIN_CACHE = os.path.join(os.path.dirname(__file__) or ".", "train_data.npz")
if os.path.exists(TRAIN_CACHE):
    _tr = np.load(TRAIN_CACHE)
    theta_train, y_train = _tr["theta_train"], _tr["y_train"]
    print(f"Loaded cached training data ({theta_train.shape[0]} samples)")
else:
    theta_train = sample_prior(rng, n=n_init)
    y_train_list = []
    for i, th in enumerate(theta_train):
        t0 = time.perf_counter()
        print(f"  Training snapshot {i+1}/{n_init} ...", end=" ", flush=True)
        y = hf_forward(th)
        y_train_list.append(y)
        print(f"done in {time.perf_counter()-t0:.1f}s")
    y_train = np.asarray(y_train_list, dtype=float)
    np.savez(TRAIN_CACHE, theta_train=theta_train, y_train=y_train)
    print(f"Saved training data to {TRAIN_CACHE}")


# %% [markdown]
# ## Fit a POD-GP surrogate

# %%
pod = POD(rank=pod_rank).fit(y_train)
coeffs = pod.transform(y_train)[:, :pod_rank]

gp = MultiOutputGP(
    X_train=theta_train,
    Y_train=coeffs,
    kernel=gp_kernel,
    ard=gp_ard,
    noise_variance=noise_variance_gp,
    update_every=200,
    n_retrain_max=0,
)

# Two independent copies -- one per inference mode
lf_surrogate_single = PODGPSurrogate(pod=copy.deepcopy(pod), gp=copy.deepcopy(gp))
lf_surrogate_adapt = PODGPSurrogate(pod=copy.deepcopy(pod), gp=copy.deepcopy(gp))


# %% [markdown]
# ## Wrap LF + HF in an ActiveMCMCModel

# %%
model_single = ActiveMCMCModel(
    lf_model=lf_surrogate_single,
    hf_model=hf_forward_timed,
    gamma_threshold=gamma_threshold_single,
)

adaptive_policy = AdaptiveSubchain(
    state=AdaptiveSubchainState(subchain_length=25),
    control=AdaptiveSubchainControl(
        update_every=10,
        target_error=sigma_obs,
        min_subchain=5,
        max_subchain=100,
        grow_factor=1.5,
        shrink_factor=0.7,
    ),
)

model_adapt = ActiveMCMCModel(
    lf_model=lf_surrogate_adapt,
    hf_model=hf_forward_timed,
    gamma_threshold=gamma_threshold_da,
    adaptive=adaptive_policy,
)


# %% [markdown]
# ## Likelihood and posterior objects

# %%
cov = (sigma_obs**2) * np.eye(len(y_obs))

loglike_coarse = ActiveGPLogLike(data=y_obs, covariance=cov)
loglike_fine = tda.AdaptiveGaussianLogLike(data=y_obs, covariance=cov)

# Single-level posterior
posterior_single = tda.Posterior(prior, loglike_coarse, model_single.coarse)

# Two-level posterior (DA-MCMC)
posterior_adapt = [
    tda.Posterior(prior, loglike_coarse, model_adapt.coarse),
    tda.Posterior(prior, loglike_fine, model_adapt.fine),
]


# %% [markdown]
# ## Proposal

# %%
theta0 = theta_centre.copy()

# Scale initial covariance from prior range
prior_range = hi - lo
C0 = 0.001 * np.diag(prior_range**2)

proposal = AdaptiveMetropolisShared(
    C0=C0,
    period=100,
    share_across_deepcopy=True,
    adaptive=True,
    sd=1,
)


# %% [markdown]
# ## Sanity check: surrogate prediction before sampling

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
    title="Single posterior: samples (E1 vs E2)",
    show=True,
)
fig1.savefig("plot_tritium_single_chain2d.png", dpi=150, bbox_inches="tight")

fig2, ax2 = plot_cumulative_hf_fraction(
    used_hf_single,
    title="Single posterior: cumulative HF fraction",
    show=True,
)
fig2.savefig("plot_tritium_single_hf_fraction.png", dpi=150, bbox_inches="tight")


# %% [markdown]
# # Part 2 — DA-MCMC guided active learning with adaptive subchain

# %%
theta0 = theta_centre.copy()

proposal = AdaptiveMetropolisShared(
    C0=C0,
    period=100,
    share_across_deepcopy=True,
    adaptive=True,
    sd=1,
)

result_adapt = sample_adaptive_active_chain(
    model=model_adapt,
    posterior=posterior_adapt,
    proposal=proposal,
    n_coarse_evals=n_coarse_evals_da,
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
    title="Adaptive DA-MCMC: samples (E1 vs E2)",
    show=True,
)
fig3.savefig("plot_tritium_adapt_chain2d.png", dpi=150, bbox_inches="tight")

fig4, ax4 = plot_cumulative_hf_fraction(
    used_hf_adapt,
    title="Adaptive DA-MCMC: cumulative HF fraction",
    show=True,
)
fig4.savefig("plot_tritium_adapt_hf_fraction.png", dpi=150, bbox_inches="tight")

if chain_adapt.extras.subchain_length is not None:
    fig5, ax5 = plot_subchain_length_history(
        chain_adapt.extras.subchain_length, show=True,
    )
    fig5.savefig("plot_tritium_adapt_subchain.png", dpi=150, bbox_inches="tight")


# %% [markdown]
# ## Post-sampling: surrogate prediction at theta_true

# %%
plot_prediction_at_theta(
    model=lf_surrogate_adapt,
    theta=theta_true,
    t=t,
    y_obs=y_obs,
    y_true=hf_forward(theta_true),
    title="Surrogate prediction (after DA-MCMC sampling)",
    show=True,
)


# %% [markdown]
# ## Corner plot of the posterior

# %%
def corner_plot(
    samples: np.ndarray,
    labels: list[str],
    theta_true: np.ndarray | None = None,
    burn_in: int = 0,
    title: str = "",
) -> tuple:
    """Pair plot with marginal KDEs on the diagonal and scatter on off-diagonal."""
    post = samples[burn_in:]
    d = post.shape[1]

    fig, axes = plt.subplots(d, d, figsize=(3 * d, 3 * d))

    for i in range(d):
        for j in range(d):
            ax = axes[i, j]

            if j > i:
                ax.axis("off")
                continue

            if i == j:
                vals = post[:, i]
                kde = gaussian_kde(vals)
                xs = np.linspace(vals.min(), vals.max(), 300)
                ax.plot(xs, kde(xs), color="steelblue")
                ax.fill_between(xs, kde(xs), alpha=0.2, color="steelblue")
                if theta_true is not None:
                    ax.axvline(theta_true[i], color="crimson", ls="--", lw=1.2)
            else:
                ax.scatter(post[:, j], post[:, i], s=1, alpha=0.3, color="steelblue")
                if theta_true is not None:
                    ax.scatter(
                        theta_true[j], theta_true[i],
                        s=60, marker="*", color="crimson", edgecolors="black", zorder=5,
                    )

            if i == d - 1:
                ax.set_xlabel(labels[j])
            else:
                ax.set_xticklabels([])
            if j == 0 and i != 0:
                ax.set_ylabel(labels[i])
            elif j != 0:
                ax.set_yticklabels([])

    if title:
        fig.suptitle(title, fontsize=14, y=1.01)
    fig.tight_layout()
    return fig, axes


# %%
labels = [r"$E_1$", r"$E_2$", r"$E_3$", r"$n_1$", r"$n_2$"]

fig_c1, _ = corner_plot(
    samples_adapt,
    labels=labels,
    theta_true=theta_true,
    burn_in=burn_in,
    title="DA-MCMC posterior (Tritium)",
)
fig_c1.savefig("plot_tritium_corner_da_mcmc.png", dpi=150, bbox_inches="tight")
plt.show()

fig_c2, _ = corner_plot(
    samples_single,
    labels=labels,
    theta_true=theta_true,
    burn_in=burn_in,
    title="Single posterior MCMC (Tritium)",
)
fig_c2.savefig("plot_tritium_corner_single.png", dpi=150, bbox_inches="tight")
plt.show()
