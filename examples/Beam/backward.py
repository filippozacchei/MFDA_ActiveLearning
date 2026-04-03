# %% [markdown]
# # Backward beam: Bayesian inversion with Active-MCMC
#
# Same workflow as ``run_backward_toy.py``, applied to the cantilever beam inverse
# problem.  Uses the library's inference machinery (``ActiveMCMCModel``,
# ``ActiveGPLogLike``, ``AdaptiveSubchain``, samplers, diagnostics).
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
from scipy.stats import multivariate_normal

from beam import make_spatial_grid, make_forward_model, make_observation
import matplotlib.pyplot as plt
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
from gp_active_mcmc.surrogates import MultiOutputGP
from gp_active_mcmc.utils.rng import set_seed


# =====================================================================
#  Direct GP surrogate (no POD -- observation space is low-dimensional)
# =====================================================================

class DirectGPSurrogate:
    """GP surrogate mapping theta -> observed beam displacements directly.

    Satisfies the ActiveSurrogate protocol (predict / update).
    """

    def __init__(self, gp: MultiOutputGP):
        self.gp = gp

    def predict(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        theta = np.asarray(theta, dtype=float).reshape(1, -1)
        y_mean, y_var = self.gp.predict(theta)
        return y_mean[0], np.maximum(y_var[0], 0.0)

    def update(self, theta: np.ndarray, y_hf: np.ndarray) -> None:
        theta_1d = np.asarray(theta, dtype=float).ravel()
        y_1d = np.asarray(y_hf, dtype=float).ravel()
        self.gp.update(theta_1d, y_1d)


# =====================================================================
#  Configuration
# =====================================================================

rng = set_seed(2)

# Spatial grid and observation locations
x = make_spatial_grid(n_pts=31, length=1.0)
obs_idx = np.array([2, 5, 8, 11, 14, 17, 20, 23, 26, 29])
x_obs = x[obs_idx]

# Custom distributed load
loads = np.array([
    13.944211, 14.107554, 14.168484, 14.127543, 14.080133, 14.031762, 14.037079,
    13.940349, 13.887439, 13.994669, 14.138576, 14.341531, 14.501729, 14.681951,
    14.879436, 15.143519, 15.300596, 15.375463, 15.359368, 15.278929, 15.114428,
    14.966691, 14.792335, 14.662425, 14.541461, 14.426502, 14.309434, 14.195700,
    14.127510, 13.982456, 13.863596,
])

# HF forward model: theta -> y_obs (observed displacements only)
hf_forward = make_forward_model(
    x=x, obs_idx=obs_idx, load=-loads, return_full_state=False,
)

# Prior over theta = [m1, m2, m3]  (log-stiffness on three sub-intervals)
prior_mean = np.array([10.0, 10.0, 10.0])
prior_cov = np.diag([2.0**2, 2.0**2, 2.0**2])
prior = multivariate_normal(mean=prior_mean, cov=prior_cov)

# Observation noise -- auto-scaled from a reference forward evaluation
y_ref = hf_forward(prior_mean)
signal_scale = float(np.max(np.abs(y_ref)))
sigma_obs = 0.02 * signal_scale          # 2 % relative noise

# Surrogate configuration
n_init = 200
gp_kernel = "matern52"
gp_ard = True

# Active coupling: trigger HF when avg LF std exceeds 10x observation noise
gamma_threshold = 100.0 * sigma_obs

# MCMC budget
n_coarse_evals = 20000
n_coarse_evals_da = 5000
burn_in = 2000
chunk_size = 500

print(f"signal_scale    = {signal_scale:.3e}")
print(f"sigma_obs       = {sigma_obs:.3e}")
print(f"gamma_threshold = {gamma_threshold:.3e}")


# %% [markdown]
# ## Synthetic observation

# %%
theta_true = prior.rvs(random_state=rng)
y_obs = make_observation(rng, theta_true, x, sigma_obs, obs_idx, load=-loads)

print(f"theta_true = {theta_true}")
print(f"y_obs      = {y_obs}")


# %% [markdown]
# ## Initial surrogate training set

# %%
theta_train = np.asarray(
    [prior.rvs(random_state=rng) for _ in range(n_init)], dtype=float,
)
y_train = np.asarray([hf_forward(th) for th in theta_train], dtype=float)


# %% [markdown]
# ## Fit a direct GP surrogate on observed outputs

# %%
gp = MultiOutputGP(
    X_train=theta_train,
    Y_train=y_train,
    kernel=gp_kernel,
    ard=gp_ard,
    noise_variance=1e-10,
    update_every=200,
    n_retrain_max=0,
)

# Two independent copies -- one per inference mode
lf_surrogate_single = DirectGPSurrogate(gp=copy.deepcopy(gp))
lf_surrogate_adapt = DirectGPSurrogate(gp=copy.deepcopy(gp))


# %% [markdown]
# ## Wrap LF + HF in an ActiveMCMCModel

# %%
model_single = ActiveMCMCModel(
    lf_model=lf_surrogate_single,
    hf_model=hf_forward,
    gamma_threshold=gamma_threshold,
)

adaptive_policy = AdaptiveSubchain(
    state=AdaptiveSubchainState(subchain_length=50),
    control=AdaptiveSubchainControl(
        update_every=5,
        target_error=sigma_obs,
        min_subchain=10,
        max_subchain=50,
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
theta0 = prior_mean.copy()

proposal = AdaptiveMetropolisShared(
    C0=0.001 * prior_cov,
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
    t=x_obs,
    y_obs=y_obs,
    y_true=hf_forward(theta_true),
    title="Surrogate prediction (before sampling)",
    show=True,
)


# %% [markdown]
# # Part 1 -- MCMC-guided active learning (single posterior)

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
    title="Single posterior: samples (m1 vs m2)",
    show=True,
)
fig1.savefig("plot_single_chain2d.png", dpi=150, bbox_inches="tight")

fig2, ax2 = plot_cumulative_hf_fraction(
    used_hf_single,
    title="Single posterior: cumulative HF fraction",
    show=True,
)
fig2.savefig("plot_single_hf_fraction.png", dpi=150, bbox_inches="tight")


# %% [markdown]
# # Part 2 -- DA-MCMC guided active learning with adaptive subchain

# %%
theta0 = prior_mean.copy()

proposal = AdaptiveMetropolisShared(
    C0=0.001 * prior_cov,
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
    title="Adaptive DA-MCMC: samples (m1 vs m2)",
    show=True,
)
fig3.savefig("plot_adapt_chain2d.png", dpi=150, bbox_inches="tight")

fig4, ax4 = plot_cumulative_hf_fraction(
    used_hf_adapt,
    title="Adaptive DA-MCMC: cumulative HF fraction",
    show=True,
)
fig4.savefig("plot_adapt_hf_fraction.png", dpi=150, bbox_inches="tight")

if chain_adapt.extras.subchain_length is not None:
    fig5, ax5 = plot_subchain_length_history(
        chain_adapt.extras.subchain_length, show=True,
    )
    fig5.savefig("plot_adapt_subchain.png", dpi=150, bbox_inches="tight")


# %% [markdown]
# ## Post-sampling: surrogate prediction at theta_true

# %%
plot_prediction_at_theta(
    model=lf_surrogate_adapt,
    theta=theta_true,
    t=x_obs,
    y_obs=y_obs,
    y_true=hf_forward(theta_true),
    title="Surrogate prediction (after DA-MCMC sampling)",
    show=True,
)


# %% [markdown]
# ## Corner plot of the posterior

# %%
from scipy.stats import gaussian_kde


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
                # Marginal KDE on the diagonal
                vals = post[:, i]
                kde = gaussian_kde(vals)
                xs = np.linspace(vals.min(), vals.max(), 300)
                ax.plot(xs, kde(xs), color="steelblue")
                ax.fill_between(xs, kde(xs), alpha=0.2, color="steelblue")
                if theta_true is not None:
                    ax.axvline(theta_true[i], color="crimson", ls="--", lw=1.2)
            else:
                # 2D scatter on the off-diagonal
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


# Corner plot for DA-MCMC posterior
fig_c1, _ = corner_plot(
    samples_adapt,
    labels=[r"$m_0$", r"$m_1$", r"$m_2$"],
    theta_true=theta_true,
    burn_in=burn_in,
    title="DA-MCMC posterior",
)
fig_c1.savefig("plot_corner_da_mcmc.png", dpi=150, bbox_inches="tight")
plt.show()

# Corner plot for single-posterior MCMC (optional comparison)
fig_c2, _ = corner_plot(
    samples_single,
    labels=[r"$m_0$", r"$m_1$", r"$m_2$"],
    theta_true=theta_true,
    burn_in=burn_in,
    title="Single posterior MCMC",
)
fig_c2.savefig("plot_corner_single.png", dpi=150, bbox_inches="tight")
plt.show()
