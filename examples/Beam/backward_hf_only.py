# %% [markdown]
# # Backward beam: Bayesian inversion with standard Metropolis-Hastings (HF only)
#
# Runs a single-level MCMC using **only** the fine (HF) forward model
# (no surrogate / no active learning).  The posterior distributions are
# visualised with marginal histograms and a corner plot.

# %% Imports
from __future__ import annotations

import numpy as np
import tinyDA as tda
from scipy.stats import multivariate_normal, gaussian_kde

from beam import make_spatial_grid, make_forward_model, make_observation
import matplotlib.pyplot as plt
from gp_active_mcmc.inference import AdaptiveMetropolisShared
from gp_active_mcmc.inference.chain import MCMCChain
from gp_active_mcmc.utils.mcmc import extract_samples
from gp_active_mcmc.utils.rng import set_seed


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
sigma_obs = 0.04 * signal_scale  # 4 % relative noise

# MCMC parameters
n_iterations = 10_000
burn_in = 2_000

print(f"signal_scale = {signal_scale:.3e}")
print(f"sigma_obs    = {sigma_obs:.3e}")


# %% [markdown]
# ## Synthetic observation

# %%
theta_true = np.array([9.3, 9.3, 9.2])
y_obs = make_observation(rng, theta_true, x, sigma_obs, obs_idx, load=-loads)

print(f"theta_true = {theta_true}")
print(f"y_obs      = {y_obs}")


# %% [markdown]
# ## Standard Gaussian log-likelihood (no variance inflation)

# %%
cov_obs = (sigma_obs**2) * np.eye(len(y_obs))
loglike = tda.AdaptiveGaussianLogLike(data=y_obs, covariance=cov_obs)


# %% [markdown]
# ## Build tinyDA Posterior (single-level, HF only)

# %%
posterior = tda.Posterior(prior, loglike, hf_forward)


# %% [markdown]
# ## Proposal distribution

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
# ## Run Metropolis-Hastings (HF only)

# %%
print(f"\nRunning MH with {n_iterations} iterations (HF model only)...")

chain_obj = tda.sample(
    posteriors=posterior,
    proposal=proposal,
    iterations=n_iterations,
    n_chains=1,
    force_sequential=True,
    initial_parameters=theta0,
    store_coarse_chain=True,
    subsampling_rate=1,
    adaptive_error_model=None,
)

# Extract samples into a numpy array
samples = extract_samples(chain=chain_obj, chain_key="chain_0")

# Wrap in MCMCChain for summary diagnostics
chain = MCMCChain.from_arrays(samples=samples)
summary = chain.summary(theta_true=theta_true, burn_in=burn_in)

print("\n--- MCMC Summary ---")
for k, v in summary.items():
    print(f"  {k}: {v}")


# %% [markdown]
# ## Trace plots

# %%
labels = [r"$m_0$", r"$m_1$", r"$m_2$"]
post_samples = samples[burn_in:]

fig_trace, axes_trace = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
for i in range(3):
    axes_trace[i].plot(samples[:, i], lw=0.4, alpha=0.7)
    axes_trace[i].axhline(theta_true[i], color="crimson", ls="--", lw=1.2, label="true")
    axes_trace[i].axvline(burn_in, color="grey", ls=":", lw=1.0, label="burn-in")
    axes_trace[i].set_ylabel(labels[i])
    axes_trace[i].legend(loc="upper right", fontsize=8)
axes_trace[-1].set_xlabel("Iteration")
fig_trace.suptitle("Trace plots (HF-only MH)", fontsize=14)
fig_trace.tight_layout()
fig_trace.savefig("plot_hf_trace.png", dpi=150, bbox_inches="tight")
plt.show()


# %% [markdown]
# ## Marginal posterior distributions

# %%
fig_marg, axes_marg = plt.subplots(1, 3, figsize=(12, 4))
for i in range(3):
    vals = post_samples[:, i]
    kde = gaussian_kde(vals)
    xs = np.linspace(vals.min(), vals.max(), 300)
    axes_marg[i].plot(xs, kde(xs), color="steelblue", lw=1.5)
    axes_marg[i].fill_between(xs, kde(xs), alpha=0.25, color="steelblue")
    axes_marg[i].axvline(theta_true[i], color="crimson", ls="--", lw=1.2, label="true")
    axes_marg[i].set_xlabel(labels[i])
    axes_marg[i].set_ylabel("Density" if i == 0 else "")
    axes_marg[i].legend(fontsize=8)
    axes_marg[i].set_title(f"Posterior {labels[i]}")
fig_marg.suptitle("Marginal posteriors (HF-only MH)", fontsize=14)
fig_marg.tight_layout()
fig_marg.savefig("plot_hf_marginals.png", dpi=150, bbox_inches="tight")
plt.show()


# %% [markdown]
# ## Corner plot

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


fig_corner, _ = corner_plot(
    samples,
    labels=labels,
    theta_true=theta_true,
    burn_in=burn_in,
    title="Posterior (HF-only Metropolis-Hastings)",
)
fig_corner.savefig("plot_hf_corner.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nDone. Figures saved: plot_hf_trace.png, plot_hf_marginals.png, plot_hf_corner.png")
