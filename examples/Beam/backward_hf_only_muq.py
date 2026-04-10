# %% [markdown]
# # Backward beam: HF-only Metropolis-Hastings with the MUQ forward model
#
# Runs a single-level MCMC using **only** the HF forward model reimplemented
# from ``model/BeamModel.py`` (with moment of inertia and the correct FD
# stencil).  Data is loaded from ``model/ProblemDefinition.h5``.
#
# No surrogate, no active learning — just plain MH to see the posterior.

# %% Imports
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import tinyDA as tda
from scipy.stats import multivariate_normal, gaussian_kde

import matplotlib.pyplot as plt
from gp_active_mcmc.inference import AdaptiveMetropolisShared
from gp_active_mcmc.inference.chain import MCMCChain
from gp_active_mcmc.utils.mcmc import extract_samples
from gp_active_mcmc.utils.rng import set_seed


# =====================================================================
#  Forward model (pure NumPy, matches model/BeamModel.py exactly)
# =====================================================================

def _build_stiffness_matrix(modulus: np.ndarray, dx: float) -> np.ndarray:
    n = len(modulus)
    E = modulus
    K = np.zeros((n, n))

    for i in range(2, n - 2):
        K[i, i + 2] = E[i]
        K[i, i + 1] = E[i + 1] - 6.0 * E[i] + E[i - 1]
        K[i, i]     = -2.0 * E[i + 1] + 10.0 * E[i] - 2.0 * E[i - 1]
        K[i, i - 1] = E[i + 1] - 6.0 * E[i] + E[i - 1]
        K[i, i - 2] = E[i]

    K[1, 3] = E[1]
    K[1, 2] = E[2] - 6.0 * E[1] + E[0]
    K[1, 1] = -2.0 * E[2] + 11.0 * E[1] - 2.0 * E[0]

    K[n - 2, n - 1] = E[n - 1] - 4.0 * E[n - 2] + E[n - 3]
    K[n - 2, n - 2] = -2.0 * E[n - 1] + 9.0 * E[n - 2] - 2.0 * E[n - 3]
    K[n - 2, n - 3] = E[n - 1] - 6.0 * E[n - 2] + E[n - 3]
    K[n - 2, n - 4] = E[n - 2]

    K[n - 1, n - 1] =  2.0 * E[n - 1]
    K[n - 1, n - 2] = -4.0 * E[n - 1]
    K[n - 1, n - 3] =  2.0 * E[n - 1]

    K[0, :] = 0.0
    K[:, 0] = 0.0
    K[0, 0] = 1.0

    return K / dx**4


def beam_forward_muq(
    theta: np.ndarray,
    x: np.ndarray,
    loads: np.ndarray,
    radius: float,
) -> np.ndarray:
    n = len(x)
    length = float(x[-1] - x[0])
    dx = length / (n - 1)
    I = np.pi / 4.0 * radius**4

    # Build piecewise field using A matrix -- matches benchmark exactly.
    # Boundary nodes (x=L/3, x=2L/3) belong to BOTH adjacent intervals,
    # so their stiffness = exp(m_i) + exp(m_{i+1}).
    n_intervals = 3
    endPts = np.linspace(0, length, n_intervals + 1)
    A_pw = np.zeros((n, n_intervals))
    for i in range(n_intervals):
        A_pw[(x >= endPts[i]) & (x <= endPts[i + 1]), i] = 1.0
    E = A_pw @ np.exp(theta)

    K = _build_stiffness_matrix(E, dx)
    rhs = loads / I
    rhs[0] = 0.0

    return np.linalg.solve(K, rhs)


# =====================================================================
#  Load data from model/ProblemDefinition.h5
# =====================================================================

h5_path = Path(__file__).with_name("model") / "ProblemDefinition.h5"
with h5py.File(h5_path, "r") as f:
    x = np.array(f["/ForwardModel/NodeLocations"]).ravel()
    loads = np.array(f["/ForwardModel/Loads"])
    modulus_true = np.array(f["/ForwardModel/Modulus"])
    u_true_full = np.array(f["/ForwardModel/TrueDisplacement"])
    beam_length = float(f["/ForwardModel"].attrs["BeamLength"])
    beam_radius = float(f["/ForwardModel"].attrs["BeamRadius"])

    B_obs = np.array(f["/Observations/ObservationMatrix"])

n_pts = len(x)
obs_idx = np.sort(np.where(B_obs == 1.0)[1])
n_obs = len(obs_idx)
x_obs = x[obs_idx]

B = np.zeros((n_obs, n_pts))
for j, i in enumerate(obs_idx):
    B[j, i] = 1.0


# =====================================================================
#  Configuration (matching UM-Bridge benchmark)
# =====================================================================

rng = set_seed(2)

# HF forward model: theta -> y_obs
def hf_forward(theta: np.ndarray) -> np.ndarray:
    u = beam_forward_muq(theta, x, loads, beam_radius)
    return B @ u

# Prior: N(10, 4*I)
prior_mean = np.array([10.0, 10.0, 10.0])
prior_cov = 4.0 * np.eye(3)
prior = multivariate_normal(mean=prior_mean, cov=prior_cov)

# Observation noise: benchmark uses noiseVar = 1e-4 (sigma = 0.01)
sigma_obs = 0.01

# Observations (from true heterogeneous field)
y_obs = B @ u_true_full

# MCMC parameters
n_iterations = 50_000
burn_in = 10_000

# Reference theta (mean log-modulus per sub-interval)
xi = x / beam_length
theta_ref = np.array([
    np.mean(np.log(modulus_true)[(xi >= 0.0) & (xi <= 1.0 / 3.0)]),
    np.mean(np.log(modulus_true)[(xi > 1.0 / 3.0) & (xi <= 2.0 / 3.0)]),
    np.mean(np.log(modulus_true)[(xi > 2.0 / 3.0) & (xi <= 1.0)]),
])

print(f"n_obs       = {n_obs}")
print(f"sigma_obs   = {sigma_obs:.3e}")
print(f"theta_ref   = {theta_ref}")
print(f"y_obs range = [{y_obs.min():.4f}, {y_obs.max():.4f}]")


# %% [markdown]
# ## Likelihood and posterior

# %%
cov_obs = (sigma_obs**2) * np.eye(n_obs)
loglike = tda.AdaptiveGaussianLogLike(data=y_obs, covariance=cov_obs)
posterior = tda.Posterior(prior, loglike, hf_forward)


# %% [markdown]
# ## Proposal

# %%
# Start from the MAP estimate (found by optimization)
theta0 = np.array([9.396, 9.419, 10.741])

proposal = AdaptiveMetropolisShared(
    C0=1e-4 * np.eye(3),
    period=100,
    share_across_deepcopy=True,
    adaptive=True,
    sd=1,
)


# %% [markdown]
# ## Run Metropolis-Hastings (HF only)

# %%
print(f"\nRunning MH with {n_iterations} iterations (HF model only, MUQ forward)...")

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

samples = extract_samples(chain=chain_obj, chain_key="chain_0")

chain = MCMCChain.from_arrays(samples=samples)
summary = chain.summary(theta_true=theta_ref, burn_in=burn_in)

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
    axes_trace[i].axhline(theta_ref[i], color="crimson", ls="--", lw=1.2, label="true")
    axes_trace[i].axvline(burn_in, color="grey", ls=":", lw=1.0, label="burn-in")
    axes_trace[i].set_ylabel(labels[i])
    axes_trace[i].legend(loc="upper right", fontsize=8)
axes_trace[-1].set_xlabel("Iteration")
fig_trace.suptitle("Trace plots (HF-only MH, MUQ model)", fontsize=14)
fig_trace.tight_layout()
fig_trace.savefig("plot_hf_muq_trace.png", dpi=150, bbox_inches="tight")
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
    axes_marg[i].axvline(theta_ref[i], color="crimson", ls="--", lw=1.2, label="true")
    axes_marg[i].set_xlabel(labels[i])
    axes_marg[i].set_ylabel("Density" if i == 0 else "")
    axes_marg[i].legend(fontsize=8)
    axes_marg[i].set_title(f"Posterior {labels[i]}")
fig_marg.suptitle("Marginal posteriors (HF-only MH, MUQ model)", fontsize=14)
fig_marg.tight_layout()
fig_marg.savefig("plot_hf_muq_marginals.png", dpi=150, bbox_inches="tight")
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
    theta_true=theta_ref,
    burn_in=burn_in,
    title="Posterior (HF-only MH, MUQ model)",
)
fig_corner.savefig("plot_hf_muq_corner.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nDone. Figures saved: plot_hf_muq_trace.png, plot_hf_muq_marginals.png, plot_hf_muq_corner.png")
