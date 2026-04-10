# %% [markdown]
# # Backward beam: Bayesian inversion using the MUQ-compatible model
#
# Uses the beam physics from ``model/BeamModel.py`` (including moment of inertia
# and the correct FD stencil) and observations from ``model/ProblemDefinition.h5``.
#
# The forward model is reimplemented in pure NumPy so MUQ is **not** required.
#
# Inference parameters: theta = [m1, m2, m3] (piecewise-constant log-stiffness
# on three equal sub-intervals).  Two inference modes:
#
# 1. **MCMC-guided active learning (single posterior)**
# 2. **DA-MCMC with adaptive subchain (recommended)**

# %% Imports
from __future__ import annotations

import copy
from pathlib import Path

import h5py
import numpy as np
import tinyDA as tda
from scipy.stats import multivariate_normal, gaussian_kde

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
#  Forward model (pure NumPy, matches model/BeamModel.py)
# =====================================================================

def _build_stiffness_matrix(modulus: np.ndarray, dx: float) -> np.ndarray:
    """Build the beam stiffness matrix using the same FD stencil as BeamModel.py.

    This discretises  (E·u'')''  with cantilever BCs:
        u(0) = 0,  u'(0) = 0   (fixed left end)
        u''(L) = 0, u'''(L) = 0 (free right end)
    """
    n = len(modulus)
    E = modulus
    K = np.zeros((n, n))

    # Interior rows  (i = 2 … n-3)
    for i in range(2, n - 2):
        K[i, i + 2] = E[i]
        K[i, i + 1] = E[i + 1] - 6.0 * E[i] + E[i - 1]
        K[i, i]     = -2.0 * E[i + 1] + 10.0 * E[i] - 2.0 * E[i - 1]
        K[i, i - 1] = E[i + 1] - 6.0 * E[i] + E[i - 1]
        K[i, i - 2] = E[i]

    # Row i = 1  (u'(0) = 0 absorbed)
    K[1, 3] = E[1]
    K[1, 2] = E[2] - 6.0 * E[1] + E[0]
    K[1, 1] = -2.0 * E[2] + 11.0 * E[1] - 2.0 * E[0]

    # Row i = n-2
    K[n - 2, n - 1] = E[n - 1] - 4.0 * E[n - 2] + E[n - 3]
    K[n - 2, n - 2] = -2.0 * E[n - 1] + 9.0 * E[n - 2] - 2.0 * E[n - 3]
    K[n - 2, n - 3] = E[n - 1] - 6.0 * E[n - 2] + E[n - 3]
    K[n - 2, n - 4] = E[n - 2]

    # Row i = n-1  (free-end BCs)
    K[n - 1, n - 1] =  2.0 * E[n - 1]
    K[n - 1, n - 2] = -4.0 * E[n - 1]
    K[n - 1, n - 3] =  2.0 * E[n - 1]

    # Dirichlet BC:  u(0) = 0
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
    """Beam forward model matching model/BeamModel.py.

    Parameters
    ----------
    theta : (3,) log-stiffness on 3 equal sub-intervals
    x     : (n_pts,) spatial grid
    loads : (n_pts,) distributed load
    radius: beam radius (for moment of inertia I = π/4 · r⁴)

    Returns
    -------
    u : (n_pts,) displacement
    """
    n = len(x)
    length = float(x[-1] - x[0])
    dx = length / (n - 1)
    I = np.pi / 4.0 * radius**4

    # Piecewise-constant log-stiffness → nodal modulus
    # Build A matrix matching benchmark (boundary nodes in both intervals)
    n_intervals = 3
    endPts = np.linspace(0, length, n_intervals + 1)
    A_pw = np.zeros((n, n_intervals))
    for i in range(n_intervals):
        A_pw[(x >= endPts[i]) & (x <= endPts[i + 1]), i] = 1.0
    E = A_pw @ np.exp(theta)

    K = _build_stiffness_matrix(E, dx)
    rhs = loads / I
    rhs[0] = 0.0  # Dirichlet BC

    return np.linalg.solve(K, rhs)


# =====================================================================
#  Direct GP surrogate (same as backward.py)
# =====================================================================

class DirectGPSurrogate:
    """GP surrogate mapping theta -> observed displacements."""

    def __init__(self, gp: MultiOutputGP):
        self.gp = gp

    def predict(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        theta = np.asarray(theta, dtype=float).reshape(1, -1)
        y_mean, y_var = self.gp.predict(theta)
        return y_mean[0], np.maximum(y_var[0], 0.0)

    def update(self, theta: np.ndarray, y_hf: np.ndarray) -> None:
        self.gp.update(
            np.asarray(theta, dtype=float).ravel(),
            np.asarray(y_hf, dtype=float).ravel(),
        )


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
    y_obs_clean = np.array(f["/Observations/ObservationData"])

n_pts = len(x)
obs_idx = np.sort(np.where(B_obs == 1.0)[1])  # sorted observation indices
n_obs = len(obs_idx)
x_obs = x[obs_idx]

# Rebuild B in sorted order for consistency
B = np.zeros((n_obs, n_pts))
for j, i in enumerate(obs_idx):
    B[j, i] = 1.0


# =====================================================================
#  Configuration
# =====================================================================

rng = set_seed(2)

# HF forward model: theta -> y_obs (observed displacements)
def hf_forward(theta: np.ndarray) -> np.ndarray:
    u = beam_forward_muq(theta, x, loads, beam_radius)
    return B @ u

# Prior over theta = [m1, m2, m3]
# UM-Bridge benchmark: m_i ~ N(10, 4)  i.e. mean=10, cov=4*I (sigma=2)
prior_mean = np.array([10.0, 10.0, 10.0])
prior_cov = 4.0 * np.eye(3)  # variance = 4, sigma = 2
prior = multivariate_normal(mean=prior_mean, cov=prior_cov)

# Observation noise: benchmark uses noiseVar = 1e-4 (sigma = 0.01)
sigma_obs = 0.01
y_ref = hf_forward(prior_mean)
signal_scale = float(np.max(np.abs(y_ref)))

# Observations: extract from true displacement using the *sorted* observation matrix B
# (y_obs_clean from HDF5 uses the original unsorted row order of B_obs, which does not
# match our sorted obs_idx / B.  Recomputing avoids the permutation mismatch.)
y_obs = B @ u_true_full

# Surrogate configuration
n_init = 200
gp_kernel = "matern52"
gp_ard = True

# Active coupling thresholds
gamma_threshold_single = 0.1 * sigma_obs
gamma_threshold_da = 0.1 * sigma_obs

# MCMC budget
n_coarse_evals = 2000
n_coarse_evals_da = 2000
burn_in = 500
chunk_size = 500

print(f"n_obs                    = {n_obs}")
print(f"obs_idx                  = {obs_idx}")
print(f"signal_scale             = {signal_scale:.3e}")
print(f"sigma_obs                = {sigma_obs:.3e}")
print(f"gamma_threshold (single) = {gamma_threshold_single:.3e}")
print(f"gamma_threshold (DA)     = {gamma_threshold_da:.3e}")

# "True" piecewise-constant approximation of log(modulus_true)
# (average log-modulus per sub-interval, for reference only)
xi = x / beam_length
theta_ref = np.array([
    np.mean(np.log(modulus_true)[(xi >= 0.0) & (xi <= 1.0 / 3.0)]),
    np.mean(np.log(modulus_true)[(xi > 1.0 / 3.0) & (xi <= 2.0 / 3.0)]),
    np.mean(np.log(modulus_true)[(xi > 2.0 / 3.0) & (xi <= 1.0)]),
])
print(f"\ntheta_ref (mean log-modulus per interval) = {theta_ref}")
print(f"y_obs range = [{y_obs.min():.4f}, {y_obs.max():.4f}]")


# %% [markdown]
# ## Initial surrogate training set

# %%
# Sample initial training points from a TIGHTER distribution than the prior.
# With prior N(10,4) (sigma=2), exp(theta) spans 5 orders of magnitude, which
# makes the GP ill-conditioned.  A tighter training range keeps outputs in a
# manageable range; the active-learning loop will add points outside as needed.
train_dist = multivariate_normal(mean=prior_mean, cov=0.5 * np.eye(3))  # sigma≈0.7
theta_train = np.asarray(
    [train_dist.rvs(random_state=rng) for _ in range(n_init)], dtype=float,
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
    noise_variance=1e-6,
    update_every=50,
    n_retrain_max=2,
)

lf_surrogate_single = DirectGPSurrogate(gp=copy.deepcopy(gp))
lf_surrogate_adapt = DirectGPSurrogate(gp=copy.deepcopy(gp))


# %% [markdown]
# ## Wrap LF + HF in an ActiveMCMCModel

# %%
model_single = ActiveMCMCModel(
    lf_model=lf_surrogate_single,
    hf_model=hf_forward,
    gamma_threshold=gamma_threshold_single,
)

adaptive_policy = AdaptiveSubchain(
    state=AdaptiveSubchainState(subchain_length=25),
    control=AdaptiveSubchainControl(
        update_every=10,
        target_error=0.05,
        min_subchain=10,
        max_subchain=500,
        grow_factor=2,
        shrink_factor=0.5,
    ),
)

model_adapt = ActiveMCMCModel(
    lf_model=lf_surrogate_adapt,
    hf_model=hf_forward,
    gamma_threshold=gamma_threshold_da,
    adaptive=adaptive_policy,
)


# %% [markdown]
# ## Likelihood and posterior objects

# %%
cov = (sigma_obs**2) * np.eye(n_obs)

loglike_coarse = ActiveGPLogLike(data=y_obs, covariance=cov)
loglike_fine = tda.AdaptiveGaussianLogLike(data=y_obs, covariance=cov)

posterior_single = tda.Posterior(prior, loglike_coarse, model_single.coarse)

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
# ## Sanity check: HF forward model and observations

# %%
# Plot the full beam displacement (HF model) and the observations
u_full_ref = beam_forward_muq(theta_ref, x, loads, beam_radius)

fig_beam, ax_beam = plt.subplots(figsize=(9, 4))
ax_beam.plot(x, u_full_ref, "b-", lw=2, label="HF displacement (piecewise $\\theta_{ref}$)")
ax_beam.plot(x, u_true_full, "g--", lw=1.5, label="HF displacement (true modulus)")
ax_beam.plot(x_obs, y_obs, "ro", ms=5, label="Noisy observations")
ax_beam.set_xlabel("x")
ax_beam.set_ylabel("Displacement u(x)")
ax_beam.set_title("Beam deformation: HF model vs observations")
ax_beam.legend()
ax_beam.grid(True, alpha=0.3)
fig_beam.tight_layout()
fig_beam.savefig("plot_muq_beam_deformation.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Surrogate prediction before sampling

# %%
plot_prediction_at_theta(
    model=lf_surrogate_single,
    theta=theta_ref,
    t=x_obs,
    y_obs=y_obs,
    y_true=hf_forward(theta_ref),
    title="Surrogate prediction at theta_ref (before sampling)",
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
chain_single.summary(theta_true=theta_ref, burn_in=burn_in)

# %%
samples_single = chain_single.samples
used_hf_single = chain_single.extras.used_hf

fig1, ax1 = plot_chain_2d(
    samples_single[:, :2],
    used_hf=used_hf_single,
    theta_true=theta_ref[:2],
    title="Single posterior: samples (m1 vs m2)",
    show=True,
)
fig1.savefig("plot_muq_single_chain2d.png", dpi=150, bbox_inches="tight")

fig2, ax2 = plot_cumulative_hf_fraction(
    used_hf_single,
    title="Single posterior: cumulative HF fraction",
    show=True,
)
fig2.savefig("plot_muq_single_hf_fraction.png", dpi=150, bbox_inches="tight")


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
chain_adapt.summary(theta_true=theta_ref, burn_in=burn_in)

# %%
samples_adapt = chain_adapt.samples
used_hf_adapt = chain_adapt.extras.used_hf

fig3, ax3 = plot_chain_2d(
    samples_adapt[:, :2],
    used_hf=used_hf_adapt,
    theta_true=theta_ref[:2],
    title="Adaptive DA-MCMC: samples (m1 vs m2)",
    show=True,
)
fig3.savefig("plot_muq_adapt_chain2d.png", dpi=150, bbox_inches="tight")

fig4, ax4 = plot_cumulative_hf_fraction(
    used_hf_adapt,
    title="Adaptive DA-MCMC: cumulative HF fraction",
    show=True,
)
fig4.savefig("plot_muq_adapt_hf_fraction.png", dpi=150, bbox_inches="tight")

if chain_adapt.extras.subchain_length is not None:
    fig5, ax5 = plot_subchain_length_history(
        chain_adapt.extras.subchain_length, show=True,
    )
    fig5.savefig("plot_muq_adapt_subchain.png", dpi=150, bbox_inches="tight")


# %% [markdown]
# ## Post-sampling: surrogate prediction at theta_ref

# %%
plot_prediction_at_theta(
    model=lf_surrogate_adapt,
    theta=theta_ref,
    t=x_obs,
    y_obs=y_obs,
    y_true=hf_forward(theta_ref),
    title="Surrogate prediction at theta_ref (after DA-MCMC)",
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


labels = [r"$m_0$", r"$m_1$", r"$m_2$"]

fig_c1, _ = corner_plot(
    samples_adapt,
    labels=labels,
    theta_true=theta_ref,
    burn_in=burn_in,
    title="DA-MCMC posterior (MUQ-compatible model)",
)
fig_c1.savefig("plot_muq_corner_da_mcmc.png", dpi=150, bbox_inches="tight")
plt.show()

fig_c2, _ = corner_plot(
    samples_single,
    labels=labels,
    theta_true=theta_ref,
    burn_in=burn_in,
    title="Single posterior MCMC (MUQ-compatible model)",
)
fig_c2.savefig("plot_muq_corner_single.png", dpi=150, bbox_inches="tight")
plt.show()
