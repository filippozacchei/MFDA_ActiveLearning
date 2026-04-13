from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.linalg import solve

import tinyDA as tda
from scipy.stats import multivariate_normal, gaussian_kde
import matplotlib.pyplot as plt
from gp_active_mcmc.inference import AdaptiveMetropolisShared
from gp_active_mcmc.inference.chain import MCMCChain
from gp_active_mcmc.utils.mcmc import extract_samples
from gp_active_mcmc.utils.rng import set_seed


def make_spatial_grid(n_pts: int = 31, length: float = 1.0) -> np.ndarray:
    """Return the 1D spatial grid for the beam."""
    return np.linspace(0.0, length, n_pts)


def build_piecewise_logE(theta: np.ndarray, x: np.ndarray, length: float = 1.0) -> np.ndarray:
    """
    Piecewise-constant log-stiffness field on 3 equal subintervals.

    theta = [m1, m2, m3]
    """
    theta = np.asarray(theta, dtype=float).reshape(-1)
    if theta.shape != (3,):
        raise ValueError("theta must have shape (3,)")

    xi = x / length
    logE = np.empty_like(x)

    logE[(xi >= 0.0) & (xi <= 1.0 / 3.0)] = theta[0]
    logE[(xi > 1.0 / 3.0) & (xi <= 2.0 / 3.0)] = theta[1]
    logE[(xi > 2.0 / 3.0) & (xi <= 1.0)] = theta[2]

    return logE


def build_modulus_from_theta(theta: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Build the nodal modulus field E(x) from the low-dimensional parameter theta.
    """
    length = float(x[-1] - x[0])
    logE = build_piecewise_logE(theta, x, length=length)
    E = np.exp(logE)
    return E


def build_beam_operator(E: np.ndarray, dx: float) -> np.ndarray:
    """
    Build the stiffness matrix K exactly following the MUQ BeamModel.py logic.

    This mirrors the first code:
        - variable modulus E on nodes
        - cantilever beam
        - FD stencil built directly row by row
        - left Dirichlet BC enforced on row/column 0

    Parameters
    ----------
    E : ndarray, shape (n,)
        Nodal modulus values.
    dx : float
        Grid spacing.

    Returns
    -------
    K : ndarray, shape (n, n)
        Beam stiffness matrix.
    """
    E = np.asarray(E, dtype=float).reshape(-1)
    n = E.size

    if n < 5:
        raise ValueError("Need at least 5 grid points for this stencil.")

    K = np.zeros((n, n), dtype=float)

    # Interior rows: i = 2, ..., n-3
    for i in range(2, n - 2):
        K[i, i + 2] = E[i]
        K[i, i + 1] = E[i + 1] - 6.0 * E[i] + E[i - 1]
        K[i, i] = -2.0 * E[i + 1] + 10.0 * E[i] - 2.0 * E[i - 1]
        K[i, i - 1] = E[i + 1] - 6.0 * E[i] + E[i - 1]
        K[i, i - 2] = E[i]

    # Row i = 1
    K[1, 3] = E[1]
    K[1, 2] = E[2] - 6.0 * E[1] + E[0]
    K[1, 1] = -2.0 * E[2] + 11.0 * E[1] - 2.0 * E[0]

    # Row i = n-2
    K[n - 2, n - 1] = E[n - 1] - 4.0 * E[n - 2] + E[n - 3]
    K[n - 2, n - 2] = -2.0 * E[n - 1] + 9.0 * E[n - 2] - 2.0 * E[n - 3]
    K[n - 2, n - 3] = E[n - 1] - 6.0 * E[n - 2] + E[n - 3]
    K[n - 2, n - 4] = E[n - 2]

    # Last row: i = n-1
    K[n - 1, n - 1] = 2.0 * E[n - 1]
    K[n - 1, n - 2] = -4.0 * E[n - 1]
    K[n - 1, n - 3] = 2.0 * E[n - 1]

    # Dirichlet BC at x = 0: u(0) = 0
    K[0, :] = 0.0
    K[:, 0] = 0.0
    K[0, 0] = 1.0

    return K / dx**4


def build_load_vector(
    x: np.ndarray,
    load: float | np.ndarray = -1.0,
    radius: float = 0.1,
) -> np.ndarray:
    """
    Build the distributed load vector, matching the first code logic.

    In the MUQ code:
        load_eff = load / I
    where I = pi/4 * radius^4.

    Parameters
    ----------
    x : ndarray
        Spatial grid.
    load : float or ndarray
        Scalar load (uniform) or nodal load vector.
    radius : float
        Beam radius for cylindrical cross-section.

    Returns
    -------
    rhs : ndarray
        Right-hand side load vector already divided by I.
    """
    I = np.pi / 4.0 * radius**4

    load_arr = np.asarray(load, dtype=float)
    if load_arr.ndim == 0:
        rhs = load_arr.item() * np.ones_like(x, dtype=float)
    else:
        if load_arr.shape != x.shape:
            raise ValueError(f"load shape {load_arr.shape} != x shape {x.shape}")
        rhs = load_arr.copy()

    rhs = rhs / I

    # Apply Dirichlet BC on load vector as in the first code
    rhs[0] = 0.0

    return rhs


def beam_forward(
    theta: np.ndarray,
    x: np.ndarray,
    load: float | np.ndarray = -1.0,
    radius: float = 0.1,
) -> np.ndarray:
    """
    Forward model in function form, but numerically matching the first MUQ code.

    Parameters
    ----------
    theta : ndarray, shape (3,)
        Low-dimensional parameters defining piecewise-constant log(E).
    x : ndarray, shape (n_pts,)
        Spatial grid.
    load : float or ndarray
        Distributed load.
    radius : float
        Beam radius.

    Returns
    -------
    u : ndarray, shape (n_pts,)
        Beam displacement on the grid.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    dx = x[1] - x[0]

    E = build_modulus_from_theta(theta, x)
    K = build_beam_operator(E, dx)
    rhs = build_load_vector(x, load=load, radius=radius)

    u = solve(K, rhs)
    return u


def make_observation_operator(n_pts: int, obs_idx: np.ndarray) -> np.ndarray:
    """Build observation matrix B selecting entries of the state vector."""
    obs_idx = np.asarray(obs_idx, dtype=int).reshape(-1)

    if np.any(obs_idx < 0) or np.any(obs_idx >= n_pts):
        raise ValueError("obs_idx contains invalid indices.")

    B = np.zeros((len(obs_idx), n_pts), dtype=float)
    for j, i in enumerate(obs_idx):
        B[j, i] = 1.0

    return B


def make_forward_model(
    x: np.ndarray,
    obs_idx: np.ndarray | None = None,
    load: float | np.ndarray = -1.0,
    radius: float = 0.1,
    return_full_state: bool = False,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Wrap the beam forward model with the spatial grid baked in.

    If return_full_state is False, returns y = B @ u.
    If return_full_state is True, returns the full displacement u.
    """
    x = np.asarray(x, dtype=float).reshape(-1)

    if not return_full_state:
        if obs_idx is None:
            raise ValueError("obs_idx must be provided when return_full_state=False")
        B = make_observation_operator(len(x), obs_idx)

    def _forward(theta: np.ndarray) -> np.ndarray:
        u = beam_forward(theta, x=x, load=load, radius=radius)
        if return_full_state:
            return u
        return B @ u

    return _forward


def make_observation(
    rng: np.random.Generator,
    theta_true: np.ndarray,
    x: np.ndarray,
    sigma_obs: float,
    obs_idx: np.ndarray,
    load: float | np.ndarray = -1.0,
    radius: float = 0.1,
) -> np.ndarray:
    """
    Generate noisy synthetic observations:
        y_obs = B @ u(theta_true) + noise
    """
    forward = make_forward_model(
        x=x,
        obs_idx=obs_idx,
        load=load,
        radius=radius,
        return_full_state=False,
    )
    y_clean = forward(theta_true)
    y_obs = y_clean + rng.normal(0.0, sigma_obs, size=y_clean.shape)
    return y_obs


if __name__ == "__main__":

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
        x=x, obs_idx=obs_idx, load=-loads, radius=0.1, return_full_state=False,
    )

    # Prior over theta = [m1, m2, m3]  (log-stiffness on three sub-intervals)
    prior_mean = np.array([10.0, 10.0, 10.0])
    prior_cov = np.diag([2.0**2, 2.0**2, 2.0**2])
    prior = multivariate_normal(mean=prior_mean, cov=prior_cov)

    # Observation noise -- auto-scaled from a reference forward evaluation
    y_ref = hf_forward(prior_mean)
    signal_scale = float(np.max(np.abs(y_ref)))
    sigma_obs = 0.01 * signal_scale  # 1 % relative noise

    # MCMC parameters
    n_iterations = 100_000
    burn_in = 50_000

    print(f"signal_scale = {signal_scale:.3e}")
    print(f"sigma_obs    = {sigma_obs:.3e}")


    # =====================================================================
    #  Synthetic observation
    # =====================================================================

    theta_true = np.array([9.3, 9.3, 9.2])
    y_obs = make_observation(rng, theta_true, x, sigma_obs, obs_idx, load=-loads, radius=0.1)

    print(f"theta_true = {theta_true}")
    print(f"y_obs      = {y_obs}")


    # =====================================================================
    #  Beam displacement & setup overview
    # =====================================================================

    # Full-state forward model for plotting
    hf_full = make_forward_model(
        x=x, load=-loads, radius=0.1, return_full_state=True,
    )
    u_true = hf_full(theta_true)
    u_prior = hf_full(prior_mean)

    fig_beam, axes_beam = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    # --- Panel 1: displacement ---
    axes_beam[0].plot(x, u_true, "k-", lw=2, label=r"$u(\theta_{\rm true})$")
    axes_beam[0].plot(x, u_prior, "b--", lw=1.5, label=r"$u(\theta_{\rm prior})$")
    axes_beam[0].scatter(x_obs, y_obs, color="crimson", marker="o", s=40,
                         zorder=5, label="noisy obs")
    axes_beam[0].set_ylabel("Displacement  $u(x)$")
    axes_beam[0].legend(fontsize=9)
    axes_beam[0].set_title("Beam displacement", fontsize=13)
    axes_beam[0].grid(True, ls=":", alpha=0.5)

    # --- Panel 2: distributed load ---
    axes_beam[1].plot(x, loads, "g-", lw=2, label="load $q(x)$")
    axes_beam[1].set_ylabel("Load")
    axes_beam[1].legend(fontsize=9)
    axes_beam[1].set_title("Distributed load along beam", fontsize=13)
    axes_beam[1].grid(True, ls=":", alpha=0.5)

    # --- Panel 3: stiffness field E(x) ---
    E_true = build_modulus_from_theta(theta_true, x)
    E_prior = build_modulus_from_theta(prior_mean, x)
    axes_beam[2].plot(x, E_true, "k-", lw=2, label=r"$E(\theta_{\rm true})$")
    axes_beam[2].plot(x, E_prior, "b--", lw=1.5, label=r"$E(\theta_{\rm prior})$")
    axes_beam[2].set_ylabel("Modulus  $E(x)$")
    axes_beam[2].set_xlabel("$x$")
    axes_beam[2].legend(fontsize=9)
    axes_beam[2].set_title("Stiffness field (piecewise-constant)", fontsize=13)
    axes_beam[2].grid(True, ls=":", alpha=0.5)

    fig_beam.tight_layout()
    fig_beam.savefig("plot_beam_overview.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: plot_beam_overview.png")


    # =====================================================================
    #  Standard Gaussian log-likelihood (no variance inflation)
    # =====================================================================

    cov_obs = (sigma_obs**2) * np.eye(len(y_obs))
    loglike = tda.AdaptiveGaussianLogLike(data=y_obs, covariance=cov_obs)


    # =====================================================================
    #  Build tinyDA Posterior (single-level, HF only)
    # =====================================================================

    posterior = tda.Posterior(prior, loglike, hf_forward)


    # =====================================================================
    #  Proposal distribution
    # =====================================================================

    theta0 = theta_true

    proposal = AdaptiveMetropolisShared(
        C0=0.001 * prior_cov,
        period=100,
        share_across_deepcopy=True,
        adaptive=True,
        sd=1,
    )


    # =====================================================================
    #  Run Metropolis-Hastings (HF only)
    # =====================================================================

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


    # =====================================================================
    #  Trace plots
    # =====================================================================

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


    # =====================================================================
    #  Marginal posterior distributions
    # =====================================================================

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


    # =====================================================================
    #  Corner plot
    # =====================================================================

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