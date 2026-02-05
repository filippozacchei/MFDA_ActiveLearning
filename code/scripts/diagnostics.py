from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_hf_cumulative_fraction(
    used_hf_flags: np.ndarray,
    *,
    burnin: int = 0,
    max_points: int | None = None,
) -> None:
    """Plot cumulative high-fidelity call fraction."""
    if used_hf_flags.size == 0:
        return

    flags = used_hf_flags.astype(float)
    if burnin > 0:
        flags = flags[burnin:]
    if max_points is not None:
        flags = flags[:max_points]

    if flags.size == 0:
        return

    steps = np.arange(1, len(flags) + 1)
    frac = np.cumsum(flags) / steps

    plt.figure(figsize=(8, 4))
    plt.plot(steps, frac)
    plt.xlabel("Iteration")
    plt.ylabel("Cumulative HF-call fraction")
    plt.title("High-fidelity usage")
    plt.grid(True)
    plt.show()


def plot_subchain_length(
    subchain_history: list[int],
) -> None:
    """Plot adaptive subchain length evolution."""
    if not subchain_history:
        return

    plt.figure(figsize=(8, 4))
    plt.plot(subchain_history)
    plt.xlabel("Coarse iteration")
    plt.ylabel("Subchain length")
    plt.title("Adaptive subchain length")
    plt.grid(True)
    plt.show()


def plot_surrogate_error(
    hf_errors: list[float],
    *,
    target_error: float | None = None,
) -> None:
    """Plot HF–surrogate error history."""
    if not hf_errors:
        return

    plt.figure(figsize=(8, 4))
    plt.plot(hf_errors, label="Surrogate error")

    if target_error is not None:
        plt.axhline(
            target_error,
            linestyle="--",
            label="Target error",
        )

    plt.xlabel("Fine iteration")
    plt.ylabel("Error")
    plt.title("HF–surrogate error")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_chain_2d(
    chain: np.ndarray,
    used_forward: np.ndarray,
    theta_true: np.ndarray,
    title: str = None,
    names: tuple[str, str] = ("A", "f"),
):
    """2D visualization of MCMC chain."""
    fig, ax = plt.subplots(figsize=(6, 5))

    if used_forward is not None:
        gp_idx = np.where(~used_forward)[0]
        fw_idx = np.where(used_forward)[0]

        if gp_idx.size:
            ax.scatter(chain[gp_idx, 0], chain[gp_idx, 1], s=15, alpha=0.4, label="GP")

        if fw_idx.size:
            ax.scatter(
                chain[fw_idx, 0], chain[fw_idx, 1], s=40, marker="x", label="Forward"
            )
    else:
        ax.scatter(chain[:, 0], chain[:, 1], s=40)

    ax.scatter(
        theta_true[0],
        theta_true[1],
        s=140,
        marker="*",
        c="k",
        label=r"$\theta_{\mathrm{true}}$",
    )

    ax.set_xlabel(names[0])
    ax.set_ylabel(names[1])
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    plt.show()


def plot_prediction_at_theta(
    emul, theta: np.ndarray, t: np.ndarray, y_obs: np.ndarray, title: str = None
):
    """Surrogate prediction with ±2σ band."""
    y_hat, y_std = emul.predict(theta)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, y_obs, "k.", alpha=0.4, label="observations")
    ax.plot(t, y_hat, lw=2, label="surrogate mean")
    ax.fill_between(
        t,
        y_hat - 2.0 * y_std,
        y_hat + 2.0 * y_std,
        alpha=0.25,
        label=r"$\pm 2\sigma$",
    )
    ax.set_xlabel("t")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    plt.show()


def binned_reliability(
    mean_std: np.ndarray,
    rmse_vals: np.ndarray,
    n_bins: int = 5,
):
    """Binned reliability curve."""
    idx = np.argsort(mean_std)
    u_sorted = mean_std[idx]
    e_sorted = rmse_vals[idx]

    bins = np.array_split(np.arange(len(mean_std)), n_bins)
    u_bin = [u_sorted[b].mean() for b in bins]
    e_bin = [e_sorted[b].mean() for b in bins]

    plt.figure()
    plt.plot(u_bin, e_bin, marker="o")
    plt.xlabel("Mean predicted uncertainty (bin)")
    plt.ylabel("Mean RMSE (bin)")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True)
    plt.show()
