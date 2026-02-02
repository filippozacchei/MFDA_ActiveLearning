# =========================
# Plotting and diagnostics utilities
# =========================

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

EPS = 1e-14


# ---------------------------------------------------------------------
# Basic metrics
# ---------------------------------------------------------------------

def rmse(y_hat: np.ndarray, y_true: np.ndarray) -> float:
    """Root-mean-square error."""
    return float(np.sqrt(np.mean((y_hat - y_true) ** 2)))


def coverage(
    y_true: np.ndarray,
    y_hat: np.ndarray,
    y_std: np.ndarray,
    z: float,
) -> float:
    """Empirical coverage probability."""
    lo = y_hat - z * y_std
    hi = y_hat + z * y_std
    return float(np.mean((y_true >= lo) & (y_true <= hi)))


# ---------------------------------------------------------------------
# Design space visualization
# ---------------------------------------------------------------------

def plot_pair_scatter_train_test(
    X_tr: np.ndarray,
    X_te: np.ndarray,
    names: tuple[str, ...],
):
    """Pairwise 2D scatter plots of training and test points."""
    pairs = [(0, 1)]

    for i, j in pairs:
        plt.figure()
        plt.scatter(X_tr[:, i], X_tr[:, j], s=18, alpha=0.6, label="train")
        plt.scatter(X_te[:, i], X_te[:, j], s=22, alpha=0.8, label="test")
        plt.xlabel(names[i])
        plt.ylabel(names[j])
        plt.title(f"{names[i]} vs {names[j]}")
        plt.grid(True)
        plt.legend()
        plt.show()


# ---------------------------------------------------------------------
# POD diagnostics
# ---------------------------------------------------------------------

def pod_energy_from_snapshots(
    Y: np.ndarray,
    center: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute POD energy fractions from snapshot matrix."""
    X = Y - Y.mean(axis=0, keepdims=True) if center else Y
    _, S, _ = np.linalg.svd(X, full_matrices=False)

    lam = S**2
    frac = lam / np.sum(lam)
    return frac, np.cumsum(frac)


def plot_pod_energy_curves(
    Y_tr: np.ndarray,
    r_max: int = 50,
    center: bool = True,
    thresholds: tuple[float, ...] = (0.90, 0.95, 0.99),
):
    """Plot per-mode and cumulative POD energy."""
    e_frac, e_cum = pod_energy_from_snapshots(Y_tr, center=center)
    m = min(len(e_frac), r_max)

    plt.figure()
    plt.semilogy(range(1, m + 1), e_frac[:m], marker="o")
    plt.xlabel("Mode index")
    plt.ylabel("Energy fraction")
    plt.title("POD energy per mode")
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(range(1, m + 1), e_cum[:m], marker="o")

    for th in thresholds:
        r_th = int(np.searchsorted(e_cum, th) + 1)
        plt.axhline(th, linestyle="--")
        plt.axvline(r_th, linestyle=":")
        plt.text(r_th, th, f" r={r_th}", va="bottom")

    plt.ylim(0.0, 1.01)
    plt.xlabel("Rank r")
    plt.ylabel("Cumulative energy")
    plt.title("Cumulative POD energy")
    plt.grid(True)
    plt.show()


def plot_pod_reconstruction_error_vs_rank(
    Y_tr: np.ndarray,
    Y_te: np.ndarray,
    r_list: list[int],
    center: bool = True,
):
    """POD-only reconstruction RMSE vs rank."""
    Y_mean = Y_tr.mean(axis=0, keepdims=True) if center else 0.0
    X = Y_tr - Y_mean if center else Y_tr

    _, _, Vt = np.linalg.svd(X, full_matrices=False)

    def reconstruct(Y: np.ndarray, r: int) -> np.ndarray:
        Xy = Y - Y_mean if center else Y
        Vr = Vt[:r].T
        return (Xy @ Vr) @ Vr.T + (Y_mean if center else 0.0)

    rmse_tr, rmse_te = [], []

    for r in r_list:
        rmse_tr.append(rmse(reconstruct(Y_tr, r), Y_tr))
        rmse_te.append(rmse(reconstruct(Y_te, r), Y_te))

    plt.figure()
    plt.plot(r_list, rmse_tr, marker="o", label="train")
    plt.plot(r_list, rmse_te, marker="o", label="test")
    plt.xlabel("Rank r")
    plt.ylabel("RMSE")
    plt.title("POD-only reconstruction error")
    plt.grid(True)
    plt.legend()
    plt.show()


# ---------------------------------------------------------------------
# Uncertainty diagnostics
# ---------------------------------------------------------------------

def plot_error_vs_uncertainty(
    mean_std: np.ndarray,
    rmse_vals: np.ndarray,
):
    """Error vs predicted uncertainty."""
    plt.figure()
    plt.scatter(mean_std, rmse_vals, alpha=0.85)
    plt.xlabel("Mean predictive standard deviation")
    plt.ylabel("RMSE")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True)
    plt.show()

    corr = np.corrcoef(mean_std, rmse_vals)[0, 1]
    print("Correlation =", float(corr))


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


# ---------------------------------------------------------------------
# Time-series prediction
# ---------------------------------------------------------------------

def plot_prediction_at_theta(
    emul,
    theta: np.ndarray,
    t: np.ndarray,
    y_obs: np.ndarray,
    title: str = None
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


# ---------------------------------------------------------------------
# MCMC diagnostics
# ---------------------------------------------------------------------

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
            ax.scatter(chain[gp_idx, 0], chain[gp_idx, 1],
                    s=15, alpha=0.4, label="GP")

        if fw_idx.size:
            ax.scatter(chain[fw_idx, 0], chain[fw_idx, 1],
                    s=40, marker="x", label="Forward")
    else: 
        ax.scatter(chain[:, 0], chain[:, 1], s=40)

    ax.scatter(theta_true[0], theta_true[1],
               s=140, marker="*", c="k", label=r"$\theta_{\mathrm{true}}$")

    ax.set_xlabel(names[0])
    ax.set_ylabel(names[1])
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    plt.show()
