from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .metrics import rmse


def pod_energy_from_snapshots(
    Y: np.ndarray,
    *,
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
    *,
    r_max: int = 50,
    center: bool = True,
    thresholds: tuple[float, ...] = (0.90, 0.95, 0.99),
) -> None:
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
    *,
    center: bool = True,
) -> None:
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
