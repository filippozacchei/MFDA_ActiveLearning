from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from numpy.typing import ArrayLike


def pod_energy_from_snapshots(Y: ArrayLike, *, center: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Return (per-mode energy fraction, cumulative energy) from snapshot matrix."""
    Y_arr = np.asarray(Y, dtype=float)
    if Y_arr.ndim != 2:
        raise ValueError(f"Y must be 2D (n_snapshots, n_time). Got shape {Y_arr.shape}.")

    X = Y_arr - Y_arr.mean(axis=0, keepdims=True) if center else Y_arr
    _, S, _ = np.linalg.svd(X, full_matrices=False)
    lam = S**2
    frac = lam / np.sum(lam) if lam.size else lam
    return frac, np.cumsum(frac)


def plot_pod_energy(
    Y: ArrayLike,
    *,
    r_max: int = 50,
    center: bool = True,
    thresholds: tuple[float, ...] = (0.90, 0.95, 0.99),
) -> tuple[tuple[Figure, Axes], tuple[Figure, Axes]]:
    """Plot per-mode and cumulative POD energy curves."""
    if r_max <= 0:
        raise ValueError("r_max must be positive.")

    e_frac, e_cum = pod_energy_from_snapshots(Y, center=center)
    m = min(len(e_frac), r_max)

    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.semilogy(range(1, m + 1), e_frac[:m], marker="o")
    ax1.set_xlabel("Mode index")
    ax1.set_ylabel("Energy fraction")
    ax1.set_title("POD energy per mode")
    ax1.grid(True)
    fig1.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot(range(1, m + 1), e_cum[:m], marker="o")
    for th in thresholds:
        if not (0.0 < th <= 1.0):
            raise ValueError("thresholds must be in (0, 1].")
        r_th = int(np.searchsorted(e_cum, th) + 1)
        ax2.axhline(th, linestyle="--", linewidth=1.0)
        ax2.axvline(min(r_th, m), linestyle="--", linewidth=1.0)
    ax2.set_xlabel("Mode index")
    ax2.set_ylabel("Cumulative energy")
    ax2.set_title("Cumulative POD energy")
    ax2.grid(True)
    fig2.tight_layout()

    return (fig1, ax1), (fig2, ax2)
