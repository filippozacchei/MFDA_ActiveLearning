from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def pod_energy_from_snapshots(
    Y: ArrayLike,
    *,
    center: bool = True,
) -> tuple[FloatArray, FloatArray]:
    """Compute POD energy fractions from a snapshot matrix.

    This helper computes the singular values of the snapshot matrix and returns:

    - the *per-mode* energy fraction, and
    - the *cumulative* energy fraction,

    which are often used to choose a POD rank.

    Parameters
    ----------
    Y
        Snapshot matrix with shape ``(n_snapshots, n_obs)`` where each row is one
        snapshot/trajectory in observation space.
    center
        If True, subtract the column-wise mean before computing the SVD. Centering is
        usually recommended for POD/PCA-style decompositions.

    Returns
    -------
    energy_fraction
        1D array where `energy_fraction[i]` is the fraction of total energy explained by
        mode `i` (0-indexed). Sums to 1 (up to numerical error).
    cumulative_energy
        1D array of cumulative sums of `energy_fraction`.

    Raises
    ------
    ValueError
        If `Y` is not two-dimensional.

    Notes
    -----
    The energy is computed from squared singular values:

    - `lambda_i = s_i^2`
    - `energy_fraction_i = lambda_i / sum(lambda_j)`
    """
    Y_arr = np.asarray(Y, dtype=float)
    if Y_arr.ndim != 2:
        raise ValueError(f"Y must be 2D (n_snapshots, n_obs). Got shape {Y_arr.shape}.")

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
    show: bool = False,
) -> tuple[tuple[Figure, Axes], tuple[Figure, Axes]]:
    """Plot per-mode and cumulative POD energy curves.

    The function produces two figures:

    1. **Per-mode energy fraction** (log scale)
    2. **Cumulative energy** with reference threshold lines

    This is typically used as a quick diagnostic for selecting a POD rank.

    Parameters
    ----------
    Y
        Snapshot matrix of shape ``(n_snapshots, n_obs)``.
    r_max
        Maximum number of modes to display. Must be positive.
    center
        Whether to mean-center the snapshots before SVD. See
        [`pod_energy_from_snapshots`][gp_active_mcmc.diagnostics.pod.pod_energy_from_snapshots].
    thresholds
        Cumulative energy thresholds to mark on the cumulative plot. Each value must be in
        `(0, 1]`. For each threshold `th`, the plot shows the smallest rank `r` such that
        cumulative energy is at least `th`.
    show
        If True, call `plt.show()` before returning. For MkDocs/Jupyter usage, leaving this
        False is recommended.

    Returns
    -------
    (fig_energy, ax_energy), (fig_cum, ax_cum)
        Two `(fig, ax)` pairs: per-mode energy and cumulative energy.

    Raises
    ------
    ValueError
        If `r_max <= 0`, if `Y` is not 2D, or if any threshold is not in `(0, 1]`.

    Notes
    -----
    The per-mode plot is shown on a log scale because POD spectra often decay quickly.
    """
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

    if show:
        plt.show()

    return (fig1, ax1), (fig2, ax2)
