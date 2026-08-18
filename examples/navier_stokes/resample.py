from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def resample_profile(
    y: NDArray[np.float64], u: NDArray[np.float64], *, T: int
) -> NDArray[np.float64]:
    """Resample a 1D profile onto a uniform grid using linear interpolation.

    This is useful when downstream workflows expect a fixed-length vector, e.g.:
    - POD on outlet profiles,
    - GP training on coefficients,
    - likelihood evaluations on a fixed observation grid.

    Parameters
    ----------
    y
        Coordinates (n,). May be unsorted.
    u
        Profile values (n,), aligned with `y`.
    T
        Number of points in the resampled grid.

    Returns
    -------
    u_new
        Resampled profile of length `T`.

    Raises
    ------
    ValueError
        If shapes mismatch, if `T < 2`, or if fewer than two samples are provided.
    """
    y = np.asarray(y, dtype=float).ravel()
    u = np.asarray(u, dtype=float).ravel()

    if y.size != u.size:
        raise ValueError("y and u must have the same length.")
    if y.size < 2:
        raise ValueError("Need at least two points to resample.")
    if int(T) < 2:
        raise ValueError("T must be >= 2.")

    # Ensure monotone y for np.interp
    if not np.all(np.diff(y) >= 0.0):
        idx = np.argsort(y)
        y = y[idx]
        u = u[idx]

    y_new = np.linspace(float(y.min()), float(y.max()), int(T))
    return np.interp(y_new, y, u)
