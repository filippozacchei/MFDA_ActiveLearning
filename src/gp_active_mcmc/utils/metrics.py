from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def rmse(y_hat: ArrayLike, y_true: ArrayLike) -> float:
    """Root-mean-square error."""
    y_hat_arr = np.asarray(y_hat, dtype=float)
    y_true_arr = np.asarray(y_true, dtype=float)
    if y_hat_arr.shape != y_true_arr.shape:
        raise ValueError(f"y_hat and y_true must have the same shape. Got {y_hat_arr.shape} vs {y_true_arr.shape}.")
    return float(np.sqrt(np.mean((y_hat_arr - y_true_arr) ** 2)))


def coverage(
    y_true: ArrayLike,
    y_hat: ArrayLike,
    y_std: ArrayLike,
    *,
    z: float,
) -> float:
    """Empirical coverage probability for y_true within [y_hat +/- z*y_std]."""
    if z < 0:
        raise ValueError("z must be non-negative.")

    y_true_arr = np.asarray(y_true, dtype=float)
    y_hat_arr = np.asarray(y_hat, dtype=float)
    y_std_arr = np.asarray(y_std, dtype=float)

    if y_true_arr.shape != y_hat_arr.shape or y_true_arr.shape != y_std_arr.shape:
        raise ValueError("y_true, y_hat, and y_std must have the same shape.")
    if np.any(y_std_arr < 0):
        raise ValueError("y_std must be non-negative.")

    lo = y_hat_arr - z * y_std_arr
    hi = y_hat_arr + z * y_std_arr
    return float(np.mean((y_true_arr >= lo) & (y_true_arr <= hi)))
