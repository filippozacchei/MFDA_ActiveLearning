from __future__ import annotations

import numpy as np


def rmse(y_hat: np.ndarray, y_true: np.ndarray) -> float:
    """Root-mean-square error."""
    return float(np.sqrt(np.mean((y_hat - y_true) ** 2)))


def coverage(
    y_true: np.ndarray,
    y_hat: np.ndarray,
    y_std: np.ndarray,
    *,
    z: float,
) -> float:
    """Empirical coverage probability."""
    lo = y_hat - z * y_std
    hi = y_hat + z * y_std
    return float(np.mean((y_true >= lo) & (y_true <= hi)))
