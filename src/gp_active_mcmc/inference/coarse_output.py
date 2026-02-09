from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class CoarseOutput(np.ndarray):
    """Array-like prediction with attached marginal variance.

    Notes
    -----
    - `variance` is interpreted as marginal (pointwise) variance.
    - The likelihood will typically add `diag(variance)` to the observation covariance.
    """

    variance: NDArray[np.floating]

    def __new__(cls, mean: np.ndarray, variance: np.ndarray):
        mean_arr = np.asarray(mean, dtype=float)
        var_arr = np.asarray(variance, dtype=float)

        if mean_arr.shape != var_arr.shape:
            raise ValueError(f"mean and variance must have the same shape. Got {mean_arr.shape} vs {var_arr.shape}.")
        if np.any(var_arr < 0.0):
            raise ValueError("variance must be non-negative.")

        obj = mean_arr.view(cls)
        obj.variance = var_arr
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.variance = getattr(obj, "variance", None)
