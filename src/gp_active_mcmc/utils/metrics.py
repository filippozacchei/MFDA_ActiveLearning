from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def rmse(y_hat: ArrayLike, y_true: ArrayLike) -> float:
    """Compute the root-mean-square error (RMSE).

    Parameters
    ----------
    y_hat
        Predicted values.
    y_true
        Reference (ground-truth) values. Must have the same shape as `y_hat`.

    Returns
    -------
    rmse
        Root-mean-square error:

        .. math::

            \\mathrm{RMSE}(\\hat y, y) =
            \\sqrt{\\frac{1}{n}\\sum_{i=1}^{n}(\\hat y_i - y_i)^2}.

    Raises
    ------
    ValueError
        If `y_hat` and `y_true` have different shapes.

    Notes
    -----
    This metric is computed elementwise and then averaged over all entries. For
    trajectory outputs, this corresponds to an average over time points.
    """
    y_hat_arr = np.asarray(y_hat, dtype=float)
    y_true_arr = np.asarray(y_true, dtype=float)
    if y_hat_arr.shape != y_true_arr.shape:
        raise ValueError(
            "y_hat and y_true must have the same shape. "
            f"Got {y_hat_arr.shape} vs {y_true_arr.shape}."
        )
    return float(np.sqrt(np.mean((y_hat_arr - y_true_arr) ** 2)))


def coverage(
    y_true: ArrayLike,
    y_hat: ArrayLike,
    y_std: ArrayLike,
    *,
    z: float,
) -> float:
    """Compute empirical coverage of predictive intervals.

    This function measures the fraction of entries of `y_true` that fall inside the
    symmetric predictive interval

    .. math::

        [\\hat y - z\\,\\sigma,\\; \\hat y + z\\,\\sigma],

    where `z` is a chosen normal quantile (e.g. `z≈1.96` for an approximate 95% interval
    under a Gaussian assumption).

    Parameters
    ----------
    y_true
        Reference (ground-truth) values.
    y_hat
        Predictive mean values.
    y_std
        Predictive standard deviations (must be non-negative).
    z
        Interval half-width multiplier. Must be non-negative.

    Returns
    -------
    coverage
        Fraction of entries that satisfy `y_true ∈ [y_hat - z*y_std, y_hat + z*y_std]`.

    Raises
    ------
    ValueError
        If shapes are inconsistent, if `y_std` contains negative values, or if `z < 0`.

    Notes
    -----
    - This is an empirical coverage computed over all entries (and therefore over time
      points for trajectory outputs).
    - Coverage is meaningful only if `y_std` corresponds to uncertainty on the same
      quantity as `y_hat` (same units and alignment).
    """
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
