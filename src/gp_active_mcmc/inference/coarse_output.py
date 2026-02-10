from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class CoarseOutput(np.ndarray):
    """Prediction container for LF (surrogate) evaluations.

    `CoarseOutput` behaves like a NumPy array containing the predictive mean, while
    attaching a `.variance` attribute that stores **marginal (pointwise) predictive
    variance** aligned with that mean.

    The class exists to keep the LF path compatible with array-oriented APIs (notably
    `tinyDA` forward models and likelihoods), without discarding uncertainty information
    produced by the surrogate.

    Typical use in this library
    ---------------------------
    - [`ActiveMCMCModel.coarse`][gp_active_mcmc.inference.model.ActiveMCMCModel.coarse]
      returns a `CoarseOutput(mean, variance)` when the LF surrogate is used.
    - [`ActiveGPLogLike`][gp_active_mcmc.inference.likelihood.ActiveGPLogLike] detects the
      `.variance` attribute and inflates the observation covariance with a diagonal term
      `diag(variance)`.

    Interpretation
    --------------
    The stored variance is interpreted as marginal variance in observation space:

    - `mean[i]` is the predicted mean for observation component `i`
    - `variance[i]` is the predictive variance for the same component

    Notes
    -----
    - This is a lightweight container: it stores *marginal* variances only. Output
      correlations are not represented.
    - `variance` must be non-negative and have the same shape as the mean.

    Parameters
    ----------
    mean
        Predictive mean array.
    variance
        Predictive marginal variance array (same shape as `mean`).

    Attributes
    ----------
    variance
        Marginal (pointwise) variance aligned with the mean.

    Examples
    --------
    >>> y = CoarseOutput(mean=np.array([1.0, 2.0]), variance=np.array([0.1, 0.2]))
    >>> np.asarray(y)
    array([1., 2.])
    >>> y.variance
    array([0.1, 0.2])

    See Also
    --------
    [`ActiveMCMCModel`][gp_active_mcmc.inference.model.ActiveMCMCModel]
        Coupled LF/HF model that returns `CoarseOutput` on the coarse path.
    [`ActiveGPLogLike`][gp_active_mcmc.inference.likelihood.ActiveGPLogLike]
        Likelihood that uses `CoarseOutput.variance` to inflate the observation covariance.
    """

    variance: NDArray[np.floating]

    def __new__(cls, mean: np.ndarray, variance: np.ndarray) -> "CoarseOutput":
        """Create a `CoarseOutput` from mean and marginal variance.

        Parameters
        ----------
        mean
            Predictive mean array.
        variance
            Predictive marginal variance array (same shape as `mean`).

        Returns
        -------
        out
            Array-like object whose values are the predictive mean and with an attached
            `.variance` attribute.

        Raises
        ------
        ValueError
            If `mean` and `variance` have different shapes, or if `variance` contains
            negative entries.
        """
        mean_arr = np.asarray(mean, dtype=float)
        var_arr = np.asarray(variance, dtype=float)

        if mean_arr.shape != var_arr.shape:
            raise ValueError(
                "mean and variance must have the same shape. "
                f"Got {mean_arr.shape} vs {var_arr.shape}."
            )
        if np.any(var_arr < 0.0):
            raise ValueError("variance must be non-negative.")

        obj = mean_arr.view(cls)
        obj.variance = var_arr
        return obj

    def __array_finalize__(self, obj: object | None) -> None:
        """Propagate `.variance` when NumPy creates a new view.

        NumPy calls `__array_finalize__` for view-casting and slicing operations.
        This method preserves the `variance` attribute when the new array is created
        from an existing `CoarseOutput`.
        """
        if obj is None:
            return
        self.variance = getattr(obj, "variance", None)
