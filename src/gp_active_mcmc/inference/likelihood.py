from __future__ import annotations

from typing import Any

import numpy as np
import tinyDA as tda


class ActiveGPLogLike(tda.AdaptiveGaussianLogLike):
    """
    Gaussian log-likelihood with optional predictive-variance inflation.

    This likelihood is intended for Bayesian inference with either:

    - a high-fidelity (HF) forward model that returns only the predicted mean as a
    NumPy array, or
    - a surrogate model that provides both a predicted mean and an estimate of
    marginal (pointwise) predictive variance.

    When predictive variances are available (e.g., via a ``CoarseOutput`` instance or
    an equivalent object exposing a ``.variance`` attribute), they are incorporated
    by augmenting the observational covariance with a diagonal matrix:

        C_total = C_obs + diag(v_pred)  (+ cov_bias, if present)

    where ``C_obs`` is the observation-noise covariance and ``v_pred`` is the
    predictive marginal variance vector aligned with the observation space.
    """


    def loglike(self, y_pred: Any) -> float:
        mean = np.atleast_1d(np.asarray(y_pred, dtype=float))

        if mean.ndim != 1:
            raise ValueError(f"Predicted mean must be 1D. Got shape {mean.shape}.")
        if mean.shape[0] != self.data.shape[0]:
            raise ValueError("Predicted mean length does not match data length.")

        total_cov = np.asarray(self.cov, dtype=float).copy()

        cov_bias = getattr(self, "cov_bias", None)
        if cov_bias is not None:
            cb = np.asarray(cov_bias, dtype=float)
            if cb.shape != total_cov.shape:
                raise ValueError("cov_bias must have the same shape as covariance.")
            total_cov += cb

        var = getattr(y_pred, "variance", None)
        if var is not None:
            var = np.atleast_1d(np.asarray(var, dtype=float))
            if var.shape != mean.shape:
                raise ValueError("variance must have the same shape as mean.")
            if np.any(var < 0.0):
                raise ValueError("variance must be non-negative.")
            total_cov += np.diag(var)

        self.total_cov = total_cov
        self.cov_inverse = np.linalg.inv(total_cov)

        return super().loglike(mean)
