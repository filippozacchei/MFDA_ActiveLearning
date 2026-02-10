from __future__ import annotations

from typing import Any

import numpy as np
import tinyDA as tda


class ActiveGPLogLike(tda.AdaptiveGaussianLogLike):
    """Gaussian log-likelihood with surrogate-variance inflation.

    `ActiveGPLogLike` is designed for inference workflows where the forward model may
    return either:

    1. **High-fidelity (HF)** output: a 1D mean prediction `y`, or
    2. **Low-fidelity (LF) surrogate** output: an array-like mean prediction that also
       exposes a pointwise predictive variance vector via a `.variance` attribute
       (for example [`CoarseOutput`][gp_active_mcmc.inference.coarse_output.CoarseOutput]).

    If a predictive variance vector is present, the likelihood inflates the observation
    covariance using a diagonal term:"""

    r"""
    \[
    C_{\mathrm{total}} = C_{\mathrm{obs}} + \mathrm{diag}(v_{\mathrm{pred}}) + C_{\mathrm{bias}}.
    \]
    """
    """
    This makes surrogate uncertainty explicit in the likelihood: higher predictive
    variance corresponds to a broader effective noise model and therefore a less
    concentrated likelihood.

    Notes
    -----
    **Accepted prediction formats**

    The [`loglike`][gp_active_mcmc.inference.likelihood.ActiveGPLogLike.loglike] method accepts:

    - a 1D array-like mean prediction (HF case), or
    - an object `y_pred` such that:
      - `np.asarray(y_pred)` yields the predictive mean, and
      - `getattr(y_pred, "variance", None)` yields a 1D variance vector aligned with that mean.

    This matches the behaviour of
    [`ActiveMCMCModel.coarse`][gp_active_mcmc.inference.model.ActiveMCMCModel.coarse], which returns
    either an HF mean array or a `CoarseOutput(mean, variance)`.

    Implementation detail
    ---------------------
    For debugging and diagnostics, the covariance used for the most recent call is stored as:

    - `total_cov`: the covariance actually used for this call
    - `cov_inverse`: its inverse

    Warnings
    --------
    This implementation recomputes a dense matrix inverse each call. For large `n_obs`,
    a Cholesky-based implementation (solve rather than invert) is typically more stable
    and faster.

    See Also
    --------
    [`CoarseOutput`][gp_active_mcmc.inference.coarse_output.CoarseOutput]
        LF prediction container that carries `.variance`.
    [`ActiveMCMCModel`][gp_active_mcmc.inference.model.ActiveMCMCModel]
        Active model whose coarse path may return `CoarseOutput`.
    """

    def loglike(self, y_pred: Any) -> float:
        """Evaluate the Gaussian log-likelihood for a prediction.

        Parameters
        ----------
        y_pred
            Model prediction. Either:

            - a 1D array-like mean prediction, or
            - an object providing a mean prediction and an attribute `variance` (1D).

        Returns
        -------
        loglike
            Log-likelihood value.

        Raises
        ------
        ValueError
            If shapes are inconsistent with the observed data, or if a provided variance
            vector is negative or mismatched.
        """
        mean = np.atleast_1d(np.asarray(y_pred, dtype=float))

        if mean.ndim != 1:
            raise ValueError(f"Predicted mean must be 1D. Got shape {mean.shape}.")
        if mean.shape[0] != self.data.shape[0]:
            raise ValueError(
                f"Predicted mean length ({mean.shape[0]}) does not match data length "
                f"({self.data.shape[0]})."
            )

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
