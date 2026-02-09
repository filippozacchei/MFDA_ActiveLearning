from __future__ import annotations

from dataclasses import dataclass
import copy

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .gp import MultiOutputGP
from .pod import POD


FloatArray = NDArray[np.floating]


@dataclass(slots=True)
class PODGPSurrogate:
    """POD–GP surrogate mapping parameters theta -> time-series / fields .

    Workflow
    --------
    - POD reduces snapshots Y(t) to coefficients a in R^r
    - GP learns theta -> a
    - Reconstruction: y_hat = mean + a @ components_

    Uncertainty
    -----------
    Returns coefficient-wise predictive variances from the GP and maps them to
    pointwise output variance assuming independent coefficients:
        Var[y(t)] = sum_j Phi(t,j)^2 Var[a_j]
    """

    pod: POD
    gp: MultiOutputGP
    coeff_var_floor: float = 1e-12
    y_var_floor: float = 1e-14

    def _predict_coeffs(self, theta: ArrayLike) -> tuple[FloatArray, FloatArray]:
        theta2 = np.atleast_2d(np.asarray(theta, dtype=float))  # (n,d)
        mean_a, var_a = self.gp.predict(theta2)                 # (n,r), (n,r)

        var_a = np.maximum(var_a, float(self.coeff_var_floor))
        return mean_a, var_a

    def _reconstruct_var(self, var_a: FloatArray) -> FloatArray:
        """Map coefficient variance to pointwise output variance.

        Parameters
        ----------
        var_a
            Coefficient variances with shape (r,) or (n,r).

        Returns
        -------
        y_var
            Output variance with shape (n_time,) or (n,n_time).
        """
        if not self.pod.is_fitted:
            raise RuntimeError("POD is not fitted. Fit POD before using PODGPSurrogate.")

        if self.pod.components_ is None:
            raise RuntimeError("POD is not fitted (components_ missing).")

        Phi = self.pod.components_.T  # (n_time, r)
        var = np.asarray(var_a, dtype=float)

        if var.ndim == 1:
            y_var = (Phi**2) @ var  # (n_time,)
            return np.maximum(y_var, float(self.y_var_floor))

        if var.ndim == 2:
            # (n,r) -> (n,n_time) via (Phi^2) (n_time,r)
            y_var = var @ (Phi**2).T
            return np.maximum(y_var, float(self.y_var_floor))

        raise ValueError(f"var_a must be 1D or 2D. Got shape {var.shape}.")

    def predict(self, theta: ArrayLike) -> tuple[FloatArray, FloatArray]:
        """Predict mean time series and pointwise variance.

        Returns
        -------
        y_mean : (n_time,) if theta is 1D else (n, n_time)
        y_var  : same shape as y_mean
        """
        theta_arr = np.asarray(theta, dtype=float)
        is_single = theta_arr.ndim == 1

        mean_a, var_a = self._predict_coeffs(theta_arr)  # (n,r), (n,r)
        y_mean = self.pod.inverse_transform(mean_a)      # (n,n_time)
        y_var = self._reconstruct_var(var_a)              # (n,n_time)

        if is_single:
            return y_mean[0], y_var[0]
        return y_mean, y_var

    def update(self, theta: ArrayLike, y_true: ArrayLike) -> None:
        """Update GP with one new high-fidelity observation."""
        theta2 = np.atleast_2d(np.asarray(theta, dtype=float))  # (1,d) expected
        y2 = np.atleast_2d(np.asarray(y_true, dtype=float))      # (1,n_time)

        if theta2.shape[0] != 1 or y2.shape[0] != 1:
            raise ValueError("update expects a single theta and a single snapshot y_true.")

        a_true = self.pod.transform(y2)[0]  # (r,)
        self.gp.update(theta2, a_true)

    def log_likelihood(self) -> float:
        return self.gp.log_likelihood()

    def copy(self) -> "PODGPSurrogate":
        """Deep copy (useful when proposals need independent surrogate state)."""
        return PODGPSurrogate(
            pod=copy.deepcopy(self.pod),
            gp=copy.deepcopy(self.gp),
            coeff_var_floor=self.coeff_var_floor,
            y_var_floor=self.y_var_floor,
        )
