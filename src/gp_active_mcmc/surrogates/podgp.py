from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .gp import MultiOutputGP
from .pod import POD

FloatArray = NDArray[np.floating]


@dataclass(slots=True)
class PODGPSurrogate:
    """POD–GP surrogate for trajectory- or field-valued model outputs.

    This surrogate implements a common two-step construction for high-dimensional outputs:

    1. **Compression (POD)**: map each snapshot/trajectory ``y`` to a low-dimensional
       coefficient vector ``a ∈ R^r``.
    2. **Regression (GP)**: learn the mapping ``θ → a`` using a Gaussian process.

    The surrogate then provides predictions in the original observation space by
    reconstructing:
    """

    r"""
    \[\hat{y(\theta)} = \mu + \hat{a(\theta)}\Phi,\]

    where $\mu$ is the POD mean (stored in `pod.mean_`) and $\Phi$ are the POD modes
    (rows of `pod.components_`). In code, the reconstruction is performed by
    [`POD.inverse_transform`][gp_active_mcmc.pod.POD.inverse_transform].

    Uncertainty propagation
    -----------------------
    The underlying GP returns a predictive *marginal* variance for each POD coefficient.
    This class maps coefficient variances to an **output-space pointwise variance**
    under an independence assumption across coefficients:

    \[\mathrm{Var}[y_i] \approx \sum_{j=1}^{r} \Phi_{j,i}^2\, \mathrm{Var}[a_j]\],

    where ``i`` indexes the observation component (e.g., time index) and ``j`` the POD mode.

    This is a pragmatic approximation that is fast and works well for diagnostics
    (e.g., uncertainty bands), but it does not represent correlated output uncertainty.

    Parameters
    ----------
    pod
        Fitted POD object. Must be fitted before calling [`predict`][gp_active_mcmc.podgp.PODGPSurrogate.predict]
        or [`update`][gp_active_mcmc.podgp.PODGPSurrogate.update].
    gp
        Multi-output GP model trained to predict POD coefficients.
        It must implement:

        - ``predict(theta) -> (mean_a, var_a)`` with shapes ``(n, r)``,
        - ``update(theta_new, a_new)`` for online updates.

    coeff_var_floor
        Small non-negative floor applied to coefficient variances returned by the GP.
        This prevents numerical issues when variances become exactly zero.
    y_var_floor
        Small non-negative floor applied to the reconstructed output variance.

    Notes
    -----
    This class is intended to satisfy the library's surrogate protocol
    ([`ActiveSurrogate`][gp_active_mcmc.protocols.ActiveSurrogate]):

    - [`predict`][gp_active_mcmc.podgp.PODGPSurrogate.predict] returns `(mean, var)` in observation space.
    - [`update`][gp_active_mcmc.podgp.PODGPSurrogate.update] ingests a new HF snapshot and updates the GP
      in coefficient space.

    See Also
    --------
    [`POD`][gp_active_mcmc.pod.POD]
        POD compression model.
    [`MultiOutputGP`][gp_active_mcmc.gp.MultiOutputGP]
        GP model used for coefficient regression.
    """

    pod: POD
    gp: MultiOutputGP
    coeff_var_floor: float = 1e-12
    y_var_floor: float = 1e-14

    def _predict_coeffs(self, theta: ArrayLike) -> tuple[FloatArray, FloatArray]:
        """Predict POD coefficients (mean and variance) from parameters.

        Parameters
        ----------
        theta
            Parameter vector of shape ``(d,)`` or batch of shape ``(n, d)``.

        Returns
        -------
        mean_a
            Predictive mean of POD coefficients, shape ``(n, r)``.
        var_a
            Predictive marginal variance of POD coefficients, shape ``(n, r)``.
            A small floor is applied for numerical robustness.
        """
        theta2 = np.atleast_2d(np.asarray(theta, dtype=float))  # (n, d)
        mean_a, var_a = self.gp.predict(theta2)  # (n, r), (n, r)

        var_a = np.maximum(var_a, float(self.coeff_var_floor))
        return mean_a, var_a

    def _reconstruct_var(self, var_a: FloatArray) -> FloatArray:
        """Map coefficient variance to pointwise output variance.

        Parameters
        ----------
        var_a
            Coefficient variances with shape ``(r,)`` or ``(n, r)``.

        Returns
        -------
        y_var
            Output pointwise variance with shape ``(n_obs,)`` or ``(n, n_obs)``.

        Raises
        ------
        RuntimeError
            If the POD object is not fitted.
        ValueError
            If `var_a` is not 1D or 2D.
        """
        if not self.pod.is_fitted:
            raise RuntimeError("POD is not fitted. Fit POD before using PODGPSurrogate.")
        if self.pod.components_ is None:
            raise RuntimeError("POD is not fitted (components_ missing).")

        # pod.components_ has shape (r, n_obs); transpose to (n_obs, r).
        Phi = self.pod.components_.T  # (n_obs, r)
        var = np.asarray(var_a, dtype=float)

        if var.ndim == 1:
            # (n_obs, r) @ (r,) -> (n_obs,)
            y_var = (Phi**2) @ var
            return np.maximum(y_var, float(self.y_var_floor))

        if var.ndim == 2:
            # (n, r) @ (r, n_obs) -> (n, n_obs)
            y_var = var @ (Phi**2).T
            return np.maximum(y_var, float(self.y_var_floor))

        raise ValueError(f"var_a must be 1D or 2D. Got shape {var.shape}.")

    def predict(self, theta: ArrayLike) -> tuple[FloatArray, FloatArray]:
        """Predict mean output and pointwise variance in observation space.

        Parameters
        ----------
        theta
            Parameters of shape ``(d,)`` or batch of shape ``(n, d)``.

        Returns
        -------
        y_mean
            Predictive mean in observation space.

            - shape ``(n_obs,)`` if `theta` is 1D
            - shape ``(n, n_obs)`` if `theta` is 2D
        y_var
            Predictive pointwise variance in observation space, same shape as `y_mean`.

        Notes
        -----
        The returned variance is obtained by propagating coefficient-wise variances
        through the POD reconstruction under an independence approximation.
        """
        theta_arr = np.asarray(theta, dtype=float)
        is_single = theta_arr.ndim == 1

        mean_a, var_a = self._predict_coeffs(theta_arr)  # (n, r), (n, r)
        y_mean = self.pod.inverse_transform(mean_a)  # (n, n_obs)
        y_var = self._reconstruct_var(var_a)  # (n, n_obs)

        if is_single:
            return y_mean[0], y_var[0]
        return y_mean, y_var

    def update(self, theta: ArrayLike, y_true: ArrayLike) -> None:
        """Update the surrogate with one new high-fidelity observation.

        This method projects the new HF snapshot into POD coefficient space and updates
        the GP with a single new training point.

        Parameters
        ----------
        theta
            Parameter vector of shape ``(d,)`` (or ``(1, d)``).
        y_true
            HF snapshot/trajectory in observation space, shape ``(n_obs,)`` (or ``(1, n_obs)``).

        Raises
        ------
        ValueError
            If a batch (more than one observation) is provided.
        RuntimeError
            If the POD object is not fitted.
        """
        if not self.pod.is_fitted:
            raise RuntimeError("POD is not fitted. Fit POD before calling update().")

        theta2 = np.atleast_2d(np.asarray(theta, dtype=float))  # (1, d) expected
        y2 = np.atleast_2d(np.asarray(y_true, dtype=float))  # (1, n_obs)

        if theta2.shape[0] != 1 or y2.shape[0] != 1:
            raise ValueError("update expects a single theta and a single snapshot y_true.")

        a_true = self.pod.transform(y2)[0]  # (r,)
        self.gp.update(theta2, a_true)

    def log_likelihood(self) -> float:
        """Return the summed marginal log-likelihood of the underlying GP(s)."""
        return self.gp.log_likelihood()

    def copy(self) -> "PODGPSurrogate":
        """Return a deep copy of the surrogate.

        This is useful when a workflow needs independent surrogate state, e.g. when
        running truly independent chains or experiments.

        Notes
        -----
        Many active-learning workflows instead prefer *shared* state across deepcopies
        (see [`AdaptiveMetropolisShared`][gp_active_mcmc.inference.proposal.AdaptiveMetropolisShared]).
        """
        return PODGPSurrogate(
            pod=copy.deepcopy(self.pod),
            gp=copy.deepcopy(self.gp),
            coeff_var_floor=self.coeff_var_floor,
            y_var_floor=self.y_var_floor,
        )
