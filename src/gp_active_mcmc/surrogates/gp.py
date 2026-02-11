from __future__ import annotations

from typing import Literal

import GPy
import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.preprocessing import StandardScaler

KernelName = Literal["rbf", "matern32", "matern52"]
FloatArray = NDArray[np.float64]


def _as_2d_float(x: ArrayLike, *, name: str) -> FloatArray:
    """Convert an array-like to a 2D float array.

    Parameters
    ----------
    x
        Input array-like. Accepted shapes are ``(d,)`` or ``(n, d)``.
        A 1D vector is interpreted as a single row and reshaped to ``(1, d)``.
    name
        Name used in error messages.

    Returns
    -------
    x_2d
        2D float array.

    Raises
    ------
    ValueError
        If `x` is not 1D or 2D.
    """
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        return arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D. Got shape {arr.shape}.")
    return arr


def _as_2d_targets(y: ArrayLike, *, name: str) -> FloatArray:
    """Convert target values to a 2D float array.

    Parameters
    ----------
    y
        Target array-like. Accepted shapes are ``(n,)`` or ``(n, m)``.
        A 1D vector is reshaped to ``(n, 1)``.
    name
        Name used in error messages.

    Returns
    -------
    y_2d
        2D float array.

    Raises
    ------
    ValueError
        If `y` is not 1D or 2D.
    """
    arr = np.asarray(y, dtype=float)
    if arr.ndim == 1:
        return arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D. Got shape {arr.shape}.")
    return arr


def _make_kernel(d: int, *, kernel: KernelName, ard: bool) -> GPy.kern.Kern:
    """Construct a GPy kernel by name.

    Parameters
    ----------
    d
        Input dimension.
    kernel
        Kernel name.
    ard
        If True, enable ARD (separate lengthscale per input dimension).

    Returns
    -------
    kern
        GPy kernel instance.

    Raises
    ------
    ValueError
        If `kernel` is unknown.
    """
    if kernel == "rbf":
        return GPy.kern.RBF(d, ARD=ard)
    if kernel == "matern32":
        return GPy.kern.Matern32(d, ARD=ard)
    if kernel == "matern52":
        return GPy.kern.Matern52(d, ARD=ard)
    raise ValueError(f"Unknown kernel: {kernel}")


class SingleOutputGP:
    """Single-output GP regression with input/output standardisation.

    This is a thin wrapper around `GPy.models.GPRegression` that standardises inputs
    and outputs using `sklearn.preprocessing.StandardScaler`.

    Standardisation is useful in practice because:

    - it makes kernel lengthscales easier to learn (inputs have comparable scales),
    - it stabilises optimisation of hyperparameters when outputs vary in magnitude.

    Parameters
    ----------
    X_train
        Training inputs of shape ``(n_train, n_dim)`` (or ``(n_dim,)`` for a single point).
    y_train
        Training targets of shape ``(n_train,)`` or ``(n_train, 1)``.
    kernel
        Kernel name. Supported: `"rbf"`, `"matern32"`, `"matern52"`.
    ard
        If True, use ARD (one lengthscale per input dimension).
    noise_variance
        Initial observation noise variance for the GP likelihood.
    update_every
        Retraining period: after every `update_every` calls to [`update`][gp_active_mcmc.surrogates.SingleOutputGP.update],
        the GP hyperparameters are re-optimised (until `n_retrain_max` is reached).
        Set `update_every<=0` to disable re-optimisation.
    n_retrain_max
        Maximum number of hyperparameter re-optimisations triggered by updates.

    Notes
    -----
    - [`update`][gp_active_mcmc.surrogates.SingleOutputGP.update] appends exactly **one**
      new observation with shape `(1, n_dim) -> (1, 1)`.
    - Hyperparameter optimisation uses `GPy`'s `.optimize()` routine.
    - This class intentionally avoids any I/O and does not expose plotting.

    See Also
    --------
    [`MultiOutputGP`][gp_active_mcmc.surrogates.MultiOutputGP]
        Multi-output wrapper implemented as independent single-output GPs.
    """

    def __init__(
        self,
        X_train: ArrayLike,  # (N,d)
        y_train: ArrayLike,  # (N,) or (N,1)
        *,
        kernel: KernelName = "matern52",
        ard: bool = True,
        noise_variance: float = 1e-6,
        update_every: int = 10,
        n_retrain_max: int = 20,
    ) -> None:
        X = _as_2d_float(X_train, name="X_train")
        y = _as_2d_targets(y_train, name="y_train")
        if y.shape[1] != 1:
            raise ValueError(f"y_train must have shape (N,) or (N,1). Got {y.shape}.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X_train and y_train must have the same number of rows.")

        self._x_scaler = StandardScaler()
        self._y_scaler = StandardScaler()

        Xs = self._x_scaler.fit_transform(X)
        ys = self._y_scaler.fit_transform(y)

        kern = _make_kernel(X.shape[1], kernel=kernel, ard=ard)
        self._gp = GPy.models.GPRegression(Xs, ys, kern)

        self._gp.Gaussian_noise.variance = float(noise_variance)
        self._gp.Gaussian_noise.unfix()
        self._gp.optimize()

        self._update_every = int(update_every)
        self._n_retrain_max = int(n_retrain_max)
        self._retrain_count = 0
        self._counter = 0

    @property
    def n_train(self) -> int:
        """Number of training points currently stored by the GP."""
        return int(self._gp.X.shape[0])

    def _maybe_optimize(self) -> None:
        """Optionally re-optimise hyperparameters (internal policy).

        This method implements the *internal* retraining schedule controlled by
        `update_every` and `n_retrain_max`. It is called by [`update`][gp_active_mcmc.gp.SingleOutputGP.update].

        Notes
        -----
        This is not exposed as a public API because it relies on internal counters and
        is intended to be a conservative, predictable default for online updates.
        If you need explicit control over optimisation, add a public method with an
        explicit name (e.g. `optimize()`).
        """
        self._counter += 1
        if self._update_every <= 0:
            return
        if self._retrain_count >= self._n_retrain_max:
            return
        if self._counter % self._update_every == 0:
            self._gp.optimize()
            self._retrain_count += 1

    def predict(self, X: ArrayLike) -> tuple[FloatArray, FloatArray]:
        """Predict mean and marginal variance.

        Parameters
        ----------
        X
            Query inputs of shape ``(n, n_dim)`` or ``(n_dim,)``.

        Returns
        -------
        mean
            Predictive mean of shape ``(n,)`` in the original (unstandardised) output space.
        var
            Predictive marginal variance of shape ``(n,)`` in the original output space.

        Notes
        -----
        `GPy` returns mean and variance in standardised output space; this method
        transforms them back to the original units. If `y = a * y_std + b`, then
        `Var[y] = Var[a * y_std] = a^2 Var[y_std]`.
        """
        Xq = _as_2d_float(X, name="X")
        mu_s, var_s = self._gp.predict(self._x_scaler.transform(Xq))  # (n,1), (n,1)

        mu = self._y_scaler.inverse_transform(mu_s)  # (n,1)
        y_std = float(self._y_scaler.scale_[0])
        var = var_s * (y_std**2)  # (n,1)

        return mu[:, 0], var[:, 0]

    def update(self, X_new: ArrayLike, y_new: ArrayLike) -> None:
        """Append one new observation and optionally retrain hyperparameters.

        Parameters
        ----------
        X_new
            New input with shape ``(1, n_dim)`` (or ``(n_dim,)`` which is interpreted as one point).
        y_new
            New target with shape ``(1, 1)``, ``(1,)``, or scalar-like.

        Raises
        ------
        ValueError
            If the provided data cannot be interpreted as exactly one new observation.

        Notes
        -----
        This method:

        1. standardises `(X_new, y_new)` using the scalers fit at construction time,
        2. appends the standardised data to the underlying `GPy` model,
        3. triggers internal re-optimisation according to the update schedule.
        """
        Xn = _as_2d_float(X_new, name="X_new")

        yn = np.asarray(y_new, dtype=float)
        if yn.ndim == 0:
            yn = yn.reshape(1, 1)
        yn2 = _as_2d_targets(yn, name="y_new")
        if yn2.shape != (1, 1) or Xn.shape[0] != 1:
            raise ValueError("update expects one point: X_new shape (1,d) and y_new shape (1,1).")

        Xs = self._x_scaler.transform(Xn)
        ys = self._y_scaler.transform(yn2)

        self._gp.set_XY(np.vstack([self._gp.X, Xs]), np.vstack([self._gp.Y, ys]))
        self._maybe_optimize()

    def log_likelihood(self) -> float:
        """Return the GP marginal log-likelihood in standardised space."""
        return float(self._gp.log_likelihood())


class MultiOutputGP:
    """Multi-output GP implemented as independent single-output GPs.

    This class represents a vector-valued regression function by training one
    [`SingleOutputGP`][gp_active_mcmc.surrogates.SingleOutputGP] per output dimension.
    Outputs are therefore conditionally independent given inputs.

    Shapes
    ------
    Training:
        - `X_train`: shape ``(n_train, n_dim)``
        - `Y_train`: shape ``(n_train, n_out)``
    Prediction:
        - `X`: shape ``(n, n_dim)``
        - `mean`: shape ``(n, n_out)``
        - `var`: shape ``(n, n_out)``
    Update:
        - `X_new`: shape ``(1, n_dim)``
        - `y_new`: shape ``(n_out,)`` or ``(1, n_out)``

    Parameters
    ----------
    X_train
        Training inputs of shape ``(n_train, n_dim)``.
    Y_train
        Training targets of shape ``(n_train, n_out)`` (or ``(n_train,)`` for `n_out=1`).
    kernel, ard, noise_variance, update_every, n_retrain_max
        Passed to each `SingleOutputGP`.

    Notes
    -----
    - This is a pragmatic design that keeps the implementation simple and robust.
      If you need correlated multi-output GPs (coregionalisation), a different model
      class should be introduced.
    - Online updates append one joint observation by updating each output GP.

    Attributes
    ----------
    n_out
        Number of output dimensions.

    See Also
    --------
    [`SingleOutputGP`][gp_active_mcmc.surrogates.SingleOutputGP]
        Underlying scalar GP used per output dimension.
    """

    def __init__(
        self,
        X_train: ArrayLike,
        Y_train: ArrayLike,
        *,
        kernel: KernelName = "matern52",
        ard: bool = True,
        noise_variance: float = 1e-6,
        update_every: int = 10,
        n_retrain_max: int = 20,
    ) -> None:
        X = _as_2d_float(X_train, name="X_train")
        Y = _as_2d_targets(Y_train, name="Y_train")
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X_train and Y_train must have the same number of rows.")

        self.n_out = int(Y.shape[1])
        self._gps = [
            SingleOutputGP(
                X,
                Y[:, j],
                kernel=kernel,
                ard=ard,
                noise_variance=noise_variance,
                update_every=update_every,
                n_retrain_max=n_retrain_max,
            )
            for j in range(self.n_out)
        ]

    @property
    def n_train(self) -> int:
        """Number of training points stored (shared across outputs)."""
        return self._gps[0].n_train

    def predict(self, X: ArrayLike) -> tuple[FloatArray, FloatArray]:
        """Predict mean and marginal variance for all outputs.

        Parameters
        ----------
        X
            Query inputs of shape ``(n, n_dim)`` or ``(n_dim,)``.

        Returns
        -------
        mean
            Array of shape ``(n, n_out)`` of predictive means.
        var
            Array of shape ``(n, n_out)`` of predictive marginal variances.
        """
        Xq = _as_2d_float(X, name="X")

        mean = np.empty((Xq.shape[0], self.n_out), dtype=float)
        var = np.empty((Xq.shape[0], self.n_out), dtype=float)

        for j, gp in enumerate(self._gps):
            mu_j, var_j = gp.predict(Xq)
            mean[:, j] = mu_j
            var[:, j] = var_j

        return mean, var

    def update(self, X_new: ArrayLike, y_new: ArrayLike) -> None:
        """Append one new joint observation for all outputs.

        Parameters
        ----------
        X_new
            New input of shape ``(1, n_dim)`` (or ``(n_dim,)`` interpreted as one point).
        y_new
            New outputs of shape ``(n_out,)`` or ``(1, n_out)``.

        Raises
        ------
        ValueError
            If the shapes do not describe exactly one new joint observation.
        """
        Xn = _as_2d_float(X_new, name="X_new")
        if Xn.shape[0] != 1:
            raise ValueError("update expects X_new shape (1,d).")

        y = np.asarray(y_new, dtype=float)
        if y.ndim == 2 and y.shape[0] == 1:
            y = y[0]
        y = y.ravel()

        if y.shape[0] != self.n_out:
            raise ValueError(f"y_new must have length {self.n_out}. Got {y.shape[0]}.")

        for j, gp in enumerate(self._gps):
            gp.update(Xn, np.array([[y[j]]], dtype=float))

    def log_likelihood(self) -> float:
        """Return the sum of marginal log-likelihoods across output GPs."""
        return float(sum(gp.log_likelihood() for gp in self._gps))
