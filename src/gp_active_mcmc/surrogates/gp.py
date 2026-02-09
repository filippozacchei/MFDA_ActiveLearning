from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.preprocessing import StandardScaler

import GPy


KernelName = Literal["rbf", "matern32", "matern52"]
FloatArray = NDArray[np.floating]


def _as_2d_float(x: ArrayLike, *, name: str) -> FloatArray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        return arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D. Got shape {arr.shape}.")
    return arr


def _as_2d_targets(y: ArrayLike, *, name: str) -> FloatArray:
    arr = np.asarray(y, dtype=float)
    if arr.ndim == 1:
        return arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D. Got shape {arr.shape}.")
    return arr


def _make_kernel(d: int, *, kernel: KernelName, ard: bool) -> GPy.kern.Kern:
    if kernel == "rbf":
        return GPy.kern.RBF(d, ARD=ard)
    if kernel == "matern32":
        return GPy.kern.Matern32(d, ARD=ard)
    if kernel == "matern52":
        return GPy.kern.Matern52(d, ARD=ard)
    raise ValueError(f"Unknown kernel: {kernel}")


class SingleOutputGP:
    """Single-output GP regression with input/output standardisation.

    Notes
    -----
    - Uses GPy.models.GPRegression.
    - `update()` appends exactly one new observation (1,d) -> (1,1).
    - Hyperparameters are optionally re-optimised periodically.
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
        return int(self._gp.X.shape[0])

    def _optimize(self) -> None:
        """
        This is just a private method. 
        If a public optimization is needed a new method should be written.
        Do not use this as it relies on internal counters and 
        number of retraining iterations.
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
        """Predict mean and variance.

        Returns
        -------
        mean : (n,)
        var  : (n,)
        """
        Xq = _as_2d_float(X, name="X")
        mu_s, var_s = self._gp.predict(self._x_scaler.transform(Xq))  # (n,1), (n,1)

        mu = self._y_scaler.inverse_transform(mu_s)  # (n,1)
        y_std = float(self._y_scaler.scale_[0])
        var = var_s * (y_std**2)  # (n,1)

        return mu[:, 0], var[:, 0]

    def update(self, X_new: ArrayLike, y_new: ArrayLike) -> None:
        """Append one new observation (1,d) -> (1,1)."""
        Xn = _as_2d_float(X_new, name="X_new")
        yn = _as_2d_targets(y_new, name="y_new")
        if Xn.shape[0] != 1 or yn.shape != (1, 1):
            raise ValueError("update expects X_new shape (1,d) and y_new shape (1,1).")

        Xs = self._x_scaler.transform(Xn)
        ys = self._y_scaler.transform(yn)

        self._gp.set_XY(np.vstack([self._gp.X, Xs]), np.vstack([self._gp.Y, ys]))
        self._optimize()

    def log_likelihood(self) -> float:
        return float(self._gp.log_likelihood())


class MultiOutputGP:
    """Multi-output GP as independent SingleOutputGPs (one per output dimension).

    Shapes
    ------
    Train:   X (N,d), Y (N,m)
    Predict: X (n,d) -> mean (n,m), var (n,m)
    Update:  X_new (1,d), y_new (m,) or (1,m)
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
        return self._gps[0].n_train

    def predict(self, X: ArrayLike) -> tuple[FloatArray, FloatArray]:
        Xq = _as_2d_float(X, name="X")

        mean = np.empty((Xq.shape[0], self.n_out), dtype=float)
        var = np.empty((Xq.shape[0], self.n_out), dtype=float)

        for j, gp in enumerate(self._gps):
            mu_j, var_j = gp.predict(Xq)
            mean[:, j] = mu_j
            var[:, j] = var_j

        return mean, var

    def update(self, X_new: ArrayLike, y_new: ArrayLike) -> None:
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
        return float(sum(gp.log_likelihood() for gp in self._gps))
