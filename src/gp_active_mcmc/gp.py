from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import GPy


KernelName = Literal["rbf", "matern32", "matern52"]


def _ensure_2d(X: np.ndarray, *, name: str) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim == 1:
        return X[None, :]
    if X.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {X.shape}")
    return X


def _ensure_col(y: np.ndarray, *, name: str) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim == 1:
        return y[:, None]
    if y.ndim == 2 and y.shape[1] == 1:
        return y
    raise ValueError(f"{name} must be 1D or (N,1), got shape {y.shape}")


def _make_kernel(d: int, *, kernel: KernelName, ard: bool) -> GPy.kern.Kern:
    if kernel == "rbf":
        return GPy.kern.RBF(d, ARD=ard)
    if kernel == "matern32":
        return GPy.kern.Matern32(d, ARD=ard)
    if kernel == "matern52":
        return GPy.kern.Matern52(d, ARD=ard)
    raise ValueError(f"Unknown kernel: {kernel}")


@dataclass(frozen=True)
class _StandardScaler:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray) -> "_StandardScaler":
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std = np.where(std == 0.0, 1.0, std)
        return cls(mean=mean, std=std)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std

    def inverse_transform(self, Xs: np.ndarray) -> np.ndarray:
        return Xs * self.std + self.mean


class SingleGP:
    """
    Single-output Gaussian Process surrogate with input/output scaling
    and optional active-learning retraining.
    """

    def __init__(
        self,
        X_train: np.ndarray,  # (N, d)
        Y_train: np.ndarray,  # (N,) or (N,1)
        kernel: KernelName = "matern52",
        ard: bool = True,
        n_retrain_max: int = 20,
        update_every: int = 10,
    ):
        X_train = _ensure_2d(X_train, name="X_train")
        Y_train = _ensure_col(Y_train, name="Y_train")
        if X_train.shape[0] != Y_train.shape[0]:
            raise ValueError("X_train and Y_train must have the same number of rows")

        self.n_out = 1
        self.n_retrain_max = int(n_retrain_max)
        self.update_every = int(update_every)
        self.retrain_count = 0
        self.counter = 0

        self._x_scaler = _StandardScaler.fit(X_train)
        self._y_scaler = _StandardScaler.fit(Y_train)

        Xs = self._x_scaler.transform(X_train)
        Ys = self._y_scaler.transform(Y_train)

        kern = _make_kernel(X_train.shape[1], kernel=kernel, ard=ard)

        self.gp = GPy.models.GPRegression(Xs, Ys, kern)
        self.gp.Gaussian_noise.variance = 1e-6
        self.gp.Gaussian_noise.unfix()
        self.gp.optimize()

    def _maybe_optimize(self) -> None:
        self.counter += 1
        if self.retrain_count >= self.n_retrain_max:
            return
        if self.update_every <= 0:
            return
        if self.counter % self.update_every == 0:
            self.gp.optimize()
            self.retrain_count += 1

    def predict(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        theta = _ensure_2d(theta, name="theta")
        mu_s, var_s = self.gp.predict(self._x_scaler.transform(theta))
        mu = self._y_scaler.inverse_transform(mu_s)
        var = var_s * (self._y_scaler.std**2)
        return mu.ravel(), var.ravel()

    def update(self, theta: np.ndarray, y_true: np.ndarray) -> None:
        theta = _ensure_2d(theta, name="theta")
        y_true = _ensure_col(y_true, name="y_true")
        if theta.shape[0] != 1 or y_true.shape[0] != 1:
            raise ValueError("update expects a single theta and a single y value")

        x_s = self._x_scaler.transform(theta)
        y_s = self._y_scaler.transform(y_true)

        Xs = np.vstack([self.gp.X, x_s])
        Ys = np.vstack([self.gp.Y, y_s])
        self.gp.set_XY(Xs, Ys)
        self._maybe_optimize()

    def log_likelihood(self) -> float:
        return float(self.gp.log_likelihood())


class MultiOutputGP:
    """Multi-output GP with independent single-output GPs."""

    def __init__(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,  # (N, m)
        kernel: KernelName = "matern52",
        ard: bool = True,
        n_retrain_max: int = 20,
        update_every: int = 10,
    ):
        X_train = _ensure_2d(X_train, name="X_train")
        Y_train = np.asarray(Y_train)
        if Y_train.ndim == 1:
            Y_train = Y_train[:, None]
        if Y_train.ndim != 2:
            raise ValueError("Y_train must be 2D (N, m)")
        if X_train.shape[0] != Y_train.shape[0]:
            raise ValueError("X_train and Y_train must have the same number of rows")

        self.n_out = Y_train.shape[1]
        self.gps = [
            SingleGP(
                X_train,
                Y_train[:, i],
                kernel=kernel,
                ard=ard,
                n_retrain_max=n_retrain_max,
                update_every=update_every,
            )
            for i in range(self.n_out)
        ]

    def predict(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mus, vars_ = [], []
        for gp in self.gps:
            mu, var = gp.predict(theta)
            mus.append(mu)
            vars_.append(var)
        return np.array(mus), np.array(vars_)

    def update(self, theta: np.ndarray, y_true: np.ndarray) -> None:
        y_true = np.asarray(y_true).ravel()
        if y_true.shape[0] != self.n_out:
            raise ValueError(f"y_true must have length {self.n_out}")
        for i, gp in enumerate(self.gps):
            gp.update(theta, np.array([[y_true[i]]]))

    def log_likelihood(self) -> float:
        return float(sum(gp.log_likelihood() for gp in self.gps))


# ===========================================================
# N-level autoregressive co-kriging
# ===========================================================
class AutoRegressiveCoKrigingN:
    """
    N-level auto-regressive surrogate (Kennedy–O'Hagan AR model):

        y_0(x) = GP_0(x)
        y_l(x) = rho_l * y_{l-1}(x) + δ_l(x),  for l = 1..L-1

    where:
    - GP_0 is a GP on level-0 data
    - δ_l is a GP on the discrepancy at level l
    - rho_l is estimated by least squares on the level-l design using mean predictions
      from level (l-1)

    Public API
    ----------
    - predict(theta, level=...) -> (mu, var)
      returns the marginal prediction of y_level(x)
    - update(theta, y_true, level=...) updates either base GP (level=0) or a discrepancy GP
      (level>=1). rho is refreshed using the new sample (local update).
    """

    def __init__(
        self,
        X_levels: list[np.ndarray],  # [X0, X1, ..., X_{L-1}]
        Y_levels: list[np.ndarray],  # [y0, y1, ..., y_{L-1}] each (N_l,) or (N_l,1)
        *,
        kernel_base: KernelName = "matern52",
        kernel_delta: KernelName = "matern52",
        ard: bool = True,
        n_retrain_max: int = 20,
        update_every: int = 10,
        ridge: float = 1e-12,
    ):
        if len(X_levels) != len(Y_levels):
            raise ValueError("X_levels and Y_levels must have the same length")
        if len(X_levels) < 1:
            raise ValueError("At least one level is required")

        self.ridge = float(ridge)
        self.n_levels = len(X_levels)

        X0 = _ensure_2d(X_levels[0], name="X_levels[0]")
        y0 = _ensure_col(Y_levels[0], name="Y_levels[0]")
        if X0.shape[0] != y0.shape[0]:
            raise ValueError("X_levels[0] and Y_levels[0] row counts must match")

        # Base GP for level 0
        self.base = SingleGP(
            X0,
            y0,
            kernel=kernel_base,
            ard=ard,
            n_retrain_max=n_retrain_max,
            update_every=update_every,
        )

        # For l>=1: store rho_l and discrepancy GP δ_l trained on level-l data
        self.rhos: list[float] = []
        self.deltas: list[SingleGP] = []

        d = X0.shape[1]
        for l in range(1, self.n_levels):
            Xl = _ensure_2d(X_levels[l], name=f"X_levels[{l}]")
            yl = _ensure_col(Y_levels[l], name=f"Y_levels[{l}]")
            if Xl.shape[0] != yl.shape[0]:
                raise ValueError(
                    f"X_levels[{l}] and Y_levels[{l}] row counts must match"
                )
            if Xl.shape[1] != d:
                raise ValueError(
                    f"All levels must share the same input dimension (expected {d})"
                )

            # Estimate rho_l via LS on level-l design using E[y_{l-1}(Xl)]
            mu_prev, _ = self.predict(Xl, level=l - 1)
            mu_prev = mu_prev.reshape(-1, 1)
            denom = float(mu_prev.T @ mu_prev + self.ridge)
            rho_l = float((mu_prev.T @ yl) / denom)

            y_delta = yl - rho_l * mu_prev
            delta_gp = SingleGP(
                Xl,
                y_delta,
                kernel=kernel_delta,
                ard=ard,
                n_retrain_max=n_retrain_max,
                update_every=update_every,
            )

            self.rhos.append(rho_l)
            self.deltas.append(delta_gp)

        self.n_out = 1  # scalar output

    def predict(
        self, theta: np.ndarray, *, level: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict y_level(theta).

        Parameters
        ----------
        theta
            (N,d) or (d,) points.
        level
            Level index in [0, n_levels-1].

        Returns
        -------
        mu, var
            Arrays of shape (N,).
        """
        theta = _ensure_2d(theta, name="theta")
        if not (0 <= level < self.n_levels):
            raise ValueError(f"level must be in [0, {self.n_levels - 1}]")

        # level 0
        mu, var = self.base.predict(theta)

        # propagate AR corrections up to requested level
        for l in range(1, level + 1):
            rho = self.rhos[l - 1]
            mu_d, var_d = self.deltas[l - 1].predict(theta)
            mu = rho * mu + mu_d
            var = (
                rho**2
            ) * var + var_d  # ignores cross-cov terms (common approximation)

        return mu.ravel(), var.ravel()

    def update(self, theta: np.ndarray, y_true: np.ndarray, *, level: int) -> None:
        """
        Add a single observation at a given level.

        Notes
        -----
        - level=0 updates the base GP
        - level>=1 updates discrepancy GP δ_level and refreshes rho_level using a local LS step
          based on the current prediction at level-1.
        """
        theta = _ensure_2d(theta, name="theta")
        y_true = _ensure_col(y_true, name="y_true")
        if theta.shape[0] != 1 or y_true.shape[0] != 1:
            raise ValueError("update expects a single theta and a single y value")
        if not (0 <= level < self.n_levels):
            raise ValueError(f"level must be in [0, {self.n_levels - 1}]")

        if level == 0:
            self.base.update(theta, y_true)
            return

        # Predict previous level at theta (mean only)
        mu_prev, _ = self.predict(theta, level=level - 1)
        mu_prev = mu_prev.reshape(1, 1)

        # Refresh rho_level with a one-step ridge regression update:
        # rho <- rho + (mu^T (y - rho mu)) / (mu^T mu + ridge)
        idx = level - 1
        rho = self.rhos[idx]
        resid = y_true - rho * mu_prev
        rho = float(rho + (mu_prev.T @ resid) / (mu_prev.T @ mu_prev + self.ridge))
        self.rhos[idx] = rho

        # Update discrepancy GP with δ = y - rho * y_{l-1}
        y_delta = y_true - rho * mu_prev
        self.deltas[idx].update(theta, y_delta)

    def log_likelihood(self) -> float:
        ll = self.base.log_likelihood()
        ll += float(sum(d.log_likelihood() for d in self.deltas))
        return float(ll)


class MultiOutputAutoRegressiveCoKrigingN:
    """
    Multi-output wrapper: independent N-level AR co-kriging per output dimension.

    Inputs
    ------
    Y_levels[j] must be (N_l, m) for each level l (or 1D -> treated as m=1).
    """

    def __init__(
        self,
        X_levels: list[np.ndarray],
        Y_levels: list[np.ndarray],
        *,
        kernel_base: KernelName = "matern52",
        kernel_delta: KernelName = "matern52",
        ard: bool = True,
        n_retrain_max: int = 20,
        update_every: int = 10,
        ridge: float = 1e-12,
    ):
        if len(X_levels) != len(Y_levels):
            raise ValueError("X_levels and Y_levels must have the same length")
        if len(X_levels) < 1:
            raise ValueError("At least one level is required")

        Ys0 = np.asarray(Y_levels[0])
        if Ys0.ndim == 1:
            Ys0 = Ys0[:, None]
        if Ys0.ndim != 2:
            raise ValueError("Y_levels[0] must be 2D (N, m) or 1D (N,)")

        self.n_levels = len(X_levels)
        self.n_out = Ys0.shape[1]

        # Normalize all Y_levels to (N_l, m)
        Ys_levels: list[np.ndarray] = []
        for l, Yl in enumerate(Y_levels):
            Yl = np.asarray(Yl)
            if Yl.ndim == 1:
                Yl = Yl[:, None]
            if Yl.ndim != 2 or Yl.shape[1] != self.n_out:
                raise ValueError(f"Y_levels[{l}] must have shape (N, {self.n_out})")
            Ys_levels.append(Yl)

        self.models = [
            AutoRegressiveCoKrigingN(
                X_levels=X_levels,
                Y_levels=[Ys_levels[l][:, j] for l in range(self.n_levels)],
                kernel_base=kernel_base,
                kernel_delta=kernel_delta,
                ard=ard,
                n_retrain_max=n_retrain_max,
                update_every=update_every,
                ridge=ridge,
            )
            for j in range(self.n_out)
        ]

    def predict(
        self, theta: np.ndarray, *, level: int
    ) -> tuple[np.ndarray, np.ndarray]:
        mus, vars_ = [], []
        for m in self.models:
            mu, var = m.predict(theta, level=level)
            mus.append(mu)
            vars_.append(var)
        return np.array(mus), np.array(vars_)

    def update(self, theta: np.ndarray, y_true: np.ndarray, *, level: int) -> None:
        y_true = np.asarray(y_true).ravel()
        if y_true.shape[0] != self.n_out:
            raise ValueError(f"y_true must have length {self.n_out}")
        for j, m in enumerate(self.models):
            m.update(theta, np.array([[y_true[j]]]), level=level)

    def log_likelihood(self) -> float:
        return float(sum(m.log_likelihood() for m in self.models))
