import numpy as np
import GPy


class SingleGP:
    """
    Single-output Gaussian Process surrogate with input/output scaling
    and optional active-learning retraining.
    """

    def __init__(
        self,
        X_train: np.ndarray,  # (N, d)
        Y_train: np.ndarray,  # (N,)
        kernel: str = "matern52",
        ard: bool = True,
        n_retrain_max: int = 20,
    ):
        # -------------------------
        # Input validation
        # -------------------------
        if X_train.ndim != 2:
            raise ValueError("X_train must be 2D (N, d)")
        if Y_train.ndim not in [1, 2]:
            raise ValueError("Y_train must be 1D or 2D")
        if Y_train.ndim == 1:
            Y_train = Y_train[:, None]

        if X_train.shape[0] != Y_train.shape[0]:
            raise ValueError("X_train and Y_train must have same number of rows")

        self.n_out = 1
        self.n_retrain_max = n_retrain_max
        self.retrain_count = 0
        self.counter = 0

        # -------------------------
        # Input scaling
        # -------------------------
        self.X_mean = X_train.mean(axis=0)
        self.X_std = X_train.std(axis=0)
        self.X_std[self.X_std == 0.0] = 1.0
        self.Xs = self._x_scale(X_train)

        # -------------------------
        # Output scaling
        # -------------------------
        self.Y_mean = Y_train.mean(axis=0)
        self.Y_std = Y_train.std(axis=0)
        self.Y_std[self.Y_std == 0.0] = 1.0
        self.Ys = self._y_scale(Y_train)

        # -------------------------
        # Build GP
        # -------------------------
        d = X_train.shape[1]
        if kernel == "rbf":
            kern = GPy.kern.RBF(d, ARD=ard)
        elif kernel == "matern32":
            kern = GPy.kern.Matern32(d, ARD=ard)
        elif kernel == "matern52":
            kern = GPy.kern.Matern52(d, ARD=ard)
        else:
            raise ValueError(f"Unknown kernel: {kernel}")

        self.gp = GPy.models.GPRegression(self.Xs, self.Ys, kern)
        self.gp.Gaussian_noise.variance = 1e-6
        self.gp.Gaussian_noise.unfix()
        self.gp.optimize()

    def _x_scale(self, X: np.ndarray) -> np.ndarray:
        return (X - self.X_mean) / self.X_std

    def _y_scale(self, Y: np.ndarray) -> np.ndarray:
        return (Y - self.Y_mean) / self.Y_std

    def _y_unscale(self, Y: np.ndarray) -> np.ndarray:
        return Y * self.Y_std + self.Y_mean

    def _add_XY(self, x_scaled: np.ndarray, y_scaled: np.ndarray):
        self.Xs = np.vstack([self.Xs, x_scaled])
        self.Ys = np.vstack([self.Ys, y_scaled])
        self.gp.set_XY(self.Xs, self.Ys)

    def _optimize(self):
        self.counter += 1
        if self.retrain_count < self.n_retrain_max and self.counter % 100 == 0:
            self.gp.optimize()
            self.retrain_count += 1

    def predict(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        theta = np.atleast_2d(theta)
        mu, var = self.gp.predict(self._x_scale(theta))
        mu = self._y_unscale(mu)
        var = var * (self.Y_std**2)
        return mu.ravel(), var.ravel()

    def update(self, theta: np.ndarray, y_true: np.ndarray) -> None:
        theta = np.atleast_2d(theta)
        y_true = np.atleast_2d(y_true).reshape(1, -1)
        y_scaled = self._y_scale(y_true)
        x_scaled = self._x_scale(theta)

        self._add_XY(x_scaled, y_scaled)
        self._optimize()

    def log_likelihood(self) -> float:
        return float(self.gp.log_likelihood())


# -----------------------------------------------------------
# Multi-output GP (independent outputs)
# -----------------------------------------------------------
class MultiOutputGP:
    """
    Multi-output GP with independent single-output GPs.
    """

    def __init__(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        kernel: str = "matern52",
        ard: bool = True,
        n_retrain_max: int = 20,
    ):
        if Y_train.ndim == 1:
            Y_train = Y_train[:, None]
        self.n_out = Y_train.shape[1]
        self.gps = [
            SingleGP(
                X_train,
                Y_train[:, i],
                kernel=kernel,
                ard=ard,
                n_retrain_max=n_retrain_max,
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
        y_true = np.atleast_1d(y_true)
        for i, gp in enumerate(self.gps):
            gp.update(theta, y_true[i])

    def log_likelihood(self) -> float:
        return sum(gp.log_likelihood() for gp in self.gps)
