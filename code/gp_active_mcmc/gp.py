from __future__ import annotations

import GPy
import numpy as np


class GPSurrogate:
    """
    GP surrogate for a forward model y(theta), with uncertainty.
    """

    def __init__(
        self,
        X_train: np.ndarray,  # (N, d)
        Y_train: np.ndarray,  # (N, m)
        kernel: str = "matern52",
        ard: bool = True,
        gamma_L_ratio: float = 1.05,
        n_retrain_max: int = 500,
    ) -> None:
        if X_train.ndim != 2:
            raise ValueError("X_train must be 2D (N, d).")

        if Y_train.ndim == 1:
            Y_train = Y_train[:, None]
        elif Y_train.ndim != 2:
            raise ValueError("Y_train must be 1D or 2D.")

        self.n_out = Y_train.shape[1]

        # -------------------------
        # Input scaling
        # -------------------------
        self.X_mean = X_train.mean(axis=0)
        self.X_std = X_train.std(axis=0)
        self.X_std[self.X_std == 0.0] = 1.0
        Xs = self._x_scale(X_train)

        # -------------------------
        # Output scaling
        # -------------------------
        self.Y_mean = Y_train.mean(axis=0)
        self.Y_std = Y_train.std(axis=0)
        self.Y_std[self.Y_std == 0.0] = 1.0
        Ys = self._y_scale(Y_train)

        # -------------------------
        # Build one GP per output
        # -------------------------
        self.models = []
        self.Xs = Xs.copy()
        self.Ys = Ys.copy()

        d = X_train.shape[1]
        for k in range(self.n_out):
            if kernel == "rbf":
                kern = GPy.kern.RBF(d, ARD=ard)
            elif kernel == "matern32":
                kern = GPy.kern.Matern32(d, ARD=ard)
            elif kernel == "matern52":
                kern = GPy.kern.Matern52(d, ARD=ard)
            else:
                raise ValueError(f"Unknown kernel: {kernel}")

            gp = GPy.models.GPRegression(Xs, Ys[:, [k]], kern)
            gp.Gaussian_noise.variance = 1e-6
            gp.Gaussian_noise.unfix()
            gp.optimize()

            self.models.append(gp)

        # -------------------------
        # Active-learning bookkeeping
        # -------------------------
        self._L_old = np.array([m.log_likelihood() for m in self.models])
        self._gamma_L_ratio = gamma_L_ratio
        self._n_retrain_max = n_retrain_max
        self._retrain_count = 0
        self.optimize_params = True

    def _x_scale(self, X: np.ndarray) -> np.ndarray:
        return (X - self.X_mean) / self.X_std

    def _y_scale(self, Y: np.ndarray) -> np.ndarray:
        return (Y - self.Y_mean) / self.Y_std

    def _y_unscale(self, Y: np.ndarray) -> np.ndarray:
        return Y * self.Y_std + self.Y_mean

    def predict(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict forward model output and variance.

        Returns
        -------
        y_mean : (m,)
        y_var  : (m,)
        """
        x = self._x_scale(theta.reshape(1, -1))

        mu = np.zeros(self.n_out)
        var = np.zeros(self.n_out)

        for k, gp in enumerate(self.models):
            mu_s, var_s = gp.predict(x)
            mu[k] = mu_s[0, 0] * self.Y_std[k] + self.Y_mean[k]
            var[k] = var_s[0, 0] * (self.Y_std[k] ** 2)

        return mu, var

    def __call__(self, theta: np.ndarray):
        return self.predict(theta)

    def update(self, theta: np.ndarray, y_true: np.ndarray) -> None:
        """
        Add a new HF observation and (optionally) retrain hyperparameters.
        """

        x_new = self._x_scale(theta.reshape(1, -1))
        y_new = self._y_scale(y_true)

        self.Xs = np.vstack([self.Xs, x_new])
        self.Ys = np.vstack([self.Ys, y_new])

        if self._retrain_count <= self._n_retrain_max:
            for k, gp in enumerate(self.models):
                gp.set_XY(self.Xs, self.Ys[:, [k]])
                L_new = gp.log_likelihood()
                if abs(L_new / self._L_old[k]) > self._gamma_L_ratio:
                    gp.optimize()
                self._L_old[k] = gp.log_likelihood()
        self._retrain_count += 1
