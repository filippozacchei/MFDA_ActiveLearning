from __future__ import annotations
import numpy as np
import GPy


class GPSurrogate:
    """
    GP surrogate for log-likelihood with:
      - X standardization: x = (theta - X_mean)/X_std
      - y standardization: z = (logL - y_mean)/y_std
    Internally, the GP is fit on (x, z).
    Public predict returns mean/var in ORIGINAL (unscaled) logL units.
    """

    def __init__(self, 
                 X_train: np.ndarray, 
                 y_train: np.ndarray, 
                 kernel: str = "matern52", 
                 ard: bool = True,
                 gamma_L_ratio: float = 1.05, 
                 n_retrain_max: int = 500
               ) -> None:
        if X_train.ndim != 2:
            raise ValueError("X_train must be 2D (N, d).")
        if y_train.ndim != 1:
            raise ValueError("y_train must be 1D (N,).")

        self.X_mean = X_train.mean(axis=0)
        self.X_std = X_train.std(axis=0)
        self.X_std[self.X_std == 0.0] = 1.0

        self.y_mean = y_train.mean()
        self.y_std = y_train.std()
        if self.y_std == 0.0:
            self.y_std = 1.0

        Xs = self._x_scale(X_train)
        ys = self._y_scale(y_train).reshape(-1, 1)

        d = X_train.shape[1]
        if kernel == "rbf":
            kern = GPy.kern.RBF(input_dim=d, ARD=ard)
        elif kernel == "matern32":
            kern = GPy.kern.Matern32(input_dim=d, ARD=ard)
        elif kernel == "matern52":
            kern = GPy.kern.Matern52(input_dim=d, ARD=ard)  
        elif kernel == "mlp":
            kern = GPy.kern.MLP(input_dim=d, ARD=ard)      
        
        self.model = GPy.models.GPRegression(Xs, ys, kern)

        # reasonable initial noise; allow optimization to adjust
        self.model.Gaussian_noise.variance = 1e-1
        self.model.Gaussian_noise.unfix()

        self.model.optimize()

        # book-keeping for active learning dataset (in scaled coordinates)
        self.Xs = self.model.X.copy()
        self.ys = self.model.Y.copy()
        self._L_old = float(self.model.log_likelihood())
        self._retrain_count = 0
        self._gamma_L_ratio = gamma_L_ratio
        self._n_retrain_max = n_retrain_max
        self.optimize_params = True

    def _x_scale(self, X: np.ndarray) -> np.ndarray:
        return (X - self.X_mean) / self.X_std

    def _y_scale(self, y: np.ndarray) -> np.ndarray:
        return (y - self.y_mean) / self.y_std

    def _y_unscale_mean(self, mu_scaled: np.ndarray) -> np.ndarray:
        return mu_scaled * self.y_std + self.y_mean

    def _y_unscale_var(self, var_scaled: np.ndarray) -> np.ndarray:
        # If z = (y - m)/s, then Var[y] = Var[z] * s^2
        return var_scaled * (self.y_std ** 2)

    def predict(self, theta: np.ndarray) -> tuple[float, float]:
        """
        Returns (mu, var) of fw model.
        No further rescaling needed.
        """
        x = self._x_scale(theta.reshape(1, -1))
        mu_s, var_s = self.model.predict(x)  # mu, variance in scaled y space
        mu = float(self._y_unscale_mean(mu_s)[0, 0])
        var = float(self._y_unscale_var(var_s)[0, 0])
        return mu, var
    
    def __call__(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.predict(theta)

    def update(self, theta: np.ndarray, logL: float) -> None:
        """
        Add (theta, logL_true) to training set and (optionally) re-optimize.
        """
        x_new = self._x_scale(theta.reshape(1, -1))
        y_new = self._y_scale(np.array([logL])).reshape(1, 1)

        self.Xs = np.vstack([self.Xs, x_new])
        self.ys = np.vstack([self.ys, y_new])
        self.model.set_XY(self.Xs, self.ys)

        if self.optimize_params:
            L_new = float(self.model.log_likelihood())
            if (abs(L_new / self._L_old) > self._gamma_L_ratio) and (self._retrain_count < self._n_retrain_max):
                self.model.optimize()
                self._L_old = float(self.model.log_likelihood())
                self._retrain_count += 1
            else:
                self._L_old = L_new
            
    def log_likelihood(self) -> float:
        return float(self.model.log_likelihood())
    
    def stop_optimize(self):
        self.optimize_params = False

