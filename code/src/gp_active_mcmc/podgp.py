import copy
from dataclasses import dataclass

import numpy as np

from .gp import GPSurrogate
from .pod import POD


@dataclass
class PODGPSurrogate:
    pod: POD
    gps: list[GPSurrogate]
    coeff_var_floor: float = 1e-12

    def predict_coeffs(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r = len(self.gps)
        mu = np.zeros(r)
        var = np.zeros(r)
        for k, gpk in enumerate(self.gps):
            mk, vk = gpk.predict(theta)  # scalar mean/var
            mu[k] = float(mk)
            var[k] = float(vk)
        var = np.maximum(var, self.coeff_var_floor)
        return mu, var

    def predict(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mu_a, var_a = self.predict_coeffs(theta)
        y_hat = self.pod.reconstruct(mu_a.reshape(1, -1))[0]
        Phi = self.pod.phi_  # (T,r)
        y_var = (Phi**2) @ var_a
        y_std = np.sqrt(np.maximum(y_var, 1e-14))
        return y_hat, y_std

    def __call__(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.predict(theta)

    def update(self, theta: np.ndarray, y_true: np.ndarray):
        a_true = self.pod.project(y_true.reshape(1, -1))[0]
        for k, gpk in enumerate(self.gps):
            gpk.update(theta, float(a_true[k]))

    def log_likelihood(self) -> float:
        total_ll = 0.0
        for gpk in self.gps:
            total_ll += float(gpk.model.log_likelihood())
        return total_ll

    def copy(self) -> "PODGPSurrogate":
        pod_copy = copy.deepcopy(self.pod)
        gps_copy = [copy.deepcopy(g) for g in self.gps]
        return PODGPSurrogate(
            pod=pod_copy, gps=gps_copy, coeff_var_floor=self.coeff_var_floor
        )

    def stop_optimize(self):
        for gpk in self.gps:
            gpk.stop_optimize()
