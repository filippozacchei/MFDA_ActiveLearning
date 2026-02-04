from __future__ import annotations
import copy
import numpy as np
from dataclasses import dataclass

from .gp import MultiOutputGP
from .pod import POD


@dataclass
class PODGPSurrogate:
    pod: POD
    gp: MultiOutputGP
    coeff_var_floor: float = 1e-12

    def reconstruct_var(self, var_a: np.ndarray) -> np.ndarray:
        Phi = self.pod.phi_  # (T,r)
        y_var = (Phi**2) @ var_a
        return np.maximum(y_var, 1e-14)

    def predict_coeffs(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        theta = np.atleast_2d(theta)
        mu, var = self.gp.predict(theta)
        mu = mu.ravel()
        var = np.maximum(var.ravel(), self.coeff_var_floor)
        return mu, var

    def predict(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mu_a, var_a = self.predict_coeffs(theta)
        y_hat = self.pod.reconstruct(mu_a.reshape(1, -1))[0]
        y_var = self.reconstruct_var(var_a)
        return y_hat, y_var

    def update(self, theta: np.ndarray, y_true: np.ndarray):
        theta = np.atleast_2d(theta)
        y_true = np.atleast_2d(y_true)
        a_true = self.pod.project(y_true)[0]
        self.gp.update(theta, a_true)

    def log_likelihood(self) -> float:
        return self.gp.log_likelihood()

    def copy(self) -> "PODGPSurrogate":
        return PODGPSurrogate(
            pod=copy.deepcopy(self.pod),
            gp=copy.deepcopy(self.gp),
            coeff_var_floor=self.coeff_var_floor,
        )
