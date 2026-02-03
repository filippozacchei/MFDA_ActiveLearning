from typing import Protocol

import numpy as np

from .coarse_output import CoarseOutput


class ActiveLF(Protocol):
    def predict(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return surrogate prediction and uncertainty."""

    def update(self, theta: np.ndarray, y: np.ndarray) -> None:
        """Update surrogate with HF data."""


class HF(Protocol):
    def predict(self, theta: np.ndarray) -> np.ndarray:
        """Return HF prediction."""


class ActiveMCMCModel:
    """
    Provides coarse and fine model for MH,DA,MLDA.
    Updates surrogate whenever HF is evaluated.
    """

    def __init__(
        self,
        lf: ActiveLF,
        hf: HF,
        gamma_var: float,
    ):
        self.lf = lf
        self.hf = hf
        self.gamma_var = gamma_var

        self.n_hf = 0
        self.used_hf: list[bool] = []

    def coarse(self, theta: np.ndarray) -> np.ndarray:
        y_pred, var = self.lf.predict(theta)
        u_bar = float(np.mean(var))

        if u_bar > self.gamma_var:
            y = self.hf(theta)
            self.lf.update(theta, y)
            self.n_hf += 1
            self.used_hf.append(True)
            return y

        self.used_hf.append(False)
        return CoarseOutput(y_pred, var)

    def fine(self, theta: np.ndarray) -> np.ndarray:
        y = self.hf(theta)
        return y
