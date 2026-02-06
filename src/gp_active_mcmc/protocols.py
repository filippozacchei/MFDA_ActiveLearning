from typing import Protocol
import numpy as np


class ActiveSurrogate(Protocol):
    """Protocol for a surrogate model with active learning capability."""

    def predict(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return surrogate prediction and associated uncertainty."""

    def update(self, theta: np.ndarray, y: np.ndarray) -> None:
        """Update surrogate with a high-fidelity (HF) observation."""


class HighFidelityModel(Protocol):
    """Protocol for a high-fidelity forward model."""

    def __call__(self, theta: np.ndarray) -> np.ndarray:
        """Return high-fidelity model prediction."""
