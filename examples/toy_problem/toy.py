from __future__ import annotations
from typing import Callable

import numpy as np


def toy_forward(theta: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Simple stable time-series forward model.

    theta = [A, f, tau]
      A   : amplitude
      f   : frequency (Hz)
      tau : decay constant (s), must be > 0

    Output:
      y(t): 1D array with same shape as t
    """
    A = float(theta[0])
    f = float(theta[1])
    tau = float(theta[2])

    # Safety: keep the model defined even if Gaussian sampling yields tau <= 0
    tau = max(tau, 1e-6)

    # Single decaying sinusoid (bounded, smooth)
    y = A * np.sin(2.0 * np.pi * f * t) * np.exp(-t / tau)
    return y


def make_forward_model(t: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Wrap the toy forward model with the timeline baked in."""

    def _forward(theta: np.ndarray) -> np.ndarray:
        return toy_forward(theta, t)

    return _forward


def make_timeline(T: int = 200, t_end: float = 0.02) -> np.ndarray:
    return np.linspace(0.0, t_end, T)


def make_observation(
    rng: np.random.Generator, theta_true: np.ndarray, t: np.ndarray, sigma_obs: float
) -> np.ndarray:
    y_clean = toy_forward(theta_true, t)
    return y_clean + rng.normal(0.0, sigma_obs, size=y_clean.shape)
