from __future__ import annotations
import numpy as np


def toy_forward(theta: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    3D parameter -> time series forward model.
    theta = [A, f, tau]
      A   : amplitude
      f   : frequency (Hz)
      tau : decay constant (s)
    """
    A, f, tau = float(theta[0]), float(theta[1]), float(theta[2])

    # Smooth, nonlinear-ish but stable, deterministic
    y = A * np.sin(2.0 * np.pi * f * t) * np.exp(-t / tau)
    # add a small nonlinear component so it's not too trivial
    y += 0.1 * (A**2) * np.cos(2.0 * np.pi * (0.5 * f) * t) * np.exp(-t / (1.5 * tau))
    return y


def make_timeline(T: int = 200, t_end: float = 0.02) -> np.ndarray:
    return np.linspace(0.0, t_end, T)


def make_observation(rng: np.random.Generator,
                     theta_true: np.ndarray,
                     t: np.ndarray,
                     sigma_obs: float) -> np.ndarray:
    y_clean = toy_forward(theta_true, t)
    return y_clean + rng.normal(0.0, sigma_obs, size=y_clean.shape)
