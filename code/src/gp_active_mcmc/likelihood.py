from __future__ import annotations
import numpy as np


def loglike_gaussian_iid(y_sim: np.ndarray, y_obs: np.ndarray, sigma: float) -> float:
    """
    Log-likelihood (up to additive constant) for iid Gaussian noise:
      log L = -0.5/sigma^2 * sum_t (y_sim[t] - y_obs[t])^2
    """
    r = y_sim - y_obs
    return float(-0.5 * np.sum(r * r) / (sigma * sigma))


def loglike_theta(theta: np.ndarray, fwd, y_obs: np.ndarray, sigma: float) -> float:
    y_sim = fwd(theta)
    return loglike_gaussian_iid(y_sim, y_obs, sigma)
