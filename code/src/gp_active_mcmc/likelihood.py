from __future__ import annotations
import numpy as np


def loglike_gaussian(y_sim: np.ndarray, y_obs: np.ndarray, sigma: np.ndarray) -> float:
    """
    Log-likelihood (up to additive constant) for Gaussian noise:
      log L = -0.5 * sum_t ((y_sim[t] - y_obs[t]) / sigma[t])**2
    """
    assert sigma.shape == y_obs.shape == y_sim.shape
    r = (y_sim - y_obs)**2 / (sigma**2)
    return float(-0.5 * np.sum(r))


def loglike_theta(theta: np.ndarray, fwd, y_obs: np.ndarray, sigma: np.ndarray) -> float:
    """
    Standard likelihood at theta using fwd model (no GP adjustment)
    """
    assert sigma.shape == y_obs.shape
    y_sim = fwd(theta)
    return loglike_gaussian(y_sim, y_obs, sigma)
  

def loglike_theta_gp(theta: np.ndarray, fwd, y_obs: np.ndarray, sigma: np.ndarray) -> float: 
    """
    Likelihood adjusted for surrogate (GP) predictive uncertainty.
    fwd(theta) must return (y_sim, y_std)
    """
    assert sigma.shape == y_obs.shape
    y_sim, y_std = fwd(theta)
    sigma_tot = np.sqrt(sigma**2 + y_std**2) 
    return loglike_gaussian(y_sim, y_obs, sigma_tot)
