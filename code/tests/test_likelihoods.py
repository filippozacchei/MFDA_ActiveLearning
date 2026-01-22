import numpy as np
import pytest
from gp_active_mcmc.likelihood import *

# Simple forward model for testing
def fwd(theta):
    return theta * np.array([1.0, 2.0, 3.0])

# GP-adjusted forward model for testing
def fwd_gp(theta):
    y_sim = theta * np.array([1.0, 2.0, 3.0])
    y_std = np.array([0.1, 0.2, 0.3])
    return y_sim, y_std

def test_loglike_gaussian():
    y_sim = np.array([1.0, 2.0, 3.0])
    y_obs = np.array([1.1, 1.9, 3.2])
    sigma = np.array([0.1, 0.1, 0.2])
    
    ll = loglike_gaussian(y_sim, y_obs, sigma)
    expected = -0.5 * np.sum(((y_sim - y_obs) / sigma)**2)
    assert np.isclose(ll, expected), "loglike_gaussian returned incorrect value"

def test_loglike_theta():
    theta = np.array([1.0, 1.0, 1.0])  # matches dimension of fwd input
    y_obs = np.array([1.0, 2.0, 3.0])
    sigma = np.array([0.1, 0.1, 0.1])
    
    ll = loglike_theta(theta, fwd, y_obs, sigma)
    y_sim = fwd(theta)
    expected = -0.5 * np.sum(((y_sim - y_obs) / sigma)**2)
    assert np.isclose(ll, expected), "loglike_theta returned incorrect value"

def test_loglike_theta_gp():
    theta = np.array([1.0, 1.0, 1.0])
    y_obs = np.array([1.0, 2.0, 3.0])
    sigma = np.array([0.1, 0.1, 0.1])
    
    ll = loglike_theta_gp(theta, fwd_gp, y_obs, sigma)
    y_sim, y_std = fwd_gp(theta)
    sigma_tot = np.sqrt(sigma**2 + y_std**2)
    expected = -0.5 * np.sum(((y_sim - y_obs) / sigma_tot)**2)
    assert np.isclose(ll, expected), "loglike_theta_gp_adjusted returned incorrect value"
