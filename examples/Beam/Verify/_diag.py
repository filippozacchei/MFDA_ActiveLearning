"""Quick diagnostic for backward.py parameters."""
import sys; sys.path.insert(0, '../../src')
import numpy as np
from scipy.stats import multivariate_normal
from beam import make_spatial_grid, make_forward_model, make_observation
from gp_active_mcmc.utils.rng import set_seed

rng = set_seed(2)
x = make_spatial_grid(n_pts=31, length=1.0)
obs_idx = np.array([2,5,8,11,14,17,20,23,26,29])
loads = np.array([
    13.944211,14.107554,14.168484,14.127543,14.080133,14.031762,14.037079,
    13.940349,13.887439,13.994669,14.138576,14.341531,14.501729,14.681951,
    14.879436,15.143519,15.300596,15.375463,15.359368,15.278929,15.114428,
    14.966691,14.792335,14.662425,14.541461,14.426502,14.309434,14.195700,
    14.127510,13.982456,13.863596,
])

hf_forward = make_forward_model(x=x, obs_idx=obs_idx, load=-loads, return_full_state=False)
prior_mean = np.array([10.0, 10.0, 10.0])
y_ref = hf_forward(prior_mean)
signal_scale = float(np.max(np.abs(y_ref)))
sigma_obs = 0.04 * signal_scale
theta_true = np.array([9.3, 9.3, 9.2])
y_true = hf_forward(theta_true)

print(f"signal_scale   = {signal_scale:.6e}")
print(f"sigma_obs(4%)  = {sigma_obs:.6e}")
print(f"sigma_obs(2%)  = {0.02*signal_scale:.6e}")
print(f"y_true range   = [{y_true.min():.6e}, {y_true.max():.6e}]")
print(f"SNR at 4%      = {signal_scale/sigma_obs:.1f}")
print(f"SNR at 2%      = {signal_scale/(0.02*signal_scale):.1f}")

# Check GP initial training set quality
prior_cov = np.diag([2.0**2, 2.0**2, 2.0**2])
prior = multivariate_normal(mean=prior_mean, cov=prior_cov)
y_obs = make_observation(rng, theta_true, x, sigma_obs, obs_idx, load=-loads)
print(f"\ny_obs = {y_obs}")
print(f"y_true= {y_true}")
print(f"residual = {np.linalg.norm(y_obs - y_true):.6e}")

theta_train = np.asarray([prior.rvs(random_state=rng) for _ in range(10)], dtype=float)
y_train = np.asarray([hf_forward(th) for th in theta_train], dtype=float)
print(f"\ntheta_train:\n{theta_train}")
print(f"\ny_train range per output: min={y_train.min(axis=0)}, max={y_train.max(axis=0)}")
print(f"theta_train range: [{theta_train.min(axis=0)}, {theta_train.max(axis=0)}]")
print(f"theta_true in training range? {np.all(theta_train.min(axis=0) <= theta_true) and np.all(theta_true <= theta_train.max(axis=0))}")
