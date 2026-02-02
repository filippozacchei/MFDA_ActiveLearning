# %% Imports
from pathlib import Path
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
import arviz as az
from scipy.stats import multivariate_normal

import tinyDA as tda

from gp_active_mcmc.toy import toy_forward, make_timeline, make_observation
from gp_active_mcmc.pod import POD
from gp_active_mcmc.gp import GPSurrogate
from gp_active_mcmc.podgp import PODGPSurrogate
from gp_active_mcmc.active_mcmc_model import ActiveMCMCModel
from gp_active_mcmc.likelihood import GaussianLogLikeWithGP
from utils import plot_prediction_at_theta, plot_chain_2d

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
SEED = 123
N_TOTAL = 5000
N_BURNIN = 2000
N_INIT = 25
POD_RANK = 10
GP_KERNEL = "matern52"
USE_ARD = True
SIGMA_OBS = 0.1
GAMMA_VAR = 0.1
GAMMA_L_RATIO = 1.05
N_RETRAIN_MAX = 100
SUBSAMPLE_RATE = 5

rng = np.random.default_rng(SEED)
t = make_timeline(T=500, t_end=0.05)

# Prior and proposal
prior_mean = np.array([0.8, 150.0, 0.01])
prior_cov = np.diag([0.5**2, 25.0**2, 0.01**2])
prior = multivariate_normal(mean=prior_mean, cov=prior_cov)
proposal = tda.AdaptiveMetropolis(C0=prior_cov, sd=0.001, adaptive=True,period=100)

# Ground truth and observations
theta_true = prior.rvs(random_state=rng)
y_obs = make_observation(rng, theta_true, t, SIGMA_OBS)
sigma_obs = SIGMA_OBS * np.ones_like(y_obs)

# Forward model wrapper
def forward_model(theta: np.ndarray) -> np.ndarray:
    return toy_forward(theta, t)

# ---------------------------------------------------------------------
# Initial dataset for POD + GP
# ---------------------------------------------------------------------
X0 = np.array([prior.rvs(random_state=rng) for _ in range(N_INIT)])
Y0 = np.array([forward_model(theta) for theta in X0])

# POD
pod = POD(r=POD_RANK).fit(Y0)
A0 = pod.project(Y0)

# GP surrogates
gps = [
    GPSurrogate(
        X0,
        A0[:, k],
        kernel=GP_KERNEL,
        ard=USE_ARD,
        gamma_L_ratio=GAMMA_L_RATIO,
        n_retrain_max=N_RETRAIN_MAX
    )
    for k in range(POD_RANK)
]

# POD-GP surrogate
emul = PODGPSurrogate(pod=pod, gps=gps)

# Active surrogate model (coarse/fine)
model = ActiveMCMCModel(
    lf=emul,
    hf=forward_model,
    gamma_var=GAMMA_VAR
)

# ---------------------------------------------------------------------
# Likelihood and posterior
# ---------------------------------------------------------------------
likelihood_coarse = GaussianLogLikeWithGP(y_obs, sigma_obs * np.eye(len(y_obs)))
likelihood_fine = tda.AdaptiveGaussianLogLike(y_obs, sigma_obs * np.eye(len(y_obs)))

posterior_coarse = tda.Posterior(prior, likelihood_coarse, model.coarse)
posterior_fine = tda.Posterior(prior, likelihood_fine, model.fine)

# Visualization: initial surrogate prediction
plot_prediction_at_theta(emul, theta_true, t, y_obs, title="Surrogate prediction")

# ---------------------------------------------------------------------
# Run MCMC chain
# ---------------------------------------------------------------------
theta0 = prior_mean.copy()
samples = tda.sample(
    posteriors=[posterior_coarse],
    proposal=proposal,
    iterations=N_TOTAL,
    n_chains=1,
    force_sequential=True,
    initial_parameters=theta0,
    # store_coarse_chain=True,
    # subsampling_rate=SUBSAMPLE_RATE,
    # adaptive_error_model=None
)

# ---------------------------------------------------------------------
# Chain analysis
# ---------------------------------------------------------------------
chain_array = np.array([link.parameters for link in samples["chain_0"]])
forward_calls = np.array(model.used_hf[:N_TOTAL])
forward_post = forward_calls[N_BURNIN:]

# Acceptance rate
accepted = np.any(np.diff(chain_array, axis=0) != 0, axis=1)
accept_rate = accepted.mean()
forward_frac = forward_post.mean()

# RMSE vs ground truth
rmse = np.mean(np.sqrt(np.sum((chain_array[N_BURNIN:] - theta_true)**2, axis=1)))

print(f"Summary:")
print(f"  Acceptance rate      : {accept_rate:.3f}")
print(f"  Forward-call fraction: {forward_frac:.3f}")
print(f"  RMSE vs theta_true   : {rmse:.5f}")
print(f"Total forward calls   : {len(forward_calls)}")
print(f"Total chain length    : {len(chain_array)}")

# ---------------------------------------------------------------------
# Plot chains
# ---------------------------------------------------------------------
plot_chain_2d(
    chain_array,
    used_forward=forward_calls,
    theta_true=theta_true,
    names=("A","f"),
    title="Full chain"
)
plot_chain_2d(
    chain_array[N_BURNIN:],
    used_forward=None,
    theta_true=theta_true,
    names=("A","f"),
    title="Post burn-in"
)

# ---------------------------------------------------------------------
# Surrogate predictions
# ---------------------------------------------------------------------
y_pred, y_std = emul.predict(theta_true)
plot_prediction_at_theta(emul, theta_true, t, y_obs, title="Surrogate prediction")
plot_prediction_at_theta(emul, theta_true, t, forward_model(theta_true), title="Surrogate error")

# ---------------------------------------------------------------------
# Forward-call fraction over iterations
# ---------------------------------------------------------------------
plt.figure(figsize=(8,4))
plt.plot(np.cumsum(forward_calls)/np.arange(1, N_TOTAL+1))
plt.xlabel("Iteration")
plt.ylabel("Cumulative Forward-call fraction")
plt.title("Forward-call fraction over iterations")
plt.show()
