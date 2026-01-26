# from __future__ import annotations

import numpy as np

from gp_active_mcmc.toy import toy_forward, make_timeline, make_observation
from gp_active_mcmc.pod import POD
from gp_active_mcmc.gp import GPSurrogate
from gp_active_mcmc.podgp import PODGPSurrogate
from gp_active_mcmc.priors import GaussianPrior
from gp_active_mcmc.proposals import AdaptiveRWMProposal
from gp_active_mcmc.sampler import ALMCMC
from gp_active_mcmc.likelihood import loglike_theta_gp

from utils import plot_prediction_at_theta, plot_chain_2d

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
SEED = 123

# Surrogate parameters
POD_RANK = 10
GP_KERNEL = "matern52"
USE_ARD = True

# Initial design
N_INIT = 25

# Observation model
SIGMA_OBS = 0.01

# MCMC parameters
N_TOTAL = 5000
STEP_SCALE = 0.1

# Algorithm 1 controls
GAMMA_VAR = 0.01
GAMMA_L_RATIO = 1.05
N_RETRAIN_MAX = 500

def positive_tau(theta: np.ndarray) -> bool:
    """Simple positivity constraint on tau."""
    return theta[2] > 1e-6

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

rng = np.random.default_rng(SEED)

t = make_timeline(T=500, t_end=0.05)

prior_mean = np.array([0.8, 150.0, 0.010])
prior_cov = np.diag([0.5**2, 10.0**2, 0.01**2])
prior = GaussianPrior(prior_mean, prior_cov)
proposal = AdaptiveRWMProposal(cov=prior_cov)

theta_true = prior.sample(rng)

y_obs = make_observation(rng, theta_true, t, SIGMA_OBS)
sigma_obs = SIGMA_OBS * np.ones_like(y_obs)
fw = lambda theta: toy_forward(theta, t)

# --------------------------------------------------------------
# Initial surrogate construction
# --------------------------------------------------------------
X0 = np.array([prior.sample(rng) for _ in range(N_INIT)])
Y0 = np.array([fw(X0[i]) for i in range(N_INIT)])

pod = POD(r=POD_RANK).fit(Y0)
A0 = pod.project(Y0)

gps = [
    GPSurrogate(X0, 
                A0[:, k], 
                kernel=GP_KERNEL, 
                ard=USE_ARD, 
                gamma_L_ratio=GAMMA_L_RATIO,
                n_retrain_max=N_RETRAIN_MAX)
    for k in range(POD_RANK)
]

emul = PODGPSurrogate(pod=pod, gps=gps)

theta0 = prior_mean.copy()

plot_prediction_at_theta(
    emul=emul,
    theta=theta_true,
    t=t,
    y_obs=y_obs,
    title="Surrogate prediction at θ_true (initial)",
)

# --------------------------------------------------------------
# Run Algorithm 1
# --------------------------------------------------------------
loglike_surrogate = lambda theta: loglike_theta_gp(theta,emul,y_obs,sigma_obs)

sampler = ALMCMC(emul,fw,loglike_surrogate=loglike_surrogate,prior=prior,proposal=proposal,log_theta_ref=theta_true)
result = sampler.run(theta0=theta0,n_total=N_TOTAL,store_gp_ref=True)
chain = result["chain"]
used_forward = result["used_forward"]

print(f"Final acceptance rate : {result['accept_rate']:.3f}")
print(f"Forward-call fraction : {np.mean(used_forward):.3f}")

plot_chain_2d(
    chain=chain,
    used_forward=used_forward,
    theta_true=theta_true,
    names=("A", "f"),
)

plot_prediction_at_theta(
    emul=emul,
    theta=theta_true,
    t=t,
    y_obs=y_obs,
    title="Surrogate prediction at θ_true (final)",
)

# %%
