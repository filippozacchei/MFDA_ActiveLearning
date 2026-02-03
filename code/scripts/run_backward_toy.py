# %% Imports
import matplotlib.pyplot as plt
import numpy as np
import tinyDA as tda
from gp_active_mcmc.active_mcmc_model import ActiveMCMCModel
from gp_active_mcmc.gp import GPSurrogate
from gp_active_mcmc.likelihood import GaussianLogLikeWithGP
from gp_active_mcmc.pod import POD
from gp_active_mcmc.podgp import PODGPSurrogate
from gp_active_mcmc.toy import make_observation, make_timeline, toy_forward
from scipy.stats import multivariate_normal
from utils import plot_chain_2d, plot_prediction_at_theta

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
SEED = 1
N_TOTAL = 5000
N_BURNIN = 2000
N_INIT = 50
POD_RANK = 20
GP_KERNEL = "matern52"
USE_ARD = True
SIGMA_OBS = 0.01
GAMMA_VAR = 0.01
N_RETRAIN_MAX = 20
SUBSAMPLE_RATE = 5

rng = np.random.default_rng(SEED)
t = make_timeline(T=500, t_end=0.05)

# Prior and proposal
prior_mean = np.array([0.8, 150.0, 0.01])
prior_cov = np.diag([0.5**2, 40.0**2, 0.01**2])
prior = multivariate_normal(mean=prior_mean, cov=prior_cov)
proposal = tda.AdaptiveMetropolis(
    C0=prior_cov, sd=0.1, adaptive=True, period=100, gamma=1.01
)

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
        n_retrain_max=N_RETRAIN_MAX,
    )
    for k in range(POD_RANK)
]

# POD-GP surrogate
emul = PODGPSurrogate(pod=pod, gps=gps)

# Active surrogate model (coarse/fine)
model = ActiveMCMCModel(lf=emul, hf=forward_model, gamma_var=GAMMA_VAR)

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
N_TOTAL = N_TOTAL // SUBSAMPLE_RATE
N_BURNIN = N_BURNIN // SUBSAMPLE_RATE
chain = "chain_coarse_0"
samples = tda.sample(
    posteriors=[posterior_coarse, posterior_fine],
    proposal=proposal,
    iterations=N_TOTAL,
    n_chains=1,
    force_sequential=True,
    initial_parameters=theta0,
    store_coarse_chain=True,
    subsampling_rate=SUBSAMPLE_RATE,
    adaptive_error_model=None,
)

# ---------------------------------------------------------------------
# Chain analysis
# ---------------------------------------------------------------------
chain_array = np.array([link.parameters for link in samples[chain]])
forward_calls = np.array(model.used_hf[:N_TOTAL])
forward_post = forward_calls[N_BURNIN:]

# Acceptance rate
accepted = np.any(np.diff(chain_array, axis=0) != 0, axis=1)
accept_rate = accepted.mean()
forward_frac = forward_calls.mean()

# RMSE vs ground truth
rmse = np.mean(np.sqrt(np.sum((chain_array[N_BURNIN:] - theta_true) ** 2, axis=1)))

print("Summary:")
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
    names=("A", "f"),
    title="Full chain",
)
plot_chain_2d(
    chain_array[N_BURNIN:],
    used_forward=forward_calls[N_BURNIN:],
    theta_true=theta_true,
    names=("A", "f"),
    title="Post burn-in",
)

# ---------------------------------------------------------------------
# Surrogate predictions
# ---------------------------------------------------------------------
y_pred, y_std = emul.predict(theta_true)
plot_prediction_at_theta(emul, theta_true, t, y_obs, title="Surrogate prediction")
plot_prediction_at_theta(
    emul, theta_true, t, forward_model(theta_true), title="Surrogate error"
)

# ---------------------------------------------------------------------
# Forward-call fraction over iterations
# ---------------------------------------------------------------------
plt.figure(figsize=(8, 4))
plt.plot(np.cumsum(forward_calls) / np.arange(1, N_TOTAL + 1))
plt.xlabel("Iteration")
plt.ylabel("Cumulative Forward-call fraction")
plt.title("Forward-call fraction over iterations")
plt.show()
