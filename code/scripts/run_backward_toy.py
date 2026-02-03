# %% Imports
import matplotlib.pyplot as plt
import numpy as np
import copy
import tinyDA as tda

from gp_active_mcmc.active_mcmc_model import AdaptiveActiveMCMCModel
from gp_active_mcmc.gp import GPSurrogate
from gp_active_mcmc.likelihood import GaussianLogLikeWithGP
from gp_active_mcmc.pod import POD
from gp_active_mcmc.active_mcmc_model import ActiveMCMCModel
from gp_active_mcmc.podgp import PODGPSurrogate
from gp_active_mcmc.toy import make_observation, make_timeline, toy_forward
from scipy.stats import multivariate_normal
from utils import plot_chain_2d

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
FIXED_SUBSAMPLE = 5

rng = np.random.default_rng(SEED)
t = make_timeline(T=500, t_end=0.05)

# ---------------------------------------------------------------------
# Prior and proposal
# ---------------------------------------------------------------------
prior_mean = np.array([0.8, 150.0, 0.01])
prior_cov = np.diag([0.5**2, 40.0**2, 0.01**2])
prior = multivariate_normal(mean=prior_mean, cov=prior_cov)
proposal = tda.AdaptiveMetropolis(
    C0=prior_cov, sd=0.1, adaptive=True, period=100, gamma=1.01
)

# ---------------------------------------------------------------------
# Ground truth and observations
# ---------------------------------------------------------------------
theta_true = prior.rvs(random_state=rng)
y_obs = make_observation(rng, theta_true, t, SIGMA_OBS)
sigma_obs = SIGMA_OBS * np.ones_like(y_obs)


# ---------------------------------------------------------------------
# Forward model wrapper
# ---------------------------------------------------------------------
def forward_model(theta: np.ndarray) -> np.ndarray:
    return toy_forward(theta, t)


# ---------------------------------------------------------------------
# Initial dataset for POD + GP
# ---------------------------------------------------------------------
X0 = np.array([prior.rvs(random_state=rng) for _ in range(N_INIT)])
Y0 = np.array([forward_model(theta) for theta in X0])

pod = POD(r=POD_RANK).fit(Y0)
A0 = pod.project(Y0)

gps = [
    GPSurrogate(
        X0, A0[:, k], kernel=GP_KERNEL, ard=USE_ARD, n_retrain_max=N_RETRAIN_MAX
    )
    for k in range(POD_RANK)
]
emul_base = PODGPSurrogate(pod=pod, gps=gps)

# ---------------------------------------------------------------------
# Setup three strategies (deep copies)
# ---------------------------------------------------------------------
# 1) Coarse-only (never call HF)
model_coarse_only = copy.deepcopy(emul_base)
# Wrap in minimal ActiveMCMCModel interface for coarse-only

model_coarse_only = ActiveMCMCModel(
    lf=copy.deepcopy(emul_base), hf=forward_model, gamma_var=GAMMA_VAR
)

# 2) Coarse+fine fixed subsample
model_fixed = copy.deepcopy(emul_base)
model_fixed = ActiveMCMCModel(
    lf=copy.deepcopy(emul_base), hf=forward_model, gamma_var=GAMMA_VAR
)

# 3) Coarse+fine adaptive subsample
model_adaptive = AdaptiveActiveMCMCModel(
    lf=copy.deepcopy(emul_base),
    hf=forward_model,
    gamma_var=GAMMA_VAR,
    initial_subchain=5,
)  # adaptive


# ---------------------------------------------------------------------
# Define likelihoods and posteriors
# ---------------------------------------------------------------------
def create_posteriors(model):
    likelihood_coarse = GaussianLogLikeWithGP(y_obs, sigma_obs * np.eye(len(y_obs)))
    likelihood_fine = tda.AdaptiveGaussianLogLike(y_obs, sigma_obs * np.eye(len(y_obs)))
    posterior_coarse = tda.Posterior(prior, likelihood_coarse, model.coarse)
    posterior_fine = tda.Posterior(prior, likelihood_fine, model.fine)
    return posterior_coarse, posterior_fine


post_coarse_only, _ = create_posteriors(model_coarse_only)
post_fixed, post_fixed_fine = create_posteriors(model_fixed)
post_adaptive, post_adaptive_fine = create_posteriors(model_adaptive)


# ---------------------------------------------------------------------
# Run MCMC for each strategy
# ---------------------------------------------------------------------
def run_chain(model, posterior, theta0, n_total, subsample_length):
    N_total_subsampled = n_total // subsample_length
    N_burnin_subsampled = N_BURNIN // subsample_length

    samples = tda.sample(
        posteriors=[posterior],
        proposal=proposal,
        iterations=N_total_subsampled,
        n_chains=1,
        force_sequential=True,
        initial_parameters=theta0,
        store_coarse_chain=True,
        subsampling_rate=model.subchain_length
        if hasattr(model, "subchain_length")
        else FIXED_SUBSAMPLE,
        adaptive_error_model=None,
    )

    chain_array = np.array([link.parameters for link in samples["chain_coarse_0"]])
    forward_calls = np.array(model.used_hf[:N_total_subsampled])

    return chain_array, forward_calls, N_total_subsampled, N_burnin_subsampled


theta0 = prior_mean.copy()

results = {}
strategies = {
    "coarse_only": (model_coarse_only, post_coarse_only, 1),
    "fixed_subsample": (model_fixed, post_fixed, FIXED_SUBSAMPLE),
    "adaptive_subsample": (
        model_adaptive,
        post_adaptive,
        model_adaptive.subchain_length,
    ),
}

for name, (mod, post, subsample) in strategies.items():
    chain_array, forward_calls, N_total_sub, N_burnin_sub = run_chain(
        mod, post, theta0, N_TOTAL, subsample
    )
    results[name] = (chain_array, forward_calls, N_total_sub, N_burnin_sub)

# ---------------------------------------------------------------------
# Analyze and plot results
# ---------------------------------------------------------------------
for name, (chain_array, forward_calls, N_total_sub, N_burnin_sub) in results.items():
    # Acceptance rate
    accepted = np.any(np.diff(chain_array, axis=0) != 0, axis=1)
    accept_rate = accepted.mean()
    forward_frac = forward_calls.mean()
    rmse = np.mean(
        np.sqrt(np.sum((chain_array[N_burnin_sub:] - theta_true) ** 2, axis=1))
    )

    print(f"\nStrategy: {name}")
    print(f"  Acceptance rate      : {accept_rate:.3f}")
    print(f"  Forward-call fraction: {forward_frac:.3f}")
    print(f"  RMSE vs theta_true   : {rmse:.5f}")
    print(f"Total forward calls   : {len(forward_calls)}")
    print(f"Total chain length    : {len(chain_array)}")

    # Plot post burn-in chain
    plot_chain_2d(
        chain_array[N_burnin_sub:],
        used_forward=forward_calls[N_burnin_sub:],
        theta_true=theta_true,
        names=("A", "f"),
        title=f"{name} post burn-in",
    )

# ---------------------------------------------------------------------
# Compare forward-call fraction over iterations
# ---------------------------------------------------------------------
plt.figure(figsize=(8, 4))
for name, (_, forward_calls, _, _) in results.items():
    plt.plot(
        np.cumsum(forward_calls) / np.arange(1, len(forward_calls) + 1), label=name
    )
plt.xlabel("Iteration")
plt.ylabel("Cumulative Forward-call fraction")
plt.title("Forward-call fraction over iterations")
plt.grid(True)
plt.legend()
plt.show()
