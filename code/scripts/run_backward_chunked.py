# %% Imports
import copy
import numpy as np
import matplotlib.pyplot as plt
import tinyDA as tda

from scipy.stats import multivariate_normal

from gp_active_mcmc.active_mcmc_model import (
    AdaptiveActiveMCMCModel,
)
from gp_active_mcmc.gp import GPSurrogate
from gp_active_mcmc.likelihood import GaussianLogLikeWithGP
from gp_active_mcmc.pod import POD
from gp_active_mcmc.podgp import PODGPSurrogate
from gp_active_mcmc.toy import make_observation, make_timeline, toy_forward
from gp_active_mcmc.diagnostics import plot_active_mcmc_diagnostics

from utils import plot_chain_2d


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
SEED = 1
N_TOTAL = 5000  # total coarse iterations
N_BURNIN = 200
CHUNK_SIZE = 20  # coarse iterations per chunk

N_INIT = 50
POD_RANK = 20
GP_KERNEL = "matern52"
USE_ARD = True
N_RETRAIN_MAX = 50

SIGMA_OBS = 0.01
GAMMA_VAR = 0.01

INITIAL_SUBCHAIN = 5

rng = np.random.default_rng(SEED)


# ---------------------------------------------------------------------
# Timeline and forward model
# ---------------------------------------------------------------------
t = make_timeline(T=500, t_end=0.05)


def forward_model(theta: np.ndarray) -> np.ndarray:
    return toy_forward(theta, t)


# ---------------------------------------------------------------------
# Prior and proposal
# ---------------------------------------------------------------------
prior_mean = np.array([0.8, 150.0, 0.01])
prior_cov = np.diag([0.5**2, 40.0**2, 0.01**2])

prior = multivariate_normal(mean=prior_mean, cov=prior_cov)

proposal = tda.AdaptiveMetropolis(
    C0=prior_cov,
    sd=0.1,
    adaptive=True,
    period=100,
    gamma=1.01,
)


# ---------------------------------------------------------------------
# Ground truth and observations
# ---------------------------------------------------------------------
theta_true = prior.rvs(random_state=rng)
y_obs = make_observation(rng, theta_true, t, SIGMA_OBS)
sigma_obs = SIGMA_OBS * np.ones_like(y_obs)


# ---------------------------------------------------------------------
# Initial training set for POD + GP
# ---------------------------------------------------------------------
X0 = np.array([prior.rvs(random_state=rng) for _ in range(N_INIT)])
Y0 = np.array([forward_model(theta) for theta in X0])

pod = POD(r=POD_RANK).fit(Y0)
A0 = pod.project(Y0)

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

emulator = PODGPSurrogate(pod=pod, gps=gps)


# ---------------------------------------------------------------------
# Active MCMC model (adaptive subsampling)
# ---------------------------------------------------------------------
model = AdaptiveActiveMCMCModel(
    lf=copy.deepcopy(emulator),
    hf=forward_model,
    gamma_var=GAMMA_VAR,
    initial_subchain=INITIAL_SUBCHAIN,
    update_every=1,
    target_error=0.01,
    max_steps=N_TOTAL,
)


# ---------------------------------------------------------------------
# Likelihoods and posteriors
# ---------------------------------------------------------------------
likelihood_coarse = GaussianLogLikeWithGP(y_obs, sigma_obs * np.eye(len(y_obs)))
likelihood_fine = tda.AdaptiveGaussianLogLike(y_obs, sigma_obs * np.eye(len(y_obs)))

posterior_coarse = tda.Posterior(prior, likelihood_coarse, model.coarse)
posterior_fine = tda.Posterior(prior, likelihood_fine, model.fine)


# ---------------------------------------------------------------------
# Chunked MCMC runner
# ---------------------------------------------------------------------
def run_chunked_mcmc(
    model,
    posteriors,
    proposal,
    theta0,
    n_total,
    burnin,
    chunk_size,
    chain_name="chain_coarse_0",
):
    """
    Run MCMC in chunks to allow adaptive subsampling without
    modifying tinyDA.
    """

    theta_current = theta0.copy()
    n_done = 0

    chains = []
    hf_calls = []

    while n_done < n_total:
        subchain = model.subchain_length
        n_chunk = min(chunk_size, n_total - n_done)
        n_chunk_sub = max(1, n_chunk // subchain)

        samples = tda.sample(
            posteriors=posteriors,
            proposal=proposal,
            iterations=n_chunk_sub,
            n_chains=1,
            force_sequential=True,
            initial_parameters=theta_current,
            store_coarse_chain=True,
            subsampling_rate=subchain,
            adaptive_error_model=None,
        )

        chain = samples[chain_name]
        theta_block = np.array([link.parameters for link in chain])
        hf_block = np.array(model.used_hf[-len(theta_block) :])

        chains.append(theta_block)
        hf_calls.append(hf_block)

        theta_current = theta_block[-1]
        n_done += subchain

    chain = np.vstack(chains)
    used_hf = np.concatenate(hf_calls)

    return chain, used_hf


# ---------------------------------------------------------------------
# Run adaptive chain
# ---------------------------------------------------------------------
theta0 = prior_mean.copy()

chain, used_hf = run_chunked_mcmc(
    model=model,
    posteriors=[posterior_coarse, posterior_fine],
    proposal=proposal,
    theta0=theta0,
    n_total=N_TOTAL,
    burnin=N_BURNIN,
    chunk_size=CHUNK_SIZE,
)


# ---------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------
accepted = np.any(np.diff(chain, axis=0) != 0, axis=1)
accept_rate = accepted.mean()
forward_frac = used_hf.mean()

rmse = np.mean(np.sqrt(np.sum((chain[N_BURNIN:] - theta_true) ** 2, axis=1)))

print("\nAdaptive subsampling summary")
print(f"Acceptance rate       : {accept_rate:.3f}")
print(f"Forward-call fraction : {forward_frac:.3f}")
print(f"RMSE vs theta_true    : {rmse:.5f}")
print(f"Final subchain length : {model.subchain_length}")


# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------
plot_active_mcmc_diagnostics(model, N_BURNIN)

plot_chain_2d(
    chain[N_BURNIN:],
    used_forward=used_hf[N_BURNIN:],
    theta_true=theta_true,
    names=("A", "f"),
    title="Adaptive subsampling (post burn-in)",
)

plt.figure(figsize=(8, 4))
plt.plot(np.cumsum(used_hf) / np.arange(1, len(used_hf) + 1))
plt.xlabel("Iteration")
plt.ylabel("Cumulative HF fraction")
plt.title("Forward-call fraction over iterations")
plt.grid(True)
plt.show()
