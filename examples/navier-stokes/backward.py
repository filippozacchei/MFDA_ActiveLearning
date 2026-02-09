# %% [markdown]
# # Backward Navier–Stokes (HF) inverse problem: clean Active-MCMC demo (VSCode live script)
#
# Bayesian inversion for the Navier–Stokes outlet profile QoI using:
# - **LF surrogate**: POD–GP trained on HF snapshots (cheap emulator)
# - **HF model**: Navier–Stokes forward solver
# combined through Active-MCMC.
#
# Two strategies are run:
# 1) **AL-MCMC**: surrogate-only likelihood (subsampling_rate = 1)
# 2) **AL-ADAMCMC**: adaptive subchain length with chunked sampling (recommended)
#
# Notes
# -----
# - Parameters: theta = [h1, U_in]
# - Observation: outlet profile u_x(y) sampled on T=100 points
# - Likelihood: Gaussian with diagonal covariance sigma_obs^2 I
#
# How to run
# - VSCode: run cells (`# %%`)
# - Terminal: `python run_backward_ns_hf.py`

# %% Imports
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import tinyDA as tda
from scipy.stats import multivariate_normal

from gp_active_mcmc.proposal import AdaptiveMetropolisShared
from gp_active_mcmc.active_mcmc_chain import ActiveMCMCChain
from gp_active_mcmc.active_mcmc_model import ActiveMCMCModel, AdaptiveActiveMCMCModel
from gp_active_mcmc.adaptive_config import AdaptiveControl, AdaptiveState
from gp_active_mcmc.diagnostics.mcmc import (
    plot_chain,
    plot_cumulative_hf_fraction,
    plot_subchain_length_history,
)
from gp_active_mcmc.diagnostics.surrogate import plot_prediction_at_theta
from gp_active_mcmc.gp import MultiOutputGP
from gp_active_mcmc.likelihood import GaussianLogLikeWithGP
from gp_active_mcmc.pod import POD
from gp_active_mcmc.podgp import PODGPSurrogate
from gp_active_mcmc.sampler import sample_active_chain, sample_adaptive_active_chain
from gp_active_mcmc.utils.rng import set_seed
from utils.outlet import resample_profile

import matplotlib.pyplot as plt
from utils.mf_ipcs import forward_model as hf_solver

# %% [markdown]
# ## Configuration

# %%
SEED = 123
rng = set_seed(SEED)

# QoI
T = 100  # outlet profile length

# MCMC budget (coarse eval units)
N_TOTAL = 2000
N_BURNIN = 800
N_CHAINS = 1

# surrogate / training
N_INIT = 25  # HF snapshots used to build the initial emulator
POD_RANK = 5
GP_KERNEL = "matern52"
USE_ARD = True
N_RETRAIN_MAX = 0
UPDATE_EVERY = 100

# observation / active threshold
SIGMA_OBS = 0.1
GAMMA_THRESHOLD = 0.1

# adaptive strategy (chunking)
SUBSAMPLE_RATE = 5
CHUNK_SIZE = 200  # in coarse eval units

# prior box (sampling support)
H1_MIN, H1_MAX = 0.05, 0.15
L_MIN, L_MAX = 0.3, 0.5
U_MIN, U_MAX = 0.25, 1.00

# %% [markdown]
# ## HF forward model wrapper (outlet profile, resampled to T points)
#
# Assumption: `hf_solver(h1, U_in) -> (y, u)`.

# %%


def make_forward_model(*, T: int):
    def f(theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, dtype=float).ravel()
        y, u = hf_solver(float(theta[0]), U_in=float(theta[1]), L_down=float(theta[2]))
        return resample_profile(y, u, T=T)

    return f


forward_model = make_forward_model(T=T)

# %% [markdown]
# ## Prior and synthetic observation
#

# %%
prior_mean = np.array([0.10, 1.0, 0.4])
prior_cov = np.diag(
    [
        (0.25 * (H1_MAX - H1_MIN)) ** 2,
        (0.25 * (U_MAX - U_MIN)) ** 2,
        (0.25 * (L_MAX - L_MIN)) ** 2,
    ]
)
prior = multivariate_normal(mean=prior_mean, cov=prior_cov)

theta_true = prior.rvs(random_state=rng)

y_clean = forward_model(theta_true)
y_obs = y_clean + SIGMA_OBS * rng.standard_normal(size=T)
sigma_obs = SIGMA_OBS * np.ones_like(y_obs)

# x-axis for plotting in diagnostics
t_plot = np.arange(T)

# %% [markdown]
# ## Proposal (tinyDA)

# %%
proposal = tda.AdaptiveMetropolis(
    C0=0.1 * prior_cov,
    sd=1.0,
    adaptive=True,
    period=100,
    gamma=1.01,
    t0=0,
)

adaptive_proposal = AdaptiveMetropolisShared(
    C0=0.1 * prior_cov,
    sd=1.0,
    adaptive=True,
    period=100,
    gamma=1.01,
    t0=0,
)

# %% [markdown]
# ## Build initial POD–GP surrogate (trained on HF snapshots)

# %%
x_train = np.array([prior.rvs(random_state=rng) for _ in range(N_INIT)])
y_train = np.array([forward_model(theta) for theta in x_train])  # (N_INIT, T)

pod = POD(r=POD_RANK).fit(y_train)
a_train = pod.project(y_train)[:, :POD_RANK]  # (N_INIT, r)

gp = MultiOutputGP(
    X_train=x_train,
    Y_train=a_train,
    kernel=GP_KERNEL,
    ard=USE_ARD,
    n_retrain_max=N_RETRAIN_MAX,
    update_every=UPDATE_EVERY,
)

emul_base = PODGPSurrogate(pod=pod, gp=gp)

# %% [markdown]
# ## Helpers


# %%
def make_posteriors(model: ActiveMCMCModel) -> tuple[tda.Posterior, tda.Posterior]:
    cov = (sigma_obs**2) * np.eye(len(y_obs))
    like_coarse = GaussianLogLikeWithGP(y_obs, cov)  # uses GP variance
    like_fine = tda.AdaptiveGaussianLogLike(y_obs, cov)  # standard Gaussian LL
    post_coarse = tda.Posterior(prior, like_coarse, model.coarse)
    post_fine = tda.Posterior(prior, like_fine, model.fine)
    return post_coarse, post_fine


def run_active_mcmc_fixed(
    *,
    model: ActiveMCMCModel,
    posterior: tda.Posterior | list[tda.Posterior],
    theta0: np.ndarray,
    chain_key: str,
    n_total: int,
    subsampling_rate: int,
) -> ActiveMCMCChain:
    n_samples = n_total // subsampling_rate
    return sample_active_chain(
        model=model,
        posterior=posterior,
        proposal=copy.deepcopy(proposal),
        n_samples=n_samples,
        n_chains=N_CHAINS,
        initial_parameter=theta0,
        subsampling_rate=subsampling_rate,
        chain_key=chain_key,
        force_sequential=True,
        store_coarse_chain=True,
        theta_true=theta_true,
    )


def run_active_mcmc_adaptive(
    *,
    model: AdaptiveActiveMCMCModel,
    posterior: tda.Posterior | list[tda.Posterior],
    theta0: np.ndarray,
    chain_key: str,
    chunk_size: int,
    n_total: int,
) -> ActiveMCMCChain:
    return sample_adaptive_active_chain(
        model=model,
        posterior=posterior,
        proposal=adaptive_proposal,
        n_coarse_evals=n_total,
        initial_parameter=theta0,
        chain_key=chain_key,
        chunk_size=chunk_size,
        n_chains=N_CHAINS,
        force_sequential=True,
        store_coarse_chain=True,
        theta_true=theta_true,
    )


# %% [markdown]
# ## Define strategies

# %%
strategies: dict[str, dict[str, object]] = {
    "AL-MCMC": {
        "model": ActiveMCMCModel(
            lf_model=copy.deepcopy(emul_base),
            hf_model=forward_model,
            gamma_threshold=GAMMA_THRESHOLD,
        ),
        "runner": "fixed",
        "subsampling_rate": 1,
        "chain_key": "chain_0",
        "posterior_kind": "coarse",
    },
    "AL-ADAMCMC": {
        "model": AdaptiveActiveMCMCModel(
            lf_model=copy.deepcopy(emul_base),
            hf_model=forward_model,
            gamma_threshold=GAMMA_THRESHOLD,
            adaptive_control=AdaptiveControl(),
            initial_adaptive_state=AdaptiveState(
                subchain_length=SUBSAMPLE_RATE,
                subsample_rate=1 / SUBSAMPLE_RATE,
            ),
        ),
        "runner": "adaptive",
        "chain_key": "chain_coarse_0",
        "posterior_kind": "both",
    },
}

theta0 = prior_mean.copy()

# %% [markdown]
# ## Run strategies and plot diagnostics

# %%
for name, cfg in strategies.items():
    model = cfg["model"]
    chain_key = str(cfg["chain_key"])

    post_coarse, post_fine = make_posteriors(model)

    posterior_kind = str(cfg["posterior_kind"])
    if posterior_kind == "coarse":
        posterior = post_coarse
    elif posterior_kind == "both":
        posterior = [post_coarse, post_fine]
    else:
        raise ValueError(f"Unknown posterior_kind: {posterior_kind}")

    plot_prediction_at_theta(
        model.lf_model,
        theta_true,
        t_plot,
        y_obs,
        title=r"Prediction at $\theta_\mathrm{true}$ before MCMC",
        y_true=y_clean,
    )

    runner = str(cfg["runner"])
    if runner == "fixed":
        subsampling_rate = int(cfg["subsampling_rate"])
        chain = run_active_mcmc_fixed(
            model=model,
            posterior=posterior,
            theta0=theta0,
            chain_key=chain_key,
            n_total=N_TOTAL,
            subsampling_rate=subsampling_rate,
        )
    elif runner == "adaptive":
        chain = run_active_mcmc_adaptive(
            model=model,
            posterior=posterior,
            theta0=theta0,
            chain_key=chain_key,
            chunk_size=CHUNK_SIZE,
            n_total=N_TOTAL,
        )
    else:
        raise ValueError(f"Unknown runner: {runner}")

    plot_prediction_at_theta(
        model.lf_model,
        theta_true,
        t_plot,
        y_obs,
        title=r"Prediction at $\theta_\mathrm{true}$ after MCMC, " + f"{name}",
        y_true=y_clean,
    )

    plot_cumulative_hf_fraction(chain.forward_calls, burnin=0)

    # parameter trace (2D) for h1 and U_in
    plot_chain(
        chain.samples[:, :2],
        used_hf=chain.forward_calls,
        theta_true=theta_true[:2],
        names=("h1", "U_in"),
        title=f"{name} post burn-in",
    )

    chain_burnin = chain.burnin(N_BURNIN)
    plot_chain(
        chain_burnin.samples[:, :2],
        used_hf=chain_burnin.forward_calls,
        theta_true=theta_true[:2],
        names=("h1", "U_in"),
        title=f"{name} post burn-in",
    )

# adaptive subchain diagnostics (if present)
if "AL-ADAMCMC" in strategies:
    plot_subchain_length_history(
        strategies["AL-ADAMCMC"]["model"].adaptive_state.subchain_history
    )
