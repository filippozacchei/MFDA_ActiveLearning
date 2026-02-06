# %% [markdown]
# # Backward toy: clean Active-MCMC demo (VSCode live script)
#
# This example performs Bayesian inversion for the toy forward model using:
# - a POD–GP surrogate (low fidelity), and
# - the true toy forward model (high fidelity),
# combined through an Active-MCMC mechanism.
#
# Three strategies are run:
# 1) **AL-MCMC**: surrogate-only likelihood (subsampling_rate = 1)
# 2) **AL-DAMCMC**: fixed periodic HF correction (fixed subsampling)
# 3) **AL-ADAMCMC**: adaptive subchain length with chunked sampling
#
# Diagnostics and plots are delegated to `gp_active_mcmc.diagnostics.*`.
#
# **How to run**
# - In VSCode: open this file and run cells (`# %%`) interactively.
# - From terminal: `python run_backward_toy_notebook.py` (plots will pop up).

# %% Imports
from __future__ import annotations

import copy

import numpy as np
import tinyDA as tda
from scipy.stats import multivariate_normal

from gp_active_mcmc.proposal import AdaptiveMetropolisShared
from gp_active_mcmc.active_mcmc_chain import ActiveMCMCChain
from gp_active_mcmc.active_mcmc_model import ActiveMCMCModel, AdaptiveActiveMCMCModel
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
from gp_active_mcmc.toy import make_forward_model, make_observation, make_timeline
from gp_active_mcmc.utils.rng import set_seed
from gp_active_mcmc.adaptive_config import AdaptiveControl, AdaptiveState


# %% [markdown]
# ## Configuration

# %%
SEED = 1
N_TOTAL = 5000  # interpreted as coarse evaluation budget for all three strategies
N_BURNIN = 2000
N_CHAINS = 1

# surrogate / training
N_INIT = 50
POD_RANK = 20
GP_KERNEL = "matern52"
USE_ARD = True
N_RETRAIN_MAX = 0
UPDATE_EVERY = 100

# observation / active threshold
SIGMA_OBS = 0.1
GAMMA_THRESHOLD = 0.1

# fixed-subsample strategy
SUBSAMPLE_RATE = 5

# adaptive strategy (chunking)
CHUNK_SIZE = 200  # in coarse eval units

# toy time grid
N_STEPS = 500
T_END = 0.05

rng = set_seed(SEED)


# %% [markdown]
# ## Prior and synthetic observation

# %%
prior_mean = np.array([0.8, 150.0, 0.01])
prior_cov = np.diag([0.5**2, 40.0**2, 0.01**2])
prior = multivariate_normal(mean=prior_mean, cov=prior_cov)

t = make_timeline(T=N_STEPS, t_end=T_END)
forward_model = make_forward_model(t)

theta_true = prior.rvs(random_state=rng)
y_obs = make_observation(rng, theta_true, t, SIGMA_OBS)
sigma_obs = SIGMA_OBS * np.ones_like(y_obs)


# %% [markdown]
# ## Proposal (tinyDA)
#
# tinyDA deepcopies the proposal internally. If you are chunking (re-entering tinyDA
# repeatedly), you may want to share the adaptive state across deepcopies.
#
# - `proposal` is the vanilla tinyDA `AdaptiveMetropolis`.
# - `adaptive_proposal` is a wrapper that shares adaptive state across deepcopies.

# %%
proposal = tda.AdaptiveMetropolis(
    C0=prior_cov,
    sd=0.1,
    adaptive=True,
    period=100,
    gamma=1.01,
    t0=0,
)

adaptive_proposal = AdaptiveMetropolisShared(
    C0=prior_cov,
    sd=0.1,
    adaptive=True,
    period=100,
    gamma=1.01,
    t0=0,
)


# %% [markdown]
# ## Build initial POD–GP surrogate

# %%
x_train = np.array([prior.rvs(random_state=rng) for _ in range(N_INIT)])
y_train = np.array([forward_model(theta) for theta in x_train])

pod = POD(r=POD_RANK).fit(y_train)
a_train = pod.project(y_train)[:, :POD_RANK]

gp = MultiOutputGP(
    x_train,
    a_train,
    kernel=GP_KERNEL,
    ard=USE_ARD,
    n_retrain_max=N_RETRAIN_MAX,
    update_every=UPDATE_EVERY,
)

emul_base = PODGPSurrogate(pod=pod, gp=gp)


# %% [markdown]
# ## Helpers (kept minimal and local)


# %%
def make_posteriors(model: ActiveMCMCModel) -> tuple[tda.Posterior, tda.Posterior]:
    """Create coarse and fine posteriors for a given ActiveMCMCModel."""
    cov = (sigma_obs**2) * np.eye(len(y_obs))

    like_coarse = GaussianLogLikeWithGP(y_obs, cov)
    like_fine = tda.AdaptiveGaussianLogLike(y_obs, cov)

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
    """Run a fixed-subsampling strategy using the repository wrapper."""
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
    """Run the adaptive-subchain strategy using chunked sampling."""
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
    "AL-DAMCMC": {
        "model": ActiveMCMCModel(
            lf_model=copy.deepcopy(emul_base),
            hf_model=forward_model,
            gamma_threshold=GAMMA_THRESHOLD,
        ),
        "runner": "fixed",
        "subsampling_rate": SUBSAMPLE_RATE,
        "chain_key": "chain_coarse_0",
        "posterior_kind": "both",
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
        t,
        y_obs,
        title=r"Prediction at $\theta_\mathrm{true}$ before MCMC",
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
        t,
        y_obs,
        title=r"Prediction at $\theta_\mathrm{true}$ after MCMC, " + f"{name}",
    )

    plot_cumulative_hf_fraction(chain.forward_calls, burnin=0)
    plot_chain(
        chain.samples[:, :2],
        used_hf=chain.forward_calls,
        theta_true=theta_true[:2],
        names=("A", "f"),
        title=f"{name} post burn-in",
    )

    chain_burnin = chain.burnin(N_BURNIN)
    plot_chain(
        chain_burnin.samples[:, :2],
        used_hf=chain_burnin.forward_calls,
        theta_true=theta_true[:2],
        names=("A", "f"),
        title=f"{name} post burn-in",
    )

plot_subchain_length_history(
    strategies["AL-ADAMCMC"]["model"].adaptive_state.subchain_history
)
