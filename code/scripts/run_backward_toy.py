# %% [markdown]
# # Backward toy: clean Active-MCMC demo (VSCode live script)
#
# This example performs Bayesian inversion for the toy forward model using:
# - a POD–GP surrogate (low fidelity) and
# - the true toy forward model (high fidelity),
# combined through an Active-MCMC mechanism.
#
# Two strategies are run:
# 1) coarse_only: surrogate-only likelihood
# 2) fixed_subsample: periodic high-fidelity correction (subsampling)
#
# Diagnostics and plots are delegated to `gp_active_mcmc.diagnostics.*`.

# %% Imports
from __future__ import annotations

import copy

import numpy as np
import tinyDA as tda
from scipy.stats import multivariate_normal

from gp_active_mcmc.active_mcmc_model import ActiveMCMCModel
from gp_active_mcmc.diagnostics.mcmc import plot_cumulative_hf_fraction, plot_chain
from gp_active_mcmc.diagnostics.surrogate import plot_prediction_at_theta
from gp_active_mcmc.gp import MultiOutputGP
from gp_active_mcmc.likelihood import GaussianLogLikeWithGP
from gp_active_mcmc.pod import POD
from gp_active_mcmc.podgp import PODGPSurrogate
from gp_active_mcmc.sampler import sample_active_chain, ActiveMCMCChain
from gp_active_mcmc.toy import make_forward_model, make_observation, make_timeline
from gp_active_mcmc.utils.rng import set_seed


# %% [markdown]
# ## Configuration

# %%
SEED = 1
N_TOTAL = (
    2000  # number of LF evaluations (for fixed_subsample: LF steps; HF is periodic)
)
N_BURNIN = 400
N_CHAINS = 1

# surrogate / training
N_INIT = 25
POD_RANK = 10
GP_KERNEL = "matern52"
USE_ARD = True
N_RETRAIN_MAX = 1
UPDATE_EVERY = 100

# observation / active threshold
SIGMA_OBS = 0.01
GAMMA_THRESHOLD = 0.01

# fixed-subsample strategy
SUBSAMPLE_RATE = 5

# toy time grid
N_STEPS = 500
T_END = 0.05

rng = set_seed(SEED)

# %% [markdown]
# ## Prior and synthetic observation

# %%
prior_mean = np.array([0.8, 150.0, 0.01])
prior_cov = np.diag([0.5**2, 10.0**2, 0.001**2])
prior = multivariate_normal(mean=prior_mean, cov=prior_cov)

t = make_timeline(T=N_STEPS, t_end=T_END)
forward_model = make_forward_model(t)

theta_true = prior.rvs(random_state=rng)
y_obs = make_observation(rng, theta_true, t, SIGMA_OBS)
sigma_obs = SIGMA_OBS * np.ones_like(y_obs)

# %% [markdown]
# ## Proposal (tinyDA)

# %%
proposal = tda.AdaptiveMetropolis(
    C0=prior_cov,
    sd=0.1,
    adaptive=True,
    period=100,
    gamma=1.01,
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
    """Create coarse and fine posteriors for the given ActiveMCMCModel."""
    cov = (sigma_obs**2) * np.eye(len(y_obs))

    like_coarse = GaussianLogLikeWithGP(y_obs, cov)
    like_fine = tda.AdaptiveGaussianLogLike(y_obs, cov)

    post_coarse = tda.Posterior(prior, like_coarse, model.coarse)
    post_fine = tda.Posterior(prior, like_fine, model.fine)
    return post_coarse, post_fine


def run_active_mcmc(
    *,
    model: ActiveMCMCModel,
    posterior: tda.Posterior | list[tda.Posterior],
    theta0: np.ndarray,
    chain_key: str,
    n_total: int,
    subsampling_rate: int,
) -> ActiveMCMCChain:
    """
    Run Active-MCMC via the repository sampler wrapper.

    Notes
    -----
    - `n_total` is interpreted as number of LF steps; the sampler internally stores
      chains according to `subsampling_rate`.
    """
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
        "subsampling_rate": 1,
        "chain_key": "chain_0",
    },
    "AL-DAMCMC": {
        "model": ActiveMCMCModel(
            lf_model=copy.deepcopy(emul_base),
            hf_model=forward_model,
            gamma_threshold=GAMMA_THRESHOLD,
        ),
        "subsampling_rate": SUBSAMPLE_RATE,
        "chain_key": "chain_coarse_0",
    },
}

theta0 = prior_mean.copy()

# %% [markdown]
# ## Run strategies and plot diagnostics

# %%
for name, cfg in strategies.items():
    model = cfg["model"]
    subsampling_rate = int(cfg["subsampling_rate"])
    chain_key = cfg["chain_key"]
    post_coarse, post_fine = make_posteriors(model)

    if name == "AL-MCMC":
        posterior = post_coarse
    elif name == "AL-DAMCMC":
        posterior = [post_coarse, post_fine]

    plot_prediction_at_theta(
        model.lf_model,
        theta_true,
        t,
        y_obs,
        title=r"Prediction at $\theta_\mathrm{true}$ before MCMC",
    )

    chain = run_active_mcmc(
        model=model,
        posterior=posterior,
        theta0=theta0,
        n_total=N_TOTAL,
        subsampling_rate=subsampling_rate,
        chain_key=chain_key,
    )

    plot_prediction_at_theta(
        model.lf_model,
        theta_true,
        t,
        y_obs,
        title=r"Prediction at $\theta_\mathrm{true}$ after MCMC, " + f"{name}",
    )

    # HF usage (from sampler output)
    plot_cumulative_hf_fraction(chain.forward_calls, burnin=0)

    # Parameter-space diagnostics (project helper)
    plot_chain(
        chain.samples[:, :2],
        used_hf=chain.forward_calls,
        theta_true=theta_true[:2],
        names=("A", "f"),
        title=f"{name} post burn-in",
    )
