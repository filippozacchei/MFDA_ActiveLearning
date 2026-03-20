# %% [markdown]
# # Backward beam: Bayesian inversion with Active-MCMC
#
# Self-contained implementation following the same logic as `run_backward_toy.py`
# but applied to the cantilever beam inverse problem.
#
# **No dependency on the library's inference machinery** — all MCMC / likelihood /
# proposal / adaptive-subchain code is implemented locally.  The only library
# import is `MultiOutputGP` for the GP surrogate (already used & tested in
# `forward.py`).
#
# Two inference modes are demonstrated:
#
# 1. **MCMC-guided active learning (single posterior)**
#    The sampler uses the *coarse* model which triggers HF when surrogate
#    uncertainty exceeds a threshold.
#
# 2. **DA-MCMC guided active learning with adaptive subchain (recommended)**
#    A coarse subchain runs for an adaptive number of steps, then a fine (HF)
#    correction step updates the surrogate and applies a delayed-acceptance ratio.
#    The subchain length adapts based on LF-HF discrepancy.

# %% Imports
from __future__ import annotations

import copy

import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

from beam import make_spatial_grid, make_forward_model, make_observation
from gp_active_mcmc.surrogates import MultiOutputGP  # only library dependency


# =====================================================================
#  Self-contained inference components
# =====================================================================


class DirectGPSurrogate:
    """GP surrogate mapping theta -> observed beam displacements directly.

    Satisfies the ActiveSurrogate protocol (predict / update).
    No POD compression: observation space is already low-dimensional.
    """

    def __init__(self, gp: MultiOutputGP):
        self.gp = gp

    def predict(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (mean, variance) each of shape (n_obs,)."""
        theta_2d = np.asarray(theta, dtype=float).reshape(1, -1)
        y_mean, y_var = self.gp.predict(theta_2d)
        return y_mean[0], y_var[0]

    def update(self, theta: np.ndarray, y_hf: np.ndarray) -> None:
        """Add one new HF observation to the GP training set."""
        theta_1d = np.asarray(theta, dtype=float).ravel()
        y_1d = np.asarray(y_hf, dtype=float).ravel()
        self.gp.update(theta_1d, y_1d)


class ActiveModel:
    """Couples a low-fidelity GP surrogate with a high-fidelity forward model.

    Exposes two evaluation paths mirroring ActiveMCMCModel from the library:

    * ``coarse(theta)`` — LF-first; triggers HF when ``mean(var) > gamma²``.
    * ``fine(theta)``   — always HF; records LF-HF error and updates surrogate.
    """

    def __init__(
        self,
        lf_model: DirectGPSurrogate,
        hf_model,
        gamma_threshold: float,
    ):
        self.lf_model = lf_model
        self.hf_model = hf_model
        self.gamma_threshold = gamma_threshold
        self.used_hf: list[bool] = []
        self.hf_errors: list[float] = []

    # -- coarse path ------------------------------------------------
    def coarse(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """LF-first evaluation with uncertainty trigger.

        Returns
        -------
        y_mean : ndarray (n_obs,)
            Predictive mean (LF) or HF output.
        y_var : ndarray or None
            Predictive variance if LF was used, ``None`` if HF was triggered.
        """
        theta = np.asarray(theta, dtype=float).ravel()
        y_mean, y_var = self.lf_model.predict(theta)
        avg_var = float(np.mean(y_var))

        if avg_var > self.gamma_threshold**2:
            y_hf = np.asarray(self.hf_model(theta), dtype=float).ravel()
            self.lf_model.update(theta, y_hf)
            self.used_hf.append(True)
            return y_hf, None

        self.used_hf.append(False)
        return y_mean, y_var

    # -- fine path --------------------------------------------------
    def fine(self, theta: np.ndarray) -> np.ndarray:
        """Always-HF evaluation; records LF-HF RMSE and updates surrogate."""
        theta = np.asarray(theta, dtype=float).ravel()
        y_hf = np.asarray(self.hf_model(theta), dtype=float).ravel()

        # LF-HF discrepancy *before* updating surrogate
        y_lf, _ = self.lf_model.predict(theta)
        rmse = float(np.sqrt(np.mean((y_lf - y_hf) ** 2)))
        self.hf_errors.append(rmse)

        # Update surrogate with new HF point
        self.lf_model.update(theta, y_hf)

        # Replace last used_hf entry (fine correction at same MCMC step)
        if self.used_hf:
            self.used_hf[-1] = True
        else:
            self.used_hf.append(True)

        return y_hf


# -- Gaussian log-likelihood ----------------------------------------

def gaussian_loglike(
    y_pred: np.ndarray,
    y_obs: np.ndarray,
    cov_obs: np.ndarray,
    pred_variance: np.ndarray | None = None,
) -> float:
    r"""Gaussian log-likelihood with optional surrogate-variance inflation.

    .. math::

        \log p(y_{\mathrm{obs}} \mid \theta) =
            -\tfrac12 r^T C^{-1} r - \tfrac12 \log|C| - \tfrac{n}{2}\log(2\pi)

    where :math:`C = C_{\mathrm{obs}} + \mathrm{diag}(v_{\mathrm{pred}})` when
    ``pred_variance`` is not ``None``.
    """
    resid = y_obs - y_pred
    C = cov_obs.copy()
    if pred_variance is not None:
        C = C + np.diag(pred_variance)

    try:
        L = np.linalg.cholesky(C)
    except np.linalg.LinAlgError:
        return -np.inf

    alpha = np.linalg.solve(L, resid)
    n = len(y_obs)
    ll = -0.5 * np.dot(alpha, alpha)
    ll -= np.sum(np.log(np.diag(L)))
    ll -= 0.5 * n * np.log(2.0 * np.pi)
    return float(ll)


# -- Adaptive Metropolis proposal ------------------------------------

class AdaptiveProposal:
    """Adaptive Metropolis proposal (Haario et al., 2001).

    After ``2 * period`` samples the empirical covariance of the chain is
    used, scaled by ``sd² × 2.38² / d`` (optimal scaling for Gaussians).
    """

    def __init__(self, C0: np.ndarray, period: int = 100, sd: float = 1.0):
        self.C0 = np.array(C0, dtype=float)
        self.C = self.C0.copy()
        self.period = period
        self.sd = sd
        self.d = C0.shape[0]
        self._eps = 1e-8 * np.eye(self.d)
        self._scale = (self.sd**2) * (2.38**2) / self.d

    def get_cov(self) -> np.ndarray:
        return self.C

    def adapt(self, samples: np.ndarray, step: int) -> None:
        """Periodically replace the proposal covariance with the empirical one."""
        if step < 2 * self.period or step < self.d + 1:
            return
        if step % self.period != 0:
            return
        emp_cov = np.cov(samples[: step + 1].T)
        self.C = self._scale * emp_cov + self._eps


# -- Adaptive subchain policy ----------------------------------------

class AdaptiveSubchainPolicy:
    """Adjusts subchain length between fine corrections.

    If the latest LF-HF RMSE is *above* ``target_error`` → shrink (more HF).
    If *below* → grow (less HF).  Updates happen every ``update_every`` HF calls.
    """

    def __init__(
        self,
        subchain_length: int = 20,
        update_every: int = 5,
        target_error: float = 0.05,
        min_subchain: int = 1,
        max_subchain: int = 500,
        grow_factor: float = 2.0,
        shrink_factor: float = 0.5,
    ):
        self.subchain_length = subchain_length
        self.update_every = update_every
        self.target_error = target_error
        self.min_subchain = min_subchain
        self.max_subchain = max_subchain
        self.grow_factor = grow_factor
        self.shrink_factor = shrink_factor

        self.subchain_history: list[int] = []
        self._hf_since_update = 0

    def record_coarse(self) -> None:
        """Record the current subchain length (one call per coarse eval)."""
        self.subchain_history.append(self.subchain_length)

    def on_fine(self, hf_errors: list[float]) -> None:
        """Called after each fine evaluation; may trigger a length update."""
        self._hf_since_update += 1
        if self._hf_since_update >= self.update_every and len(hf_errors) > 0:
            err = hf_errors[-1]
            if err > self.target_error:
                new = int(np.floor(self.subchain_length * self.shrink_factor))
            else:
                new = int(np.ceil(self.subchain_length * self.grow_factor))
            self.subchain_length = max(
                self.min_subchain, min(self.max_subchain, new)
            )
            self._hf_since_update = 0


# =================================================================
#  Sampler 1 — MCMC-guided active learning (single posterior)
# =================================================================

def sample_active_mcmc(
    model: ActiveModel,
    y_obs: np.ndarray,
    cov_obs: np.ndarray,
    prior,
    theta0: np.ndarray,
    n_iter: int,
    proposal: AdaptiveProposal,
    rng: np.random.Generator,
) -> dict:
    """Standard Metropolis-Hastings with ``model.coarse`` as forward model.

    HF calls happen *inside* ``model.coarse`` when the uncertainty trigger fires.
    """
    n_dim = len(theta0)
    samples = np.zeros((n_iter, n_dim))
    accepted = np.zeros(n_iter, dtype=bool)

    theta = theta0.copy()

    # Evaluate initial state via LF (no model.coarse to avoid logging)
    y_init, v_init = model.lf_model.predict(theta)
    ll_cur = gaussian_loglike(y_init, y_obs, cov_obs, v_init)
    lp_cur = prior.logpdf(theta)

    for i in range(n_iter):
        theta_star = rng.multivariate_normal(theta, proposal.get_cov())

        y_star, yvar_star = model.coarse(theta_star)
        ll_star = gaussian_loglike(y_star, y_obs, cov_obs, yvar_star)
        lp_star = prior.logpdf(theta_star)

        log_alpha = (lp_star + ll_star) - (lp_cur + ll_cur)

        if np.log(rng.uniform()) < log_alpha:
            theta = theta_star.copy()
            ll_cur = ll_star
            lp_cur = lp_star
            accepted[i] = True

        samples[i] = theta.copy()
        proposal.adapt(samples, i)

    used_hf = np.array(model.used_hf, dtype=bool)
    return {"samples": samples, "used_hf": used_hf, "accepted": accepted}


# =================================================================
#  Sampler 2 — DA-MCMC with adaptive subchain
# =================================================================

def sample_da_active_mcmc(
    model: ActiveModel,
    y_obs: np.ndarray,
    cov_obs: np.ndarray,
    prior,
    theta0: np.ndarray,
    n_coarse_evals: int,
    proposal: AdaptiveProposal,
    policy: AdaptiveSubchainPolicy,
    rng: np.random.Generator,
) -> dict:
    """Delayed-Acceptance MCMC with adaptive subchain length.

    1. Run a coarse MH sub-chain of length ``policy.subchain_length``.
    2. One fine (HF) correction via the DA ratio.
    3. Adapt subchain length from LF-HF discrepancy.
    4. Repeat until budget exhausted.
    """
    n_dim = len(theta0)
    all_samples: list[np.ndarray] = []
    all_accepted: list[bool] = []

    theta = theta0.copy()

    # Initial coarse evaluation (no logging — used to seed likelihoods)
    y_c_init, v_c_init = model.lf_model.predict(theta)
    ll_c_cur = gaussian_loglike(y_c_init, y_obs, cov_obs, v_c_init)
    lp_cur = prior.logpdf(theta)

    # Initial fine evaluation (needed for the DA ratio)
    used_hf_cursor = len(model.used_hf)  # bookmark before initial fine
    y_f = model.fine(theta)
    ll_f_cur = gaussian_loglike(y_f, y_obs, cov_obs)
    ll_c_at_fine = ll_c_cur  # coarse log-like at the fine chain's state

    coarse_step = 0

    while coarse_step < n_coarse_evals:
        S = policy.subchain_length
        chunk = min(S, n_coarse_evals - coarse_step)

        # ---------- coarse sub-chain ----------
        for _ in range(chunk):
            policy.record_coarse()

            theta_star = rng.multivariate_normal(theta, proposal.get_cov())
            y_c_star, yvar_star = model.coarse(theta_star)
            ll_c_star = gaussian_loglike(y_c_star, y_obs, cov_obs, yvar_star)
            lp_star = prior.logpdf(theta_star)

            log_alpha = (lp_star + ll_c_star) - (lp_cur + ll_c_cur)

            acc = False
            if np.log(rng.uniform()) < log_alpha:
                theta = theta_star.copy()
                ll_c_cur = ll_c_star
                lp_cur = lp_star
                acc = True

            all_samples.append(theta.copy())
            all_accepted.append(acc)
            coarse_step += 1

            proposal.adapt(np.array(all_samples), len(all_samples) - 1)

        # ---------- fine correction (delayed acceptance) ----------
        if coarse_step < n_coarse_evals:
            y_f_new = model.fine(theta)
            ll_f_new = gaussian_loglike(y_f_new, y_obs, cov_obs)

            # DA ratio: compare fine/coarse balance at new vs old state
            log_alpha_da = (ll_f_new - ll_c_cur) - (ll_f_cur - ll_c_at_fine)

            if np.log(rng.uniform()) < log_alpha_da:
                ll_f_cur = ll_f_new
                ll_c_at_fine = ll_c_cur

            policy.on_fine(model.hf_errors)

    # Align used_hf with samples (skip the initial fine entry)
    raw_hf = np.array(model.used_hf, dtype=bool)
    used_hf = raw_hf[used_hf_cursor:]
    n_samples = len(all_samples)
    if len(used_hf) > n_samples:
        used_hf = used_hf[:n_samples]
    elif len(used_hf) < n_samples:
        used_hf = np.concatenate(
            [used_hf, np.zeros(n_samples - len(used_hf), dtype=bool)]
        )

    subchain_hist = np.array(policy.subchain_history, dtype=int)
    return {
        "samples": np.array(all_samples),
        "used_hf": used_hf,
        "accepted": np.array(all_accepted, dtype=bool),
        "subchain_history": subchain_hist,
    }


# =====================================================================
#  Diagnostic helpers
# =====================================================================


def print_summary(
    samples: np.ndarray,
    used_hf: np.ndarray,
    accepted: np.ndarray,
    theta_true: np.ndarray | None = None,
    burn_in: int = 0,
) -> None:
    n = samples.shape[0]
    post_burn = samples[burn_in:]
    n_hf = min(len(used_hf), n)

    print(f"  Chain length     : {n}")
    print(f"  Burn-in          : {burn_in}")
    print(f"  Acceptance rate  : {np.mean(accepted):.3f}")
    print(f"  HF fraction      : {np.mean(used_hf[:n_hf]):.3f}")
    print(f"  Posterior mean   : {np.mean(post_burn, axis=0)}")
    print(f"  Posterior std    : {np.std(post_burn, axis=0)}")
    if theta_true is not None:
        post_mean = np.mean(post_burn, axis=0)
        rmse = float(np.sqrt(np.mean((post_mean - theta_true) ** 2)))
        print(f"  theta_true       : {theta_true}")
        print(f"  Posterior RMSE   : {rmse:.6f}")


def plot_chain_2d(
    samples, used_hf, theta_true=None, labels=("m1", "m2"), title=""
):
    fig, ax = plt.subplots(figsize=(7, 5))
    n = min(len(samples), len(used_hf))
    lf_mask = ~used_hf[:n]
    hf_mask = used_hf[:n]

    ax.scatter(
        samples[:n][lf_mask, 0],
        samples[:n][lf_mask, 1],
        s=4, alpha=0.3, label="LF", color="steelblue",
    )
    ax.scatter(
        samples[:n][hf_mask, 0],
        samples[:n][hf_mask, 1],
        s=10, alpha=0.6, label="HF", color="crimson", marker="x",
    )
    if theta_true is not None:
        ax.scatter(
            *theta_true[:2], s=100, marker="*",
            color="gold", edgecolors="black", zorder=5, label="true",
        )
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    return fig, ax


def plot_cumulative_hf_fraction(used_hf, title="Cumulative HF fraction"):
    n = len(used_hf)
    cumsum = np.cumsum(used_hf.astype(float))
    frac = cumsum / np.arange(1, n + 1)

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(frac, color="steelblue")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cumulative HF fraction")
    ax.set_title(title)
    ax.set_ylim(-0.02, 1.02)
    plt.tight_layout()
    return fig, ax


def plot_subchain_history(subchain_hist, title="Subchain length history"):
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(subchain_hist, color="steelblue")
    ax.set_xlabel("Coarse iteration")
    ax.set_ylabel("Subchain length")
    ax.set_title(title)
    plt.tight_layout()
    return fig, ax


def plot_prediction_at_theta(
    surrogate, theta, x_obs, y_obs, y_true=None, title=""
):
    y_hat, y_var = surrogate.predict(theta)
    y_std = np.sqrt(np.maximum(y_var, 0.0))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x_obs, y_hat, "s--", label="Surrogate mean")
    ax.fill_between(
        x_obs, y_hat - 1.96 * y_std, y_hat + 1.96 * y_std,
        alpha=0.3, label="95 % CI",
    )
    ax.plot(x_obs, y_obs, "o", label="Observed (noisy)")
    if y_true is not None:
        ax.plot(x_obs, y_true, "k-", label="True (HF)")
    ax.set_xlabel("x")
    ax.set_ylabel("Displacement")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    return fig, ax


# =====================================================================
#  Configuration
# =====================================================================

rng = np.random.default_rng(42)

# Spatial grid and observation locations
x = make_spatial_grid(n_pts=31, length=1.0)
obs_idx = np.array([2, 5, 8, 11, 14, 17, 20, 23, 26, 29])
x_obs = x[obs_idx]

# HF forward model: theta -> y_obs (observed displacements only)
hf_forward = make_forward_model(
    x=x, obs_idx=obs_idx, load_scale=-1.0, return_full_state=False,
)

# Prior over theta = [m1, m2, m3]  (log-stiffness on three sub-intervals)
prior_mean = np.array([10.0, 10.0, 10.0])
prior_cov = np.diag([2.0**2, 2.0**2, 2.0**2])
prior = multivariate_normal(mean=prior_mean, cov=prior_cov)

# Observation noise — auto-scaled from a reference forward evaluation
y_ref = hf_forward(prior_mean)
signal_scale = float(np.max(np.abs(y_ref)))
sigma_obs = 0.02 * signal_scale          # 2 % relative noise

# Surrogate configuration  (small budget for a quick run)
n_init = 50
gp_kernel = "matern52"
gp_ard = True

# Active coupling: trigger HF when avg LF std ≈ observation noise
gamma_threshold = sigma_obs

# MCMC budget
n_coarse_evals = 1000
burn_in = 100
chunk_size = 250

print(f"signal_scale   = {signal_scale:.3e}")
print(f"sigma_obs      = {sigma_obs:.3e}")
print(f"gamma_threshold= {gamma_threshold:.3e}")


# %% [markdown]
# ## Synthetic observation
#
# Sample a true parameter from the prior, evaluate HF, corrupt with noise.

# %%
theta_true = prior.rvs(random_state=rng)
y_obs = make_observation(rng, theta_true, x, sigma_obs, obs_idx)

print(f"theta_true = {theta_true}")
print(f"y_obs      = {y_obs}")


# %% [markdown]
# ## Initial surrogate training set
#
# Sample parameters from the prior, evaluate HF, build initial design.

# %%
theta_train = np.asarray(
    [prior.rvs(random_state=rng) for _ in range(n_init)], dtype=float,
)
y_train = np.asarray([hf_forward(th) for th in theta_train], dtype=float)


# %% [markdown]
# ## Fit a direct GP surrogate on observed outputs
#
# Since the observation space is only 10 points, we skip POD and use a
# ``MultiOutputGP`` mapping ``theta → y_obs`` directly.

# %%
gp = MultiOutputGP(
    X_train=theta_train,
    Y_train=y_train,
    kernel=gp_kernel,
    ard=gp_ard,
)

# Two independent copies — one per inference mode
lf_surrogate_single = DirectGPSurrogate(gp=copy.deepcopy(gp))
lf_surrogate_adapt = DirectGPSurrogate(gp=copy.deepcopy(gp))


# %% [markdown]
# ## Build active models (LF + HF coupling)

# %%
model_single = ActiveModel(
    lf_model=lf_surrogate_single,
    hf_model=hf_forward,
    gamma_threshold=gamma_threshold,
)

model_adapt = ActiveModel(
    lf_model=lf_surrogate_adapt,
    hf_model=hf_forward,
    gamma_threshold=gamma_threshold,
)


# %% [markdown]
# ## Observation covariance

# %%
cov_obs = (sigma_obs**2) * np.eye(len(y_obs))


# %% [markdown]
# ## Sanity check: surrogate prediction before sampling

# %%
plot_prediction_at_theta(
    lf_surrogate_single,
    theta_true,
    x_obs=x_obs,
    y_obs=y_obs,
    y_true=hf_forward(theta_true),
    title="Surrogate prediction (before sampling)",
)
plt.show()


# %% [markdown]
# # Part 1 — MCMC-guided active learning (single posterior)
#
# The chain is driven by the coarse model.  HF calls occur *internally*
# when the uncertainty trigger fires inside ``model.coarse``.

# %%
print("=" * 60)
print("Part 1: MCMC-guided active learning (single posterior)")
print("=" * 60)

theta0 = prior_mean.copy()

proposal_single = AdaptiveProposal(
    C0=0.1 * prior_cov,
    period=100,
    sd=1.0,
)

result_single = sample_active_mcmc(
    model=model_single,
    y_obs=y_obs,
    cov_obs=cov_obs,
    prior=prior,
    theta0=theta0,
    n_iter=n_coarse_evals,
    proposal=proposal_single,
    rng=rng,
)

print_summary(
    result_single["samples"],
    result_single["used_hf"],
    result_single["accepted"],
    theta_true=theta_true,
    burn_in=burn_in,
)

# %%
plot_chain_2d(
    result_single["samples"],
    result_single["used_hf"],
    theta_true=theta_true,
    labels=("m1", "m2"),
    title="Single posterior — samples (m1 vs m2)",
)
plt.show()

plot_cumulative_hf_fraction(
    result_single["used_hf"],
    title="Single posterior — cumulative HF fraction",
)
plt.show()


# %% [markdown]
# # Part 2 — DA-MCMC guided active learning with adaptive subchain
#
# The coarse sub-chain runs for an adaptive number of steps, then a fine
# (HF) correction updates the surrogate.  The DA ratio decides acceptance
# at the fine level.  The subchain length adapts based on LF-HF RMSE.

# %%
print("\n" + "=" * 60)
print("Part 2: DA-MCMC with adaptive subchain")
print("=" * 60)

theta0 = prior_mean.copy()

proposal_adapt = AdaptiveProposal(
    C0=0.1 * prior_cov,
    period=100,
    sd=1.0,
)

adaptive_policy = AdaptiveSubchainPolicy(
    subchain_length=20,
    update_every=5,
    target_error=0.05 * signal_scale,   # scaled to the problem's magnitude
    min_subchain=1,
    max_subchain=500,
    grow_factor=2.0,
    shrink_factor=0.5,
)

result_adapt = sample_da_active_mcmc(
    model=model_adapt,
    y_obs=y_obs,
    cov_obs=cov_obs,
    prior=prior,
    theta0=theta0,
    n_coarse_evals=n_coarse_evals,
    proposal=proposal_adapt,
    policy=adaptive_policy,
    rng=rng,
)

print_summary(
    result_adapt["samples"],
    result_adapt["used_hf"],
    result_adapt["accepted"],
    theta_true=theta_true,
    burn_in=burn_in,
)

# %%
plot_chain_2d(
    result_adapt["samples"],
    result_adapt["used_hf"],
    theta_true=theta_true,
    labels=("m1", "m2"),
    title="DA-MCMC — samples (m1 vs m2)",
)
plt.show()

plot_cumulative_hf_fraction(
    result_adapt["used_hf"],
    title="DA-MCMC — cumulative HF fraction",
)
plt.show()

if len(result_adapt["subchain_history"]) > 0:
    plot_subchain_history(
        result_adapt["subchain_history"],
        title="DA-MCMC — subchain length history",
    )
    plt.show()


# %% [markdown]
# ## Post-sampling: surrogate prediction at theta_true
#
# After sampling, the surrogate has been enriched with HF evaluations
# triggered during the chain.  Compare the improved prediction.

# %%
plot_prediction_at_theta(
    lf_surrogate_adapt,
    theta_true,
    x_obs=x_obs,
    y_obs=y_obs,
    y_true=hf_forward(theta_true),
    title="Surrogate prediction (after DA-MCMC sampling)",
)
plt.show()
