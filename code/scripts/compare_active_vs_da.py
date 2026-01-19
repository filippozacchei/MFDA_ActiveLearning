from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from tqdm import tqdm

from gp_active_mcmc.utils import set_seed
from gp_active_mcmc.toy import toy_forward, make_timeline, make_observation
from gp_active_mcmc.pod import POD
from gp_active_mcmc.gp_surrogate import GPSurrogate
from gp_active_mcmc.priors import GaussianPrior


# -------------------------
# Likelihood (log)
# -------------------------
def loglike_gaussian_iid(y: np.ndarray, y_obs: np.ndarray, sigma: float) -> float:
    r = y - y_obs
    return float(-0.5 * np.sum((r / sigma) ** 2))


def rmse_vec(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


# -------------------------
# Design
# -------------------------
def make_design_gaussian(
    rng: np.random.Generator, mean: np.ndarray, cov: np.ndarray, n: int, tau_min=1e-6
) -> np.ndarray:
    X = rng.multivariate_normal(mean=mean, cov=cov, size=n)
    bad = X[:, 2] <= tau_min
    while np.any(bad):
        X[bad] = rng.multivariate_normal(mean=mean, cov=cov, size=int(np.sum(bad)))
        bad = X[:, 2] <= tau_min
    return X


def sample_theta_from_prior(rng, mean, cov, tau_min=1e-6):
    while True:
        th = rng.multivariate_normal(mean=mean, cov=cov)
        if th[2] > tau_min:
            return th


# -------------------------
# POD+GP emulator
# -------------------------
@dataclass
class PODGPEmulator:
    pod: POD
    gps: list[GPSurrogate]
    coeff_var_floor: float = 1e-12

    def predict_coeffs(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns mean/variance for POD coefficients a_k(theta).
        NOTE: Ensure GPSurrogate.predict_loglike() actually predicts coefficient mean/var.
        """
        r = len(self.gps)
        mu = np.zeros(r)
        var = np.zeros(r)
        for k, gpk in enumerate(self.gps):
            mk, vk = gpk.predict_loglike(theta)  # <-- must return coeff mean/var
            mu[k] = float(mk)
            var[k] = float(vk)
        var = np.maximum(var, self.coeff_var_floor)
        return mu, var

    def predict_series(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mu_a, var_a = self.predict_coeffs(theta)
        y_hat = self.pod.reconstruct(mu_a.reshape(1, -1))[0]
        Phi = self.pod.phi_  # (T,r)
        # independent-coefficient variance propagation
        y_var = (Phi**2) @ var_a
        y_std = np.sqrt(np.maximum(y_var, 1e-14))
        return y_hat, y_std

    def update_from_true_series(
        self, theta: np.ndarray, y_true: np.ndarray, gamma_L_ratio: float, n_retrain_max: int
    ) -> None:
        a_true = self.pod.project(y_true.reshape(1, -1))[0]
        for k, gpk in enumerate(self.gps):
            gpk.update(theta, float(a_true[k]), gamma_L_ratio, n_retrain_max)


# -------------------------
# MH accept in log domain
# -------------------------
def mh_accept_log(rng: np.random.Generator, logpost_star: float, logpost_old: float) -> bool:
    if np.isneginf(logpost_star):
        return False
    d = logpost_star - logpost_old
    return np.log(rng.uniform(0.0, 1.0)) < d


# -------------------------
# Whitening transform
# -------------------------
@dataclass
class WhitenTransform:
    mean: np.ndarray
    L: np.ndarray  # Cholesky factor of covariance (theta space)

    def to_z(self, theta: np.ndarray) -> np.ndarray:
        return np.linalg.solve(self.L, theta - self.mean)

    def to_theta(self, z: np.ndarray) -> np.ndarray:
        return self.mean + self.L @ z


# -------------------------
# Correct online covariance (Welford M2)
# -------------------------
@dataclass
class OnlineCov:
    d: int
    mean: np.ndarray
    M2: np.ndarray  # sum of centered outer products
    n: int = 1

    @classmethod
    def init_from(cls, x0: np.ndarray) -> "OnlineCov":
        d = x0.size
        return cls(d=d, mean=x0.copy(), M2=np.zeros((d, d)), n=1)

    def update(self, x: np.ndarray) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean = self.mean + delta / self.n
        delta2 = x - self.mean
        self.M2 = self.M2 + np.outer(delta, delta2)

    def cov(self) -> np.ndarray:
        if self.n <= 1:
            return np.eye(self.d)
        return self.M2 / (self.n - 1)


# -------------------------
# Sampler results container
# -------------------------
@dataclass
class RunResult:
    method: str
    p_stage2: float

    chain: np.ndarray            # (n_total+1, dtheta)
    accepted: np.ndarray         # (n_total,)
    used_forward: np.ndarray     # (n_total,)
    updated_gp: np.ndarray       # (n_total,)
    stage1_acc: np.ndarray       # (n_total,) for DA else zeros
    stage2_acc: np.ndarray       # (n_total,) for DA else zeros

    u_prop: np.ndarray           # (n_total,) mean std_y at proposal
    rmse_prop: np.ndarray        # (n_total,) rmse(y_hat,y_true) when forward called else nan
    step_hist: np.ndarray        # (n_total,) proposal step size in z-space

    # NEW: track surrogate quality at theta_true (every iteration)
    rmse_true_theta: np.ndarray  # (n_total,) surrogate RMSE at theta_true
    u_true_theta: np.ndarray     # (n_total,) surrogate mean std at theta_true

    # NEW: track sampling distance to theta_true
    dist_to_true: np.ndarray     # (n_total+1,)

    reject_prior: int
    reject_constraint: int


# -------------------------
# Rolling helper
# -------------------------
def rolling_nanmean(x: np.ndarray, w: int) -> np.ndarray:
    """Rolling mean ignoring NaNs; returns array same length as x with NaN for insufficient window."""
    out = np.full_like(x, np.nan, dtype=float)
    if w <= 1:
        return x.astype(float)
    for i in range(len(x)):
        a = max(0, i - w + 1)
        seg = x[a : i + 1]
        m = np.nanmean(seg)
        out[i] = m
    return out


# -------------------------
# One run (baseline or DA)
# -------------------------
def run_sampler(
    *,
    rng: np.random.Generator,
    method: str,
    p_stage2: float,

    t: np.ndarray,
    y_obs: np.ndarray,
    sigma_obs: float,

    prior: GaussianPrior,
    W: WhitenTransform,

    emul: PODGPEmulator,

    theta0: np.ndarray,
    theta_true: np.ndarray,      # NEW
    n_total: int,

    # AM knobs
    target_accept: float,
    adapt_until: int,
    step0: float,
    step_min: float,
    step_max: float,

    # active-learning knobs
    u_gate: float,
    err_thresh: float,
    gamma_L_ratio: float,
    n_retrain_max: int,

    # constraints
    tau_min: float = 1e-6,

    # scaling
    eps_cov: float = 1e-6,
) -> RunResult:
    assert method in ("baseline_gate", "da_mh")

    dtheta = theta0.size
    d = dtheta

    chain = np.zeros((n_total + 1, dtheta))
    chain[0] = theta0
    accepted = np.zeros(n_total, dtype=bool)
    used_forward = np.zeros(n_total, dtype=bool)
    updated_gp = np.zeros(n_total, dtype=bool)

    stage1_acc = np.zeros(n_total, dtype=bool)
    stage2_acc = np.zeros(n_total, dtype=bool)

    u_prop = np.zeros(n_total, dtype=float)
    rmse_prop = np.full(n_total, np.nan, dtype=float)
    step_hist = np.zeros(n_total, dtype=float)

    # NEW tracking
    rmse_true_theta = np.full(n_total, np.nan, dtype=float)
    u_true_theta = np.full(n_total, np.nan, dtype=float)
    dist_to_true = np.zeros(n_total + 1, dtype=float)
    dist_to_true[0] = float(np.linalg.norm(theta0 - theta_true))

    reject_prior = 0
    reject_constraint = 0

    # z-state
    z_chain = np.zeros((n_total + 1, d))
    z_chain[0] = W.to_z(theta0)

    # AM state (in z-space)
    step = float(step0)
    oc = OnlineCov.init_from(z_chain[0])

    # Precompute truth trajectory at theta_true once (cheap and stable)
    y_true_theta = toy_forward(theta_true, t)

    # cache true loglikes for DA (only current state needed)
    ll_true_curr = None
    y_true_curr = None

    # surrogate logpost at current (always available)
    y_hat0, _ = emul.predict_series(theta0)
    ll_hat_curr = loglike_gaussian_iid(y_hat0, y_obs, sigma_obs)
    lp_curr = prior.logpdf(theta0)
    logpost_hat_curr = ll_hat_curr + lp_curr

    # true logpost at start
    y0 = toy_forward(theta0, t)
    ll_true_curr = loglike_gaussian_iid(y0, y_obs, sigma_obs)
    y_true_curr = y0
    logpost_true_curr = ll_true_curr + lp_curr

    for n in tqdm(range(n_total), desc=f"run {method}, p2={p_stage2:g}"):
        theta_n = chain[n].copy()
        z_n = z_chain[n].copy()

        # --- track surrogate improvement at theta_true (every iter) ---
        y_hat_tt, y_std_tt = emul.predict_series(theta_true)
        rmse_true_theta[n] = rmse_vec(y_hat_tt, y_true_theta)
        u_true_theta[n] = float(np.mean(y_std_tt))

        # covariance for proposal
        cov_z = oc.cov() + eps_cov * np.eye(d)
        if not ((n < adapt_until) and (oc.n >= 5)):
            # freeze logic could be inserted here; kept simple and stable
            pass

        # Haario scaling
        scale_mat = (2.38**2 / d) * cov_z

        z_star = z_n + step * rng.multivariate_normal(np.zeros(d), scale_mat)
        theta_star = W.to_theta(z_star)

        # constraints
        if theta_star[2] <= tau_min:
            reject_constraint += 1
            chain[n + 1] = theta_n
            z_chain[n + 1] = z_n
            if n < adapt_until:
                oc.update(z_n)
            step_hist[n] = step
            dist_to_true[n + 1] = float(np.linalg.norm(chain[n + 1] - theta_true))
            continue

        lp_star = prior.logpdf(theta_star)
        if np.isneginf(lp_star):
            reject_prior += 1
            chain[n + 1] = theta_n
            z_chain[n + 1] = z_n
            if n < adapt_until:
                oc.update(z_n)
            step_hist[n] = step
            dist_to_true[n + 1] = float(np.linalg.norm(chain[n + 1] - theta_true))
            continue

        # surrogate at proposal
        y_hat_star, y_std_star = emul.predict_series(theta_star)
        ll_hat_star = loglike_gaussian_iid(y_hat_star, y_obs, sigma_obs)
        logpost_hat_star = ll_hat_star + lp_star

        ubar = float(np.mean(y_std_star))
        u_prop[n] = ubar

        if method == "baseline_gate":
            # Gate: if uncertain, use true model
            if ubar > u_gate:
                y_true_star = toy_forward(theta_star, t)
                ll_star = loglike_gaussian_iid(y_true_star, y_obs, sigma_obs)
                used_forward[n] = True

                rmse_prop[n] = rmse_vec(y_hat_star, y_true_star)

                if rmse_prop[n] > err_thresh:
                    emul.update_from_true_series(theta_star, y_true_star, gamma_L_ratio, n_retrain_max)
                    updated_gp[n] = True
            else:
                ll_star = ll_hat_star

            logpost_star = ll_star + lp_star
            is_acc = mh_accept_log(rng, logpost_star, logpost_true_curr)
            accepted[n] = is_acc

            if is_acc:
                chain[n + 1] = theta_star
                z_chain[n + 1] = z_star

                if used_forward[n]:
                    ll_true_curr = ll_star
                    y_true_curr = y_true_star
                    logpost_true_curr = logpost_star
                else:
                    # pragmatic: treat surrogate as truth when no forward is called
                    logpost_true_curr = logpost_star

                logpost_hat_curr = logpost_hat_star
            else:
                chain[n + 1] = theta_n
                z_chain[n + 1] = z_n

            if n < adapt_until:
                oc.update(z_star if is_acc else z_n)

        else:
            # -------------------------
            # Delayed-acceptance MH
            # -------------------------
            is_acc1 = mh_accept_log(rng, logpost_hat_star, logpost_hat_curr)
            stage1_acc[n] = is_acc1

            if not is_acc1:
                accepted[n] = False
                chain[n + 1] = theta_n
                z_chain[n + 1] = z_n
                if n < adapt_until:
                    oc.update(z_n)
                step_hist[n] = step
            else:
                do_stage2 = (ubar > u_gate) or (rng.uniform(0.0, 1.0) < p_stage2)

                if not do_stage2:
                    accepted[n] = True
                    chain[n + 1] = theta_star
                    z_chain[n + 1] = z_star
                    logpost_hat_curr = logpost_hat_star
                    if n < adapt_until:
                        oc.update(z_star)
                    step_hist[n] = step
                else:
                    y_true_star = toy_forward(theta_star, t)
                    ll_true_star = loglike_gaussian_iid(y_true_star, y_obs, sigma_obs)
                    logpost_true_star = ll_true_star + lp_star
                    used_forward[n] = True

                    if ll_true_curr is None:
                        y_true_curr = toy_forward(theta_n, t)
                        ll_true_curr = loglike_gaussian_iid(y_true_curr, y_obs, sigma_obs)
                        logpost_true_curr = ll_true_curr + prior.logpdf(theta_n)

                    corr_star = logpost_true_star - logpost_hat_star
                    corr_curr = logpost_true_curr - logpost_hat_curr
                    log_alpha2 = (corr_star - corr_curr)

                    is_acc2 = (np.log(rng.uniform(0.0, 1.0)) < log_alpha2)
                    stage2_acc[n] = is_acc2
                    accepted[n] = is_acc2

                    rmse_prop[n] = rmse_vec(y_hat_star, y_true_star)

                    if rmse_prop[n] > err_thresh:
                        emul.update_from_true_series(theta_star, y_true_star, gamma_L_ratio, n_retrain_max)
                        updated_gp[n] = True

                    if is_acc2:
                        chain[n + 1] = theta_star
                        z_chain[n + 1] = z_star
                        logpost_hat_curr = logpost_hat_star
                        ll_true_curr = ll_true_star
                        y_true_curr = y_true_star
                        logpost_true_curr = logpost_true_star
                        if n < adapt_until:
                            oc.update(z_star)
                    else:
                        chain[n + 1] = theta_n
                        z_chain[n + 1] = z_n
                        if n < adapt_until:
                            oc.update(z_n)

        # Robbins–Monro step adaptation
        if n < adapt_until and (n + 1) % 25 == 0:
            acc_rate_local = float(np.mean(accepted[max(0, n - 500): n + 1]))
            step *= np.exp(0.05 * (acc_rate_local - target_accept))
            step = float(np.clip(step, step_min, step_max))

        step_hist[n] = step
        dist_to_true[n + 1] = float(np.linalg.norm(chain[n + 1] - theta_true))

    return RunResult(
        method=method,
        p_stage2=float(p_stage2),
        chain=chain,
        accepted=accepted,
        used_forward=used_forward,
        updated_gp=updated_gp,
        stage1_acc=stage1_acc,
        stage2_acc=stage2_acc,
        u_prop=u_prop,
        rmse_prop=rmse_prop,
        step_hist=step_hist,
        rmse_true_theta=rmse_true_theta,
        u_true_theta=u_true_theta,
        dist_to_true=dist_to_true,
        reject_prior=reject_prior,
        reject_constraint=reject_constraint,
    )


# -------------------------
# Plot helpers
# -------------------------
def plot_run_summary(res: RunResult, theta_true: np.ndarray, names=("A", "f", "tau"), burn=0, thin=1) -> None:
    chain_thin = res.chain[burn::thin]
    it = np.arange(res.accepted.size)

    # running rates
    acc_run = np.cumsum(res.accepted) / (it + 1)
    fw_run = np.cumsum(res.used_forward) / (it + 1)
    upd_run = np.cumsum(res.updated_gp) / (it + 1)

    # rolling summaries for sparse RMSE at proposals
    rmse_prop_roll = rolling_nanmean(res.rmse_prop, w=200)
    u_prop_roll = rolling_nanmean(res.u_prop, w=200)

    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(5, 3)

    ax0  = fig.add_subplot(gs[0, :])     # rates
    ax1  = fig.add_subplot(gs[1, 0])     # trace A
    ax2  = fig.add_subplot(gs[1, 1])     # trace f
    ax3  = fig.add_subplot(gs[1, 2])     # trace tau
    ax4  = fig.add_subplot(gs[2, 0])     # u_prop
    ax5  = fig.add_subplot(gs[2, 1])     # rmse_prop
    ax6  = fig.add_subplot(gs[2, 2])     # DA stages
    ax7  = fig.add_subplot(gs[3, 0])     # surrogate rmse at theta_true
    ax8  = fig.add_subplot(gs[3, 1])     # surrogate u at theta_true
    ax9  = fig.add_subplot(gs[3, 2])     # distance to true
    ax10 = fig.add_subplot(gs[4, 0])     # pair A-f
    ax11 = fig.add_subplot(gs[4, 1])     # pair A-tau
    ax12 = fig.add_subplot(gs[4, 2])     # pair f-tau

    title = f"{res.method} | p_stage2={res.p_stage2:g}"
    fig.suptitle(title, fontsize=12)

    # rates + step (two axes)
    ax0.plot(acc_run, label="accept rate")
    ax0.plot(fw_run, label="forward-call fraction")
    ax0.plot(upd_run, label="GP-update fraction")
    ax0.set_ylim(0, 1.05)
    ax0.grid(True)

    ax0b = ax0.twinx()
    ax0b.plot(res.step_hist, linestyle="--", label="step(z)")
    ax0b.set_yscale("log")
    ax0b.set_ylabel("step (log)")

    lines1, labels1 = ax0.get_legend_handles_labels()
    lines2, labels2 = ax0b.get_legend_handles_labels()
    ax0.legend(lines1 + lines2, labels1 + labels2, loc="best")

    # traces with true line
    for j, ax in enumerate([ax1, ax2, ax3]):
        ax.plot(res.chain[:, j], alpha=0.85)
        ax.axhline(float(theta_true[j]), linestyle="--")
        ax.set_title(f"trace {names[j]} (true dashed)")
        ax.grid(True)

    # proposal uncertainty
    ax4.plot(res.u_prop, alpha=0.25, label="u_prop")
    ax4.plot(u_prop_roll, alpha=0.9, label="rolling mean (w=200)")
    ax4.set_yscale("log")
    ax4.set_title("proposal uncertainty u = mean_t std_y (log)")
    ax4.grid(True); ax4.legend(loc="best")

    # proposal RMSE where forward called
    ax5.plot(res.rmse_prop, alpha=0.25, label="rmse_prop (NaN if no forward)")
    ax5.plot(rmse_prop_roll, alpha=0.9, label="rolling mean (w=200)")
    ax5.set_yscale("log")
    ax5.set_title("proposal RMSE(surrogate vs true) when forward called (log)")
    ax5.grid(True); ax5.legend(loc="best")

    # DA stage rates
    ax6.plot(np.cumsum(res.stage1_acc) / (it + 1), label="stage1 accept")
    ax6.plot(np.cumsum(res.stage2_acc) / (it + 1), label="stage2 accept")
    ax6.set_ylim(0, 1.05)
    ax6.set_title("DA stage acceptance (baseline ~0)")
    ax6.grid(True); ax6.legend(loc="best")

    # surrogate improvement at theta_true
    ax7.plot(res.rmse_true_theta, alpha=0.9)
    ax7.set_yscale("log")
    ax7.set_title("Surrogate RMSE at θ_true (log) — should decrease with updates")
    ax7.grid(True)

    ax8.plot(res.u_true_theta, alpha=0.9)
    ax8.set_yscale("log")
    ax8.set_title("Surrogate mean std at θ_true (log)")
    ax8.grid(True)

    # sampling improvement: distance to true
    ax9.plot(res.dist_to_true, alpha=0.9)
    ax9.set_yscale("log")
    ax9.set_title("||θ - θ_true|| (log) — should shrink as chain finds posterior mass")
    ax9.grid(True)

    # pair scatter with true marker
    ax10.scatter(chain_thin[:, 0], chain_thin[:, 1], s=8, alpha=0.25)
    ax10.scatter(theta_true[0], theta_true[1], s=80, marker="*", label="θ_true")
    ax10.set_xlabel(names[0]); ax10.set_ylabel(names[1])
    ax10.set_title("pair: A vs f")
    ax10.grid(True); ax10.legend(loc="best")

    ax11.scatter(chain_thin[:, 0], chain_thin[:, 2], s=8, alpha=0.25)
    ax11.scatter(theta_true[0], theta_true[2], s=80, marker="*", label="θ_true")
    ax11.set_xlabel(names[0]); ax11.set_ylabel(names[2])
    ax11.set_title("pair: A vs tau")
    ax11.grid(True); ax11.legend(loc="best")

    ax12.scatter(chain_thin[:, 1], chain_thin[:, 2], s=8, alpha=0.25)
    ax12.scatter(theta_true[1], theta_true[2], s=80, marker="*", label="θ_true")
    ax12.set_xlabel(names[1]); ax12.set_ylabel(names[2])
    ax12.set_title("pair: f vs tau")
    ax12.grid(True); ax12.legend(loc="best")

    plt.tight_layout()
    plt.show()


def plot_compare_overlays(results: list[RunResult], theta_true: np.ndarray, names=("A", "f", "tau"), burn=0, thin=1) -> None:
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(4, 3)

    axR  = fig.add_subplot(gs[0, :])   # running rates
    axS  = fig.add_subplot(gs[1, 0])   # scatter A vs f
    axD  = fig.add_subplot(gs[1, 1])   # dist to true
    axTT = fig.add_subplot(gs[1, 2])   # surrogate RMSE at theta_true

    axU  = fig.add_subplot(gs[2, 0])   # u_prop
    axE  = fig.add_subplot(gs[2, 1])   # rmse_prop
    axUT = fig.add_subplot(gs[2, 2])   # u_true_theta

    axH0 = fig.add_subplot(gs[3, 0])
    axH1 = fig.add_subplot(gs[3, 1])
    axH2 = fig.add_subplot(gs[3, 2])

    for res in results:
        it = np.arange(res.accepted.size)
        label = f"{res.method}, p2={res.p_stage2:g}"

        axR.plot(np.cumsum(res.accepted) / (it + 1), label=f"acc {label}")
        axR.plot(np.cumsum(res.used_forward) / (it + 1), linestyle="--", label=f"fw {label}")
        axR.plot(np.cumsum(res.updated_gp) / (it + 1), linestyle=":", label=f"upd {label}")

        chain = res.chain[burn::thin]
        axS.scatter(chain[:, 0], chain[:, 1], s=6, alpha=0.12, label=label)

        axD.plot(res.dist_to_true, alpha=0.8, label=label)
        axTT.plot(res.rmse_true_theta, alpha=0.8, label=label)

        axU.plot(res.u_prop, alpha=0.15, label=label)
        axE.plot(res.rmse_prop, alpha=0.15, label=label)
        axUT.plot(res.u_true_theta, alpha=0.8, label=label)

        axH0.hist(chain[:, 0], bins=60, density=True, alpha=0.25, label=label)
        axH1.hist(chain[:, 1], bins=60, density=True, alpha=0.25, label=label)
        axH2.hist(chain[:, 2], bins=60, density=True, alpha=0.25, label=label)

    axR.set_title("Running acceptance / forward-call / GP-update fractions")
    axR.grid(True); axR.legend(loc="best", fontsize=8)

    axS.scatter(theta_true[0], theta_true[1], s=120, marker="*", label="θ_true")
    axS.set_title(f"Scatter overlay: {names[0]} vs {names[1]}")
    axS.set_xlabel(names[0]); axS.set_ylabel(names[1])
    axS.grid(True); axS.legend(loc="best", fontsize=8)

    axD.set_yscale("log")
    axD.set_title("||θ - θ_true|| (log) overlay")
    axD.grid(True); axD.legend(loc="best", fontsize=8)

    axTT.set_yscale("log")
    axTT.set_title("Surrogate RMSE at θ_true (log) overlay")
    axTT.grid(True); axTT.legend(loc="best", fontsize=8)

    axU.set_yscale("log")
    axU.set_title("u_prop (log) overlay")
    axU.grid(True)

    axE.set_yscale("log")
    axE.set_title("rmse_prop (log) overlay; NaN if no forward")
    axE.grid(True)

    axUT.set_yscale("log")
    axUT.set_title("Surrogate u at θ_true (log) overlay")
    axUT.grid(True)

    axH0.set_title(f"Posterior histogram: {names[0]}")
    axH1.set_title(f"Posterior histogram: {names[1]}")
    axH2.set_title(f"Posterior histogram: {names[2]}")
    for ax in (axH0, axH1, axH2):
        ax.grid(True)
        ax.legend(loc="best", fontsize=8)

    plt.tight_layout()
    plt.show()


# -------------------------
# Main experiment
# -------------------------
def main():
    # ---------- knobs ----------
    seed = 32534
    N0 = 50
    r_pod = 10
    kernel = "matern52"
    ard = True

    sigma_obs = 0.1
    n_total = 5000

    # AM knobs
    target_accept = 0.25
    adapt_until = 5000
    step0 = 0.5
    step_min, step_max = 1e-3, 3.0

    # gating + update policy
    u_gate = 0.0025
    err_thresh = 0.01
    gamma_L_ratio = 1.5
    n_retrain_max = 50

    # compare DA subsampling rates
    p_stage2_list = [1.0, 0.5, 0.2, 0.1]

    # plotting
    names = ("A", "f", "tau")
    burn = int(0.25 * n_total)
    thin = 5

    # ---------- setup ----------
    rng = set_seed(seed)
    t = make_timeline(T=500, t_end=0.05)
    
    prior_mean = np.array([0.8, 150.0, 0.010])
    prior_cov = np.diag([0.25**2, 40.0**2, 0.004**2])
    prior = GaussianPrior(prior_mean, prior_cov)

    
    theta_true = sample_theta_from_prior(rng, prior_mean, prior_cov, tau_min=1e-6)
    y_obs = make_observation(rng, theta_true, t, sigma_obs)
    # initial emulator design
    X0 = make_design_gaussian(rng, prior_mean, prior_cov, N0)
    Y0 = np.array([toy_forward(X0[i], t) for i in range(N0)])

    # whitening transform
    L_prior = np.linalg.cholesky(prior_cov + 1e-12 * np.eye(3))
    W = WhitenTransform(mean=prior_mean, L=L_prior)

    theta0 = sample_theta_from_prior(rng, prior_mean, prior_cov, tau_min=1e-6)

    # fresh emulator per run (important for fair comparison)
    def fresh_emulator():
        pod_local = POD(r=r_pod).fit(Y0)
        A0_local = pod_local.project(Y0)
        gps_local = [GPSurrogate(X0, A0_local[:, k], kernel=kernel, ard=ard) for k in range(r_pod)]
        return PODGPEmulator(pod=pod_local, gps=gps_local)

    results: list[RunResult] = []

    # baseline
    res0 = run_sampler(
        rng=rng,
        method="baseline_gate",
        p_stage2=0.0,
        t=t, y_obs=y_obs, sigma_obs=sigma_obs,
        prior=prior, W=W,
        emul=fresh_emulator(),
        theta0=theta0,
        theta_true=theta_true,
        n_total=n_total,
        target_accept=target_accept,
        adapt_until=adapt_until,
        step0=step0, step_min=step_min, step_max=step_max,
        u_gate=u_gate,
        err_thresh=err_thresh,
        gamma_L_ratio=gamma_L_ratio,
        n_retrain_max=n_retrain_max,
    )
    results.append(res0)

    # DA runs
    for p2 in p_stage2_list:
        res = run_sampler(
            rng=rng,
            method="da_mh",
            p_stage2=p2,
            t=t, y_obs=y_obs, sigma_obs=sigma_obs,
            prior=prior, W=W,
            emul=fresh_emulator(),
            theta0=theta0,
            theta_true=theta_true,
            n_total=n_total,
            target_accept=target_accept,
            adapt_until=adapt_until,
            step0=step0, step_min=step_min, step_max=step_max,
            u_gate=u_gate,
            err_thresh=err_thresh,
            gamma_L_ratio=gamma_L_ratio,
            n_retrain_max=n_retrain_max,
        )
        results.append(res)

    # -------------------------
    # Plots
    # -------------------------
    for res in results:
        plot_run_summary(res, theta_true=theta_true, names=names, burn=burn, thin=thin)

    plot_compare_overlays(results, theta_true=theta_true, names=names, burn=burn, thin=thin)

    # -------------------------
    # Diagnostics
    # -------------------------
    print("\n=== Diagnostics (rates are full-run) ===")
    for res in results:
        acc = float(np.mean(res.accepted))
        fw = float(np.mean(res.used_forward))
        upd = float(np.mean(res.updated_gp))
        s1 = float(np.mean(res.stage1_acc)) if np.any(res.stage1_acc) else 0.0
        s2 = float(np.mean(res.stage2_acc)) if np.any(res.stage2_acc) else 0.0

        # surrogate at theta_true improvement summary
        rmse_tt_final = float(res.rmse_true_theta[-1])
        u_tt_final = float(res.u_true_theta[-1])

        print(
            f"{res.method:14s} p2={res.p_stage2:4.2f} | "
            f"acc={acc:6.3f} fw={fw:6.3f} upd={upd:6.3f} "
            f"s1={s1:6.3f} s2={s2:6.3f} "
            f"rmse_tt(final)={rmse_tt_final:.3e} u_tt(final)={u_tt_final:.3e} "
            f"rej_prior={res.reject_prior} rej_tau={res.reject_constraint}"
        )


if __name__ == "__main__":
    main()
