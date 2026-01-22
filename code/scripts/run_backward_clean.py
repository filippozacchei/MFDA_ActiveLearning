from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

from gp_active_mcmc.utils import set_seed
from gp_active_mcmc.toy import toy_forward, make_timeline, make_observation
from gp_active_mcmc.pod import POD
from gp_active_mcmc.gp_surrogate import GPSurrogate
from gp_active_mcmc.priors import GaussianPrior
from gp_active_mcmc.likelihood import loglike_theta_gp_adjusted, loglike_theta

from matplotlib.colors import LogNorm


# -------------------------
# Likelihood (log)
# -------------------------
def loglike_gaussian_iid(y: np.ndarray, y_obs: np.ndarray, sigma: float) -> float:
    r = y - y_obs
    return float(-0.5 * np.sum((r / sigma) ** 2))

def loglike_gaussian_hetero(y_mean: np.ndarray, y_obs: np.ndarray,
                           sigma_obs: float, y_std: np.ndarray) -> float:
    # total variance per time point
    var = sigma_obs**2 + np.maximum(y_std, 0.0)**2
    r = y_obs - y_mean
    return float(-0.5 * np.sum((r**2) / var + np.log(var)))

def fd_jacobian_theta(theta: np.ndarray, t: np.ndarray, eps_rel: float = 1e-6) -> np.ndarray:
    """
    Finite-difference Jacobian J = dy/dtheta at theta.
    Returns J with shape (T, d).
    """
    theta = theta.astype(float).copy()
    y0 = toy_forward(theta, t)
    T = y0.size
    d = theta.size
    J = np.zeros((T, d), dtype=float)

    for j in range(d):
        h = eps_rel * (abs(theta[j]) + 1.0)
        thp = theta.copy()
        thp[j] += h
        yp = toy_forward(thp, t)
        J[:, j] = (yp - y0) / h
    return J


def gauss_newton_ls(
    theta0: np.ndarray,
    t: np.ndarray,
    y_obs: np.ndarray,
    sigma_obs: float,
    tau_min: float = 1e-6,
    n_iter: int = 25,
    lam0: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Constrained nonlinear LS via damped Gauss–Newton (Levenberg-style),
    with projection tau >= tau_min. Returns (theta_ls, cov_ls).
    cov_ls is the Gauss–Newton covariance approx: sigma^2 (J^T J)^{-1}.
    """
    theta = theta0.astype(float).copy()
    theta[2] = max(theta[2], tau_min)
    lam = float(lam0)

    def obj(th: np.ndarray) -> float:
        r = (toy_forward(th, t) - y_obs) / sigma_obs
        return 0.5 * float(np.dot(r, r))

    f_old = obj(theta)

    for _ in range(n_iter):
        y = toy_forward(theta, t)
        r = (y - y_obs) / sigma_obs                          # (T,)
        J = fd_jacobian_theta(theta, t) / sigma_obs          # (T,d) weighted
        H = J.T @ J                                          # (d,d)
        g = J.T @ r                                          # (d,)

        # damped GN step: (H + lam I) delta = -g
        A = H + lam * np.eye(theta.size)
        try:
            delta = -np.linalg.solve(A, g)
        except np.linalg.LinAlgError:
            delta = -np.linalg.pinv(A) @ g

        # backtracking on objective + keep tau feasible
        alpha = 1.0
        accepted = False
        for _bt in range(12):
            th_new = theta + alpha * delta
            th_new[2] = max(th_new[2], tau_min)
            f_new = obj(th_new)
            if np.isfinite(f_new) and f_new <= f_old:
                theta = th_new
                f_old = f_new
                lam = max(lam * 0.5, 1e-12)
                accepted = True
                break
            alpha *= 0.5

        if not accepted:
            lam = min(lam * 5.0, 1e12)

        # small step -> stop
        if np.linalg.norm(alpha * delta) < 1e-10:
            break

    # covariance at LS (Gauss–Newton)
    J = fd_jacobian_theta(theta, t)                          # unweighted
    JTJ = J.T @ J
    try:
        cov = (sigma_obs ** 2) * np.linalg.inv(JTJ)
    except np.linalg.LinAlgError:
        cov = (sigma_obs ** 2) * np.linalg.pinv(JTJ)

    # jitter for PD Cholesky downstream
    cov = 0.5 * (cov + cov.T)
    cov = cov + 1e-12 * np.eye(theta.size)
    return theta, cov

# -------------------------
# Design
# -------------------------
def make_design_gaussian(rng: np.random.Generator, mean: np.ndarray, cov: np.ndarray, n: int, tau_min=1e-6):
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
        r = len(self.gps)
        mu = np.zeros(r)
        var = np.zeros(r)
        for k, gpk in enumerate(self.gps):
            mk, vk = gpk.predict_loglike(theta)  # scalar mean/var
            mu[k] = float(mk)
            var[k] = float(vk)
        var = np.maximum(var, self.coeff_var_floor)
        return mu, var

    def predict_series(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mu_a, var_a = self.predict_coeffs(theta)
        y_hat = self.pod.reconstruct(mu_a.reshape(1, -1))[0]
        Phi = self.pod.phi_  # (T,r)
        y_var = (Phi**2) @ var_a
        y_std = np.sqrt(np.maximum(y_var, 1e-14))
        return y_hat, y_std

    def update_from_true_series(self, theta: np.ndarray, y_true: np.ndarray, gamma_L_ratio: float, n_retrain_max: int):
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
# 2D maps (uncertainty + true error for toy)
# -------------------------
def grid_2d_bounds(X: np.ndarray, idx_x: int, idx_y: int, p_low=1.0, p_high=99.0):
    x_min, x_max = np.percentile(X[:, idx_x], [p_low, p_high])
    y_min, y_max = np.percentile(X[:, idx_y], [p_low, p_high])
    return float(x_min), float(x_max), float(y_min), float(y_max)


def eval_maps_on_slice(
    emul: PODGPEmulator,
    t: np.ndarray,
    theta_center: np.ndarray,
    idx_x: int,
    idx_y: int,
    x_min: float, x_max: float,
    y_min: float, y_max: float,
    grid: int = 60,
):
    xs = np.linspace(x_min, x_max, grid)
    ys = np.linspace(y_min, y_max, grid)
    U = np.zeros((grid, grid), dtype=float)
    E = np.zeros((grid, grid), dtype=float)

    for ix, xv in enumerate(xs):
        th_base = theta_center.copy()
        th_base[idx_x] = xv
        for iy, yv in enumerate(ys):
            th = th_base.copy()
            th[idx_y] = yv

            y_hat, y_std = emul.predict_series(th)
            U[iy, ix] = float(np.mean(y_std))

            y_true = toy_forward(th, t)
            E[iy, ix] = float(np.sqrt(np.mean((y_hat - y_true) ** 2)))

    return xs, ys, U, E


# -------------------------
# Adaptive Metropolis in whitened space
# -------------------------
@dataclass
class WhitenTransform:
    mean: np.ndarray
    L: np.ndarray  # Cholesky factor of covariance (theta space)

    def to_z(self, theta: np.ndarray) -> np.ndarray:
        # z = L^{-1} (theta - mean)
        return np.linalg.solve(self.L, theta - self.mean)

    def to_theta(self, z: np.ndarray) -> np.ndarray:
        return self.mean + self.L @ z


def adaptive_cov_update(C: np.ndarray, x: np.ndarray, mean: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Online update for mean and covariance (numerically stable).
    n is number of samples seen so far AFTER including x.
    """
    # Welford-style
    delta = x - mean
    mean_new = mean + delta / n
    delta2 = x - mean_new
    C_new = C + np.outer(delta, delta2)
    return C_new, mean_new


def main():
    # ---------- user knobs ----------
    seed = 123
    N0 = 25
    r_pod= 10
    kernel = "matern52"
    ard = True

    sigma_obs = 0.1
    n_total = 30000

    # AM / scaling knobs
    target_accept = 0.25
    adapt_until = 10000          # adapt step and cov until this iteration
    step0 = 0.1                # initial scale in z-space (dimensionless)
    step_min, step_max = 1e-5, 3.0

    u_thresh = 0.01
    gamma_L_ratio = 1.05
    n_retrain_max = 500

    plot_every = 100
    map_every = 400
    grid = 60
    idx_x, idx_y = 0, 1
    names = ("A", "f", "tau")

    # ---------- setup ----------
    rng = set_seed(seed)
    t = make_timeline(T=500, t_end=0.05)
    
    prior_mean = np.array([0.8, 150.0, 0.010])
    prior_cov = np.diag([0.5**2, 10.0**2, 0.01**2])
    prior = GaussianPrior(prior_mean, prior_cov)

    theta_true = sample_theta_from_prior(rng, prior_mean, prior_cov, tau_min=1e-6)
    y_obs = make_observation(rng, theta_true, t, sigma_obs)

    # initial emulator design
    X0 = make_design_gaussian(rng, prior_mean, prior_cov, N0)
    Y0 = np.array([toy_forward(X0[i], t) for i in range(N0)])

    pod = POD(r=r_pod).fit(Y0)
    A0 = pod.project(Y0)
    gps = [GPSurrogate(X0, A0[:, k], kernel=kernel, ard=ard) for k in range(r_pod)]
    emul = PODGPEmulator(pod=pod, gps=gps)

    # ---- whitening transform based on prior covariance ----
    L_prior = np.linalg.cholesky(prior_cov + 1e-12 * np.eye(3))
    W = WhitenTransform(mean=prior_mean, L=L_prior)

    # initialise chain at prior mean (or you can plug LS estimate here)
# ---- LS initialization + proposal covariance from LS ----
    # theta_ls, cov_ls = gauss_newton_ls(
    #     theta0=prior_mean,      # starting guess
    #     t=t,
    #     y_obs=y_obs,
    #     sigma_obs=sigma_obs,
    #     tau_min=1e-6,
    #     n_iter=30,
    #     lam0=1e-3,
    # )

    # L_ls = np.linalg.cholesky(cov_ls + 1e-12 * np.eye(3))
    # W = WhitenTransform(mean=theta_ls, L=L_ls)

    # theta0 = theta_ls
    # z0 = W.to_z(theta0)   # == 0 (up to numerical noise)

    # # logpost at start (TRUE forward so it's consistent)
    # y0 = toy_forward(theta0, t)
    # ll0 = loglike_gaussian_iid(y0, y_obs, sigma_obs)
    # lp0 = prior.logpdf(theta0)
    # logpost_old = ll0 + lp0
    
    # ---- whitening transform based on prior covariance ----
    L_prior = np.linalg.cholesky(prior_cov + 1e-12 * np.eye(3))
    W = WhitenTransform(mean=prior_mean, L=L_prior)

    # ---- initialise chain (choose one) ----
    theta0 = prior_mean.copy()  # simplest
    # theta0 = sample_theta_from_prior(rng, prior_mean, prior_cov, tau_min=1e-6)  # alternative

    z0 = W.to_z(theta0)

    # logpost at start (TRUE forward so it's consistent)
    y0 = toy_forward(theta0, t)
    ll0 = loglike_gaussian_iid(y0, y_obs, sigma_obs)
    lp0 = prior.logpdf(theta0)
    logpost_old = ll0 + lp0

    # ---------- live plotting ----------
    plt.ion()
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 3)

    ax_ts = fig.add_subplot(gs[0, :2])
    ax_ll = fig.add_subplot(gs[1, 0])
    ax_acc = fig.add_subplot(gs[1, 1])
    ax_sc = fig.add_subplot(gs[2, 0])
    ax_U = fig.add_subplot(gs[0, 2])
    ax_E = fig.add_subplot(gs[1, 2])
    ax_dlp = fig.add_subplot(gs[2, 1])
    ax_u = fig.add_subplot(gs[2, 2])

    fig.suptitle("Live Adaptive-Metropolis + Active-Learning (Toy + POD+GP)", fontsize=12)

    Xref = make_design_gaussian(rng, prior_mean, prior_cov, 400)
    x_min, x_max, y_min, y_max = grid_2d_bounds(Xref, idx_x, idx_y, 1, 99)

    _, _, U, E = eval_maps_on_slice(
        emul=emul, t=t, theta_center=prior_mean.copy(),
        idx_x=idx_x, idx_y=idx_y,
        x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
        grid=grid
    )
    imU = ax_U.imshow(
        U, origin="lower", aspect="auto",
        extent=[x_min, x_max, y_min, y_max],
        norm=LogNorm(vmin=max(U.min(), 1e-12), vmax=max(U.max(), 1e-12))
    )    
    # --- mark theta_true on the 2D slice (A vs f) ---
    x_true = float(theta_true[idx_x])
    y_true = float(theta_true[idx_y])

    ptU = ax_U.scatter(x_true, y_true, s=140, marker="*", c="k", edgecolors="w", linewidths=1.2, zorder=5)
    ax_U.set_title("Uncertainty map: mean_t std_y")
    ax_U.set_xlabel(names[idx_x]); ax_U.set_ylabel(names[idx_y])
    fig.colorbar(imU, ax=ax_U, fraction=0.046, pad=0.04)

    imE = ax_E.imshow(
        E, origin="lower", aspect="auto",
        extent=[x_min, x_max, y_min, y_max],
        norm=LogNorm(vmin=max(E.min(), 1e-12), vmax=max(E.max(), 1e-12))
    )
    ptE = ax_E.scatter(x_true, y_true, s=140, marker="*", c="k", edgecolors="w", linewidths=1.2, zorder=5)
    ax_E.set_title("True error map: RMSE(y_hat, y_true)")
    ax_E.set_xlabel(names[idx_x]); ax_E.set_ylabel(names[idx_y])
    fig.colorbar(imE, ax=ax_E, fraction=0.046, pad=0.04)

    # storage
    chain = np.zeros((n_total + 1, 3))
    chain[0] = theta0
    used_forward = np.zeros(n_total, dtype=bool)
    accepted = np.zeros(n_total, dtype=bool)

    # histories
    ll_true_hist, ll_sur_hist = [], []
    rmse_true_hist, u_true_hist = [], []
    acc_hist, fw_hist = [], []
    dlp_hist, uprop_hist = [], []
    step_hist = []
    reject_prior = 0
    reject_constraint = 0

    # AM state in z-space
    d = 3
    step = float(step0)
    z_chain = np.zeros((n_total + 1, d))
    z_chain[0] = z0

    # Running mean/cov accumulator for z (for adaptation)
    z_mean = z0.copy()
    C = np.eye(d)   # start with cov_z ≈ I
    eps = 1e-6

    def update_true_theta_panels():
        y_true = toy_forward(theta_true, t)
        y_hat, y_std = emul.predict_series(theta_true)

        ll_true = loglike_gaussian_iid(y_true, y_obs, sigma_obs)
        ll_sur = loglike_gaussian_iid(y_hat, y_obs, sigma_obs)

        rmse_t = float(np.sqrt(np.mean((y_hat - y_true) ** 2)))
        ubar_t = float(np.mean(y_std))

        ll_true_hist.append(ll_true)
        ll_sur_hist.append(ll_sur)
        rmse_true_hist.append(rmse_t)
        u_true_hist.append(ubar_t)

        ax_ts.clear()
        ax_ts.plot(t, y_true, label="true forward(θ_true)")
        ax_ts.plot(t, y_hat, label="surrogate mean(θ_true)")
        ax_ts.fill_between(t, y_hat - 2*y_std, y_hat + 2*y_std, alpha=0.2, label="±2 std (surrogate)")
        ax_ts.plot(t, y_obs, alpha=0.6, label="obs (noisy)")
        ax_ts.set_title(f"At θ_true: RMSE(true vs surrogate)={rmse_t:.3e}, mean_std={ubar_t:.3e}")
        ax_ts.grid(True); ax_ts.legend(loc="best")

        ax_ll.clear()
        it = np.arange(len(ll_true_hist))
        ax_ll.plot(it, ll_true_hist, label="loglike true(θ_true)")
        ax_ll.plot(it, ll_sur_hist, label="loglike surrogate(θ_true)")
        ax_ll.set_title("Loglike at θ_true (true vs surrogate)")
        ax_ll.grid(True); ax_ll.legend(loc="best")

    update_true_theta_panels()

    for n in range(n_total):
        theta_n = chain[n].copy()
        z_n = z_chain[n].copy()

        # --- adaptive covariance in z-space ---
        if n < adapt_until and n >= 10:
            cov_z = (C / max(n, 1)) + eps * np.eye(d)
        else:
            cov_z = (C / max(min(adapt_until, n), 1)) + eps * np.eye(d)

        # Haario scaling suggestion
        scale_mat = (2.38**2 / d) * cov_z

        # propose in z, then map to theta
        z_star = z_n + step * rng.multivariate_normal(np.zeros(d), scale_mat)
        theta_star = W.to_theta(z_star)

        # constraints
        if theta_star[2] <= 1e-6:
            reject_constraint += 1
            chain[n + 1] = theta_n
            z_chain[n + 1] = z_n
            continue

        lp_star = prior.logpdf(theta_star)
        if np.isneginf(lp_star):
            reject_prior += 1
            chain[n + 1] = theta_n
            z_chain[n + 1] = z_n
            continue

        # --- surrogate likelihood at current state (under current GP) ---
        y_hat_n, y_std_n = emul.predict_series(theta_n)
        ll_old_hat = loglike_gaussian_hetero(y_hat_n, y_obs, sigma_obs, y_std_n)
        lp_old = prior.logpdf(theta_n)
        logpost_old_hat = ll_old_hat + lp_old

        # --- surrogate prediction at proposal ---
        y_hat_star, y_std_star = emul.predict_series(theta_star)
        ubar = float(np.mean(y_std_star))
        uprop_hist.append(ubar)

        # --- active-learning gate ---
        if ubar > u_thresh:
            # truth call (used only for retraining / diagnostics)
            y_true = toy_forward(theta_star, t)
            # ll_true_star = loglike_gaussian_iid(y_true, y_obs, sigma_obs)  # optional diagnostic

            emul.update_from_true_series(
                theta_star, y_true,
                gamma_L_ratio=gamma_L_ratio,
                n_retrain_max=n_retrain_max
            )
            used_forward[n] = True

            # GP changed -> recompute surrogate predictions for BOTH old and new
            y_hat_n, y_std_n = emul.predict_series(theta_n)
            ll_old_hat = loglike_gaussian_hetero(y_hat_n, y_obs, sigma_obs, y_std_n)
            logpost_old_hat = ll_old_hat + lp_old  # lp_old unchanged

            y_hat_star, y_std_star = emul.predict_series(theta_star)
            ll_star_hat = loglike_gaussian_hetero(y_hat_star, y_obs, sigma_obs, y_std_star)
        else:
            ll_star_hat = loglike_gaussian_hetero(y_hat_star, y_obs, sigma_obs, y_std_star)

        # --- MH on surrogate-consistent target ---
        logpost_star_hat = ll_star_hat + lp_star
        dlp = float(logpost_star_hat - logpost_old_hat)
        dlp_hist.append(dlp)

        is_acc = mh_accept_log(rng, logpost_star_hat, logpost_old_hat)
        accepted[n] = is_acc

        if is_acc:
            chain[n + 1] = theta_star
            z_chain[n + 1] = z_star
            # (no persistent logpost_old needed; we recompute ll_old_hat each iter)

            if n < adapt_until:
                C, z_mean = adaptive_cov_update(C, z_star, z_mean, n + 1)
        else:
            chain[n + 1] = theta_n
            z_chain[n + 1] = z_n

            if n < adapt_until:
                C, z_mean = adaptive_cov_update(C, z_n, z_mean, n + 1)

        # adapt step to hit target acceptance (Robbins–Monro)
        if n < adapt_until and (n + 1) % 25 == 0:
            acc_rate_local = float(np.mean(accepted[max(0, n - 500): n + 1]))
            step *= np.exp(0.05 * (acc_rate_local - target_accept))
            step = float(np.clip(step, step_min, step_max))

        step_hist.append(step)

        # running stats
        acc_rate = float(np.mean(accepted[: n + 1]))
        fw_rate = float(np.mean(used_forward[: n + 1]))
        acc_hist.append(acc_rate)
        fw_hist.append(fw_rate)

        if (n + 1) % plot_every == 0:
            update_true_theta_panels()

            ax_acc.clear()
            ax_acc.plot(acc_hist, label="accept rate")
            ax_acc.plot(fw_hist, label="forward-call fraction")
            ax_acc.plot(step_hist, label="step(z-space)")
            ax_acc.set_ylim(0.0, 1.2)
            ax_acc.set_title("Running acc/fwd + proposal step")
            ax_acc.grid(True); ax_acc.legend(loc="best")

            ax_sc.clear()
            th = chain[: n + 2]
            ax_sc.scatter(th[:, idx_x], th[:, idx_y], s=10, alpha=0.25, label="chain")
            fw_idx = np.where(used_forward[: n + 1])[0]
            if fw_idx.size > 0:
                th_fw = chain[fw_idx + 1]
                ax_sc.scatter(th_fw[:, idx_x], th_fw[:, idx_y], s=55, marker="x", alpha=0.9, label="active forward")
            ax_sc.scatter(theta_true[idx_x], theta_true[idx_y], s=120, marker="*", label="θ_true")
            ax_sc.set_xlabel(names[idx_x]); ax_sc.set_ylabel(names[idx_y])
            ax_sc.set_title("Sampling in parameter space (A vs f)")
            ax_sc.grid(True); ax_sc.legend(loc="best")

            ax_dlp.clear()
            abs_dlp = np.abs(dlp_hist) + 1e-12
            ax_dlp.plot(abs_dlp, alpha=0.9)
            ax_dlp.set_yscale("log")
            ax_dlp.set_title("|Δ log-posterior| (log scale)")
            ax_dlp.grid(True)

            ax_u.clear()
            ax_u.plot(uprop_hist, alpha=0.9)
            ax_u.axhline(u_thresh, linestyle="--", label="u_thresh")
            ax_u.set_yscale("log")
            ax_u.set_title("mean_t std_y at proposals (gate)")
            ax_u.grid(True); ax_u.legend(loc="best")

            fig.canvas.draw()
            fig.canvas.flush_events()

        if (n + 1) % map_every == 0:
            _, _, U, E = eval_maps_on_slice(
                emul=emul, t=t, theta_center=prior_mean.copy(),
                idx_x=idx_x, idx_y=idx_y,
                x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
                grid=grid
            )
            imU.set_data(U)
            imU.set_norm(LogNorm(vmin=max(U.min(), 1e-12), vmax=max(U.max(), 1e-12)))

            imE.set_data(E)
            imE.set_norm(LogNorm(vmin=max(E.min(), 1e-12), vmax=max(E.max(), 1e-12)))
            ax_U.set_title(f"Uncertainty map (updated @ iter {n+1})")
            ax_E.set_title(f"True error map (updated @ iter {n+1})")
            fig.canvas.draw()
            fig.canvas.flush_events()

        if (n + 1) % (5 * plot_every) == 0:
            dlp_arr = np.array(dlp_hist[-5 * plot_every:])
            print(
                f"[{n+1:6d}] acc={acc_rate:.3f} fw={fw_rate:.3f} step={step:.3g} "
                f"rej_prior={reject_prior} rej_constr={reject_constraint} "
                f"dlp(mean/med/min)={dlp_arr.mean():.2f}/{np.median(dlp_arr):.2f}/{dlp_arr.min():.2f} "
                f"u(mean)={np.mean(uprop_hist[-5*plot_every:]):.3e}"
            )

    plt.ioff()
    plt.show()

    print("\nFinal acceptance:", float(np.mean(accepted)))
    print("Final forward-call fraction:", float(np.mean(used_forward)))
    print("Rejected by prior:", reject_prior, "Rejected by constraint:", reject_constraint)


if __name__ == "__main__":
    main()
