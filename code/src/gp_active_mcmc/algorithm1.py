from __future__ import annotations
import numpy as np
from .proposals import rwm_proposal


def mh_accept_prob(logpost_new: float, logpost_old: float) -> float:
    if np.isneginf(logpost_new):
        return 0.0
    a = logpost_new - logpost_old
    return float(min(1.0, np.exp(a)))


def run_algorithm1_rwm(
    rng: np.random.Generator,
    theta0: np.ndarray,
    cov: np.ndarray,
    n_total: int,
    gamma_var: float,
    gamma_L_ratio: float,
    n_retrain_max: int,
    step_scale: float,
    gp,
    loglike_true_fn,
    prior=None,
    constraint_fn=None,   # optional hard constraint (physics)
) -> dict:

    if prior is None:
        class _Flat:
            def logpdf(self, theta): return 0.0
        prior = _Flat()

    if constraint_fn is None:
        constraint_fn = lambda th: True

    d = theta0.size
    chain = np.zeros((n_total + 1, d), dtype=float)
    used_forward = np.zeros(n_total, dtype=bool)
    gp_var_hist = np.zeros(n_total, dtype=float)
    acc = 0

    chain[0] = theta0

    lp0 = prior.logpdf(theta0)
    if np.isneginf(lp0) or (not constraint_fn(theta0)):
        raise ValueError("theta0 is outside prior support or violates constraints.")

    logpost_old = loglike_true_fn(theta0) + lp0

    for n in range(n_total):
        theta_n = chain[n].copy()
        theta_star = rwm_proposal(rng, theta_n, cov, step_scale)

        if not constraint_fn(theta_star):
            chain[n + 1] = theta_n
            continue

        lp_star = prior.logpdf(theta_star)
        if np.isneginf(lp_star):
            chain[n + 1] = theta_n
            continue

        mu_star, var_star = gp.predict_loglike(theta_star)
        gp_var_hist[n] = var_star

        if var_star < gamma_var:
            loglike_star = mu_star
        else:
            loglike_star = loglike_true_fn(theta_star)
            gp.update(theta_star, loglike_star, gamma_L_ratio, n_retrain_max)
            used_forward[n] = True

        logpost_star = loglike_star + lp_star

        alpha = mh_accept_prob(logpost_star, logpost_old)
        if rng.uniform(0.0, 1.0) < alpha:
            chain[n + 1] = theta_star
            logpost_old = logpost_star
            acc += 1
        else:
            chain[n + 1] = theta_n

    return {
        "chain": chain,
        "accept_rate": acc / n_total,
        "used_forward": used_forward,
        "gp_var": gp_var_hist,
    }
