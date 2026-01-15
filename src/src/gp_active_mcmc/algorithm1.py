from __future__ import annotations
import numpy as np
from .utils import in_box
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
    bounds_low: np.ndarray,
    bounds_high: np.ndarray,
    n_total: int,
    gamma_var: float,
    gamma_L_ratio: float,
    n_retrain_max: int,
    step_scale: float,
    gp,
    loglike_true_fn,
) -> dict:
    """
    gp: object with methods
        - predict_loglike(theta) -> (mu, var) in original loglike units
        - update(theta, logL_true, gamma_L_ratio, n_retrain_max)

    loglike_true_fn(theta) -> float  (forward model + likelihood)
    """

    d = theta0.size
    chain = np.zeros((n_total + 1, d), dtype=float)
    used_forward = np.zeros(n_total, dtype=bool)
    gp_var_hist = np.zeros(n_total, dtype=float)
    acc = 0

    chain[0] = theta0
    logpost_old = loglike_true_fn(theta0)  # assumes uniform prior inside bounds

    for n in range(n_total):
        theta_n = chain[n].copy()
        theta_star = rwm_proposal(rng, theta_n, cov, step_scale)

        # prior support (box)
        if not in_box(theta_star, bounds_low, bounds_high):
            chain[n + 1] = theta_n
            continue

        mu_star, var_star = gp.predict_loglike(theta_star)
        gp_var_hist[n] = var_star

        if var_star < gamma_var:
            # accept GP prediction for log-likelihood
            logpost_star = mu_star
        else:
            # evaluate forward model, update GP
            logL_star_true = loglike_true_fn(theta_star)
            gp.update(theta_star, logL_star_true, gamma_L_ratio, n_retrain_max)
            logpost_star = logL_star_true
            used_forward[n] = True

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
