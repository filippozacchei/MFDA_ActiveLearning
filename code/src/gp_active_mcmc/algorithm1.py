from __future__ import annotations

import numpy as np
from .proposals import rwm_proposal
from tqdm import tqdm


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
    constraint_fn=None,          # optional hard constraint (physics)
    verbose: bool = True,        # <- default True
    print_every: int = 200,      # <- print cadence (iterations)
) -> dict:
    """
    Algorithm 1 (RWM) with GP-guided active learning.

    gp must expose:
      - predict_loglike(theta) -> (mu, var)
      - update(theta, y_true, gamma_L_ratio, n_retrain_max)

    loglike_true_fn(theta) -> float  (true forward + likelihood)
    prior.logpdf(theta) -> float
    constraint_fn(theta) -> bool
    """

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
    accepted = np.zeros(n_total, dtype=bool)

    chain[0] = theta0

    lp0 = prior.logpdf(theta0)
    if np.isneginf(lp0) or (not constraint_fn(theta0)):
        raise ValueError("theta0 is outside prior support or violates constraints.")

    logL0 = loglike_true_fn(theta0)
    logpost_old = logL0 + lp0

    n_forward = 0
    n_constr_rej = 0
    n_prior_rej = 0
    n_gp_used = 0

    it = range(n_total)
    it = tqdm(it, disable=not verbose)

    for n in it:
        theta_n = chain[n].copy()
        theta_star = rwm_proposal(rng, theta_n, cov, step_scale)

        if not constraint_fn(theta_star):
            chain[n + 1] = theta_n
            n_constr_rej += 1
            if verbose and (print_every > 0) and ((n + 1) % print_every == 0):
                acc_rate = accepted[:n+1].mean()
                fw_rate = n_forward / (n + 1)
                gp_rate = n_gp_used / (n + 1)
                print(
                    f"[{n+1:6d}] acc={acc_rate:.3f}  fw={fw_rate:.3f}  gp={gp_rate:.3f}  "
                    f"constr_rej={n_constr_rej}  prior_rej={n_prior_rej}"
                )
            continue

        lp_star = prior.logpdf(theta_star)
        if np.isneginf(lp_star):
            chain[n + 1] = theta_n
            n_prior_rej += 1
            if verbose and (print_every > 0) and ((n + 1) % print_every == 0):
                acc_rate = accepted[:n+1].mean()
                fw_rate = n_forward / (n + 1)
                gp_rate = n_gp_used / (n + 1)
                print(
                    f"[{n+1:6d}] acc={acc_rate:.3f}  fw={fw_rate:.3f}  gp={gp_rate:.3f}  "
                    f"constr_rej={n_constr_rej}  prior_rej={n_prior_rej}"
                )
            continue

        mu_star, var_star = gp.predict_loglike(theta_star)
        var_star = float(var_star)
        mu_star = float(mu_star)
        gp_var_hist[n] = var_star

        if var_star < gamma_var:
            loglike_star = mu_star
            n_gp_used += 1
        else:
            loglike_star = float(loglike_true_fn(theta_star))
            gp.update(theta_star, loglike_star, gamma_L_ratio, n_retrain_max)
            used_forward[n] = True
            n_forward += 1

        logpost_star = loglike_star + float(lp_star)

        alpha = mh_accept_prob(logpost_star, logpost_old)
        u = rng.uniform(0.0, 1.0)

        if u < alpha:
            chain[n + 1] = theta_star
            logpost_old = logpost_star
            accepted[n] = True
        else:
            chain[n + 1] = theta_n

        if verbose:
            # update tqdm bar with a compact live summary
            if hasattr(it, "set_postfix"):
                it.set_postfix(
                    acc=f"{accepted[:n+1].mean():.3f}",
                    fw=f"{n_forward/(n+1):.3f}",
                    gp=f"{n_gp_used/(n+1):.3f}",
                    var=f"{var_star:.2e}",
                )

            if (print_every > 0) and ((n + 1) % print_every == 0):
                acc_rate = accepted[:n+1].mean()
                fw_rate = n_forward / (n + 1)
                gp_rate = n_gp_used / (n + 1)
                print(
                    f"[{n+1:6d}] acc={acc_rate:.3f}  fw={fw_rate:.3f}  gp={gp_rate:.3f}  "
                    f"last_var={var_star:.2e}  "
                    f"constr_rej={n_constr_rej}  prior_rej={n_prior_rej}"
                )

    return {
        "chain": chain,
        "accept_rate": float(accepted.mean()),
        "used_forward": used_forward,
        "gp_var": gp_var_hist,
        "accepted": accepted,
        "stats": {
            "n_forward": int(n_forward),
            "n_gp_used": int(n_gp_used),
            "n_constr_rej": int(n_constr_rej),
            "n_prior_rej": int(n_prior_rej),
        },
    }
