from __future__ import annotations

import numpy as np
from .proposals import rwm_proposal
from .likelihood import loglike_theta, loglike_gaussian, loglike_theta_gp
from tqdm import tqdm


def print_algo(
    n: int,
    accepted: np.ndarray,
    n_forward: int,
    n_gp_used: int,
    n_constr_rej: int,
    n_prior_rej: int,
) -> None:
    """Print a concise summary of the MCMC algorithm at iteration n."""
    acc_rate = accepted[: n + 1].mean()
    fw_rate = n_forward / (n + 1)
    gp_rate = n_gp_used / (n + 1)

    print(
        f"[{n + 1:6d}] acc={acc_rate:.3f}  fw={fw_rate:.3f}  gp={gp_rate:.3f}  "
        f"constr_rej={n_constr_rej}  prior_rej={n_prior_rej}"
    )


def mh_accept_prob(logpost_new: float, logpost_old: float) -> float:
    """Return Metropolis-Hastings acceptance probability."""
    if np.isneginf(logpost_new):
        return 0.0
    delta = logpost_new - logpost_old
    return float(min(1.0, np.exp(delta)))


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
    fw_true,
    y_obs: np.ndarray,
    sigma_obs: np.ndarray,
    prior=None,
    constraint_fn=None,
    verbose: bool = True,
    print_every: int = 200,
) -> dict:
    """
    Run Algorithm 1 (Random Walk Metropolis) with GP-guided active learning.

    Parameters
    ----------
    gp : object
        Must implement predict(theta) -> (mu, var) and update(theta, y_true, gamma_L_ratio, n_retrain_max).
    fw_true : callable
        True forward model.
    """

    # Default flat prior
    if prior is None:

        class _Flat:
            def logpdf(self, theta: np.ndarray) -> float:
                return 0.0

        prior = _Flat()

    # Default unconstrained space
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

    logL0 = loglike_theta(theta0, fw_true, y_obs, sigma_obs)
    logpost_old = logL0 + lp0

    # Counters
    n_forward = 0
    n_constr_rej = 0
    n_prior_rej = 0
    n_gp_used = 0

    it = tqdm(range(n_total), disable=not verbose)

    for n in it:
        theta_n = chain[n].copy()
        theta_star = rwm_proposal(rng, theta_n, cov, step_scale)

        # Constraint rejection
        if not constraint_fn(theta_star):
            chain[n + 1] = theta_n
            n_constr_rej += 1
            if verbose and print_every > 0 and (n + 1) % print_every == 0:
                print_algo(n, accepted, n_forward, n_gp_used, n_constr_rej, n_prior_rej)
            continue

        # Prior rejection
        lp_star = prior.logpdf(theta_star)
        if np.isneginf(lp_star):
            chain[n + 1] = theta_n
            n_prior_rej += 1
            if verbose and print_every > 0 and (n + 1) % print_every == 0:
                print_algo(n, accepted, n_forward, n_gp_used, n_constr_rej, n_prior_rej)
            continue

        # GP prediction
        _, var_star = gp.predict(theta_star)
        ubar = float(np.mean(var_star))
        gp_var_hist[n] = ubar

        if ubar < gamma_var:
            loglike_star = loglike_theta_gp(theta_star, gp, y_obs, sigma_obs)
            n_gp_used += 1
        else:
            y_true = fw_true(theta_star)
            gp.update(theta_star, y_true, gamma_L_ratio, n_retrain_max)
            used_forward[n] = True
            n_forward += 1

            ll_old_hat = loglike_theta_gp(theta_n, gp, y_obs, sigma_obs)
            lp_old = prior.logpdf(theta_n)
            logpost_old = ll_old_hat + lp_old

            loglike_star = loglike_theta_gp(theta_star, gp, y_obs, sigma_obs)

        logpost_star = loglike_star + float(lp_star)

        # MH acceptance
        alpha = mh_accept_prob(logpost_star, logpost_old)
        u = rng.uniform(0.0, 1.0)

        if u < alpha:
            chain[n + 1] = theta_star
            logpost_old = logpost_star
            accepted[n] = True
        else:
            chain[n + 1] = theta_n

        if verbose:
            # Update tqdm postfix
            if hasattr(it, "set_postfix"):
                it.set_postfix(
                    acc=f"{accepted[: n + 1].mean():.3f}",
                    fw=f"{n_forward / (n + 1):.3f}",
                    gp=f"{n_gp_used / (n + 1):.3f}",
                    var=f"{ubar:.2e}",
                )

            # Periodic summary print
            if print_every > 0 and (n + 1) % print_every == 0:
                print_algo(n, accepted, n_forward, n_gp_used, n_constr_rej, n_prior_rej)
                print(f"last_var={ubar:.2e}")

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
