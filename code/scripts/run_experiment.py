import numpy as np
import matplotlib.pyplot as plt

from gp_active_mcmc.config import AlgorithmConfig, BoxBounds
from gp_active_mcmc.utils import set_seed
from gp_active_mcmc.likelihood import loglike_theta
from gp_active_mcmc.gp_surrogate import GPSurrogate
from gp_active_mcmc.algorithm1 import run_algorithm1_rwm

# ---- connect your forward model here ----
from modelF import coarse_model as forward_model  # adapt import to your repo


def main() -> None:
    cfg = AlgorithmConfig(
        n_total=1000,
        gamma_var=0.04,
        gamma_L_ratio=2.5,
        n_retrain_max=50,
        step_scale=0.1,
        random_seed=42,
    )
    rng = set_seed(cfg.random_seed)

    bounds = BoxBounds(
        low=np.array([0.1, -0.5, 29.0]),
        high=np.array([0.5,  0.5, 31.0]),
    )

    # Observations
    theta_true = np.array([0.3, 0.1, 30.2])
    y_obs = forward_model(theta_true)
    sigma = 0.5

    # Initial design (replace with your own initial dataset logic)
    N0 = 50
    X0 = np.column_stack([
        rng.uniform(bounds.low[i], bounds.high[i], size=N0) for i in range(3)
    ])
    y0 = np.array([loglike_theta(X0[i], forward_model, y_obs, sigma) for i in range(N0)])

    gp = GPSurrogate(X0, y0)

    # Starting point and proposal covariance
    theta0 = X0.mean(axis=0)
    cov = np.cov(X0.T) + 1e-8 * np.eye(X0.shape[1])

    loglike_true_fn = lambda th: loglike_theta(th, forward_model, y_obs, sigma)

    out = run_algorithm1_rwm(
        rng=rng,
        theta0=theta0,
        cov=cov,
        bounds_low=bounds.low,
        bounds_high=bounds.high,
        n_total=cfg.n_total,
        gamma_var=cfg.gamma_var,
        gamma_L_ratio=cfg.gamma_L_ratio,
        n_retrain_max=cfg.n_retrain_max,
        step_scale=cfg.step_scale,
        gp=gp,
        loglike_true_fn=loglike_true_fn,
    )

    chain = out["chain"]
    print("Acceptance rate:", out["accept_rate"])
    print("Forward eval fraction:", out["used_forward"].mean())

    for j, name in enumerate(["p1", "p2", "p3"]):
        plt.figure()
        plt.plot(chain[:, j])
        plt.title(f"Trace: {name}")
        plt.grid(True)
        plt.show()

    plt.figure()
    plt.plot(out["gp_var"])
    plt.title("GP predictive variance at proposals")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
