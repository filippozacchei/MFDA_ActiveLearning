import numpy as np
import matplotlib.pyplot as plt

from gp_active_mcmc.config import AlgorithmConfig, BoxBounds
from gp_active_mcmc.utils import set_seed
from gp_active_mcmc.likelihood import loglike_gaussian_iid
from gp_active_mcmc.gp_surrogate import GPSurrogate
from gp_active_mcmc.algorithm1 import run_algorithm1_rwm
from gp_active_mcmc.toy import toy_forward, make_timeline, make_observation


def main() -> None:
    cfg = AlgorithmConfig(
        n_total=2000,
        gamma_var=0.02,       # variance threshold in ORIGINAL loglike units
        gamma_L_ratio=2.5,
        n_retrain_max=50,
        step_scale=0.15,
        random_seed=7,
    )
    rng = set_seed(cfg.random_seed)

    # Bounds (3D) analogous to your real problem
    # theta = [A, f, tau]
    bounds = BoxBounds(
        low=np.array([0.2, 80.0, 0.003]),
        high=np.array([1.2, 220.0, 0.020]),
    )

    t = make_timeline(T=250, t_end=0.02)
    theta_true = np.array([0.8, 150.0, 0.010])
    sigma_obs = 0.02
    y_obs = make_observation(rng, theta_true, t, sigma_obs)

    # Build initial design (cheap)
    N0 = 80
    X0 = np.column_stack([
        rng.uniform(bounds.low[i], bounds.high[i], size=N0) for i in range(3)
    ])

    def loglike_true_fn(theta: np.ndarray) -> float:
        y = toy_forward(theta, t)
        return loglike_gaussian_iid(y, y_obs, sigma_obs)

    y0 = np.array([loglike_true_fn(X0[i]) for i in range(N0)])

    gp = GPSurrogate(X0, y0)

    theta0 = X0.mean(axis=0)
    cov = np.cov(X0.T) + 1e-8 * np.eye(3)

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
    acc = out["accept_rate"]
    fe = out["used_forward"].mean()

    print("Acceptance rate:", acc)
    print("Forward eval fraction:", fe)

    # Quick diagnostics
    names = ["A", "f", "tau"]
    for j, name in enumerate(names):
        plt.figure()
        plt.plot(chain[:, j])
        plt.title(f"Trace: {name} (true={theta_true[j]:.4g})")
        plt.grid(True)
        plt.show()

        plt.figure()
        plt.hist(chain[int(0.3 * len    (chain)):, j], bins=40, density=True)
        plt.axvline(theta_true[j], linestyle="--")
        plt.title(f"Posterior (rough): {name}")
        plt.grid(True)
        plt.show()

    plt.figure()
    plt.plot(out["gp_var"])
    plt.title("GP predictive variance at proposals")
    plt.grid(True)
    plt.show()

    # Show data fit at posterior mean (rough)
    theta_hat = chain[int(0.5 * len(chain)):, :].mean(axis=0)
    y_hat = toy_forward(theta_hat, t)

    plt.figure()
    plt.plot(t, y_obs, label="obs")
    plt.plot(t, y_hat, label="forward(theta_hat)")
    plt.title("Toy forward fit")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
