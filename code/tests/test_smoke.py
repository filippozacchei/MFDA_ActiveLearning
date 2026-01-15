import numpy as np
from gp_active_mcmc.algorithm1 import run_algorithm1_rwm
from gp_active_mcmc.utils import set_seed

class DummyGP:
    def predict_loglike(self, theta):
        return 0.0, 0.0  # always low variance
    def update(self, theta, logL_true, gamma_L_ratio, n_retrain_max):
        return None

def test_smoke_runs():
    rng = set_seed(0)
    theta0 = np.array([0.5, 0.5])
    cov = np.eye(2)
    low = np.array([0.0, 0.0])
    high = np.array([1.0, 1.0])

    out = run_algorithm1_rwm(
        rng=rng,
        theta0=theta0,
        cov=cov,
        bounds_low=low,
        bounds_high=high,
        n_total=10,
        gamma_var=1e-12,
        gamma_L_ratio=2.5,
        n_retrain_max=3,
        step_scale=0.1,
        gp=DummyGP(),
        loglike_true_fn=lambda th: -np.sum(th**2),
    )
    assert out["chain"].shape == (11, 2)
