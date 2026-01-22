from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

from gp_active_mcmc.utils import set_seed
from gp_active_mcmc.toy import toy_forward, make_timeline, make_observation
from gp_active_mcmc.pod import POD
from gp_active_mcmc.gp_surrogate import GPSurrogate
from gp_active_mcmc.podgp_surrogate import PODGPSurrogate
from gp_active_mcmc.priors import GaussianPrior
from gp_active_mcmc.likelihood import loglike_theta_gp, loglike_theta
from gp_active_mcmc.algorithm1 import run_algorithm1_rwm
from matplotlib.colors import LogNorm


# -----------------from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from dataclasses import dataclass
from tqdm import tqdm

#SOME to REFACTOR

def make_design_gaussian(rng: np.random.Generator, mean: np.ndarray, cov: np.ndarray, n: int, tau_min=1e-6):
    X = rng.multivariate_normal(mean=mean, cov=cov, size=n)
    bad = X[:, 2] <= tau_min
    while np.any(bad):
        X[bad] = rng.multivariate_normal(mean=mean, cov=cov, size=int(np.sum(bad)))
        bad = X[:, 2] <= tau_min
    return X

import matplotlib.pyplot as plt

def plot_final_chain(chain: np.ndarray, used_forward: np.ndarray, theta_true: np.ndarray, names=("A","f","tau")):
    """
    Plot the final chain in 2D parameter space (first 2 params),
    marking GP vs forward-evaluated samples.
    """
    fig, ax = plt.subplots(figsize=(6,5))

    # GP samples
    gp_idx = np.where(~used_forward)[0] + 1  # +1 because used_forward corresponds to n_total
    if gp_idx.size > 0:
        ax.scatter(chain[gp_idx, 0], chain[gp_idx, 1], s=20, alpha=0.5, c='blue', label='GP')

    # True forward samples
    fw_idx = np.where(used_forward)[0] + 1
    if fw_idx.size > 0:
        ax.scatter(chain[fw_idx, 0], chain[fw_idx, 1], s=50, marker='x', c='red', label='Forward')

    # True theta
    ax.scatter(theta_true[0], theta_true[1], s=150, marker='*', c='k', edgecolors='w', label='θ_true')

    ax.set_xlabel(names[0])
    ax.set_ylabel(names[1])
    ax.set_title("Final chain in θ1 vs θ2 space")
    ax.grid(True)
    ax.legend()
    plt.show()
    
# ---------- user knobs ----------
seed = 1
N0 = 25
r_pod = 10
kernel = "matern52"
ard = True

sigma_obs = 0.1
n_total = 5000   # smaller for testing / demo

gamma_var = 0.01
gamma_L_ratio = 1.05
n_retrain_max = 500
step_scale = 0.1  # initial proposal scale in theta space

idx_x, idx_y = 0, 1
names = ("A", "f", "tau")

# ---------- setup ----------
rng = np.random.default_rng(seed)
t = make_timeline(T=500, t_end=0.05)

prior_mean = np.array([0.8, 150.0, 0.010])
prior_cov = np.diag([0.5**2, 10.0**2, 0.01**2])
prior = GaussianPrior(prior_mean, prior_cov)

theta_true = prior.sample(rng)
y_obs = make_observation(rng, theta_true, t, sigma_obs)
sigma_obs = sigma_obs*np.ones_like(y_obs)

# initial emulator design
X0 = make_design_gaussian(rng, prior_mean, prior_cov, N0)
Y0 = np.array([toy_forward(X0[i], t) for i in range(N0)])

pod = POD(r=r_pod).fit(Y0)
A0 = pod.project(Y0)
gps = [GPSurrogate(X0, A0[:, k], kernel=kernel, ard=ard) for k in range(r_pod)]
emul = PODGPSurrogate(pod=pod, gps=gps)

# ---- initialise chain ----
theta0 = prior_mean.copy()

# -------------------------
# Run RWM + GP active learning
# -------------------------
result = run_algorithm1_rwm(
    rng=rng,
    theta0=theta0,
    cov=prior_cov,
    n_total=n_total,
    gamma_var=gamma_var,
    gamma_L_ratio=gamma_L_ratio,
    n_retrain_max=n_retrain_max,
    step_scale=step_scale,
    gp=emul,
    fw_true=lambda th: toy_forward(th, t),
    y_obs=y_obs,
    sigma_obs=sigma_obs,
    prior=prior,
    constraint_fn=lambda th: th[2] > 1e-6,
    verbose=True,
    print_every=500
)

chain = result["chain"]
used_forward = result["used_forward"]
accepted = result["accepted"]

# -------------------------
# Final plot
# -------------------------
y_hat, y_std = np.array([emul.predict(th)[0] for th in chain]), \
                np.array([emul.predict(th)[1] for th in chain])

rmse_hist = np.sqrt(np.mean((y_hat - toy_forward(theta_true, t))**2, axis=1))
mean_std_hist = np.mean(y_std, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(rmse_hist, label="RMSE vs θ_true")
plt.plot(mean_std_hist, label="Mean surrogate std")
plt.xlabel("Iteration")
plt.ylabel("Error / Uncertainty")
plt.title("Final POD+GP RWM run")
plt.legend()
plt.grid(True)
plt.show()

print(f"Final acceptance rate: {result['accept_rate']:.3f}")
print(f"Forward-call fraction: {np.mean(used_forward):.3f}")

plot_final_chain(result["chain"], result["used_forward"], theta_true)
