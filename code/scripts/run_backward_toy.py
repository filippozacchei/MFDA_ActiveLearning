from __future__ import annotations

import numpy as np
from utils import *

from gp_active_mcmc.toy import toy_forward, make_timeline, make_observation
from gp_active_mcmc.pod import POD
from gp_active_mcmc.gp_surrogate import GPSurrogate
from gp_active_mcmc.podgp_surrogate import PODGPSurrogate
from gp_active_mcmc.priors import GaussianPrior
from gp_active_mcmc.algorithm1 import run_algorithm1_rwm

seed = 123
N0 = 25
r_pod = 25
kernel = "matern52"
ard = True

sigma_obs = 0.01
n_total = 5000   

gamma_var = 0.01
gamma_L_ratio = 1.05
n_retrain_max = 500
step_scale = 0.1  

idx_x, idx_y = 0, 1
names = ("A", "f", "tau")

rng = np.random.default_rng(seed)
t = make_timeline(T=500, t_end=0.05)

prior_mean = np.array([0.8, 150.0, 0.010])
prior_cov = np.diag([0.5**2, 10.0**2, 0.01**2])
prior = GaussianPrior(prior_mean, prior_cov)

theta_true = prior.sample(rng)
y_obs = make_observation(rng, theta_true, t, sigma_obs)
sigma_obs = sigma_obs*np.ones_like(y_obs)

X0 = np.array([prior.sample(rng) for i in range(N0)])
Y0 = np.array([toy_forward(X0[i], t) for i in range(N0)])

pod = POD(r=r_pod).fit(Y0)
A0 = pod.project(Y0)
gps = [GPSurrogate(X0, A0[:, k], kernel=kernel, ard=ard) for k in range(r_pod)]
emul = PODGPSurrogate(pod=pod, gps=gps)

theta0 = prior_mean.copy()

plot_prediction_at_theta(
    emul=emul,
    theta=theta_true,
    t=t,
    y_obs=y_obs,
    title="Surrogate prediction at θ_true (initial)",
    fname="prediction_begin.png",
)

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

y_hat, y_std = np.array([emul.predict(th)[0] for th in chain]), \
                np.array([emul.predict(th)[1] for th in chain])

rmse_hist = np.sqrt(np.mean((y_hat - toy_forward(theta_true, t))**2, axis=1))
mean_std_hist = np.mean(y_std, axis=1)

print(f"Final acceptance rate: {result['accept_rate']:.3f}")
print(f"Forward-call fraction: {np.mean(used_forward):.3f}")

plot_rmse_and_uncertainty(
    rmse_hist=rmse_hist,
    mean_std_hist=mean_std_hist,
    fname="rmse_uncertainty.png",
)

plot_chain_2d(
    chain=chain,
    used_forward=used_forward,
    theta_true=theta_true,
    names=("A", "f"),
    fname="final_chain_2d.png",
)
plot_prediction_at_theta(
    emul=emul,
    theta=theta_true,
    t=t,
    y_obs=y_obs,
    title="Surrogate prediction at θ_true (final)",
    fname="prediction_final.png",
)
