# from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

from gp_active_mcmc.toy import toy_forward, make_timeline, make_observation
from gp_active_mcmc.pod import POD
from gp_active_mcmc.gp import GPSurrogate
from gp_active_mcmc.podgp import PODGPSurrogate
from gp_active_mcmc.priors import GaussianPrior
from gp_active_mcmc.proposals import AdaptiveRWMProposal
from gp_active_mcmc.sampler import ALMCMC, RALMCMC, ARALMCMC
from gp_active_mcmc.likelihood import loglike_theta_gp, loglike_theta
from utils import plot_prediction_at_theta, plot_chain_2d

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
SEED = 123
N_TOTAL = 5000  # quick test
N_BURNIN = 2000  # quick test
N_INIT = 25
POD_RANK = 10
GP_KERNEL = "matern52"
USE_ARD = True
SIGMA_OBS = 0.01
GAMMA_VAR = 0.01
GAMMA_L_RATIO = 1.05
N_RETRAIN_MAX = 50

rng = np.random.default_rng(SEED)
t = make_timeline(T=500, t_end=0.05)

prior_mean = np.array([0.8, 150.0, 0.010])
prior_cov = np.diag([0.5**2, 10.0**2, 0.01**2])
prior = GaussianPrior(prior_mean, prior_cov)
proposal = AdaptiveRWMProposal(cov=prior_cov)

theta_true = prior.sample(rng)
y_obs = make_observation(rng, theta_true, t, SIGMA_OBS)
sigma_obs = SIGMA_OBS * np.ones_like(y_obs)
fw = lambda theta: toy_forward(theta, t)

def positive_tau(theta: np.ndarray) -> bool:
    return theta[2] > 1e-6

# --------------------------------------------------------------
# Build initial surrogate
# --------------------------------------------------------------
X0 = np.array([prior.sample(rng) for _ in range(N_INIT)])
Y0 = np.array([fw(X0[i]) for i in range(N_INIT)])

pod = POD(r=POD_RANK).fit(Y0)
A0 = pod.project(Y0)

gps = [
    GPSurrogate(
        X0, A0[:, k],
        kernel=GP_KERNEL, ard=USE_ARD,
        gamma_L_ratio=GAMMA_L_RATIO,
        n_retrain_max=N_RETRAIN_MAX
    ) for k in range(POD_RANK)
]

emul = PODGPSurrogate(pod=pod, gps=gps)
theta0 = prior_mean.copy()

loglike = lambda theta: loglike_theta(theta, fw, y_obs, sigma_obs)
loglike_surrogate = lambda theta: loglike_theta_gp(theta, emul, y_obs, sigma_obs)

# --------------------------------------------------------------
# Define methods
# --------------------------------------------------------------
methods = {
    "ALMCMC": ALMCMC(emul, fw, gamma_var=GAMMA_VAR, loglike_surrogate=loglike_surrogate, prior=prior, proposal=proposal, log_theta_ref=theta_true),
    "RALMCMC": RALMCMC(emul.copy(), fw, gamma_var=GAMMA_VAR, loglike=loglike,loglike_surrogate=loglike_surrogate, prior=prior, proposal=proposal.copy(), log_theta_ref=theta_true, subsample_rate=0.25),
    "ARALMCMC": ARALMCMC(emul.copy(), fw, gamma_var=GAMMA_VAR, loglike=loglike,loglike_surrogate=loglike_surrogate, prior=prior, proposal=proposal.copy(), log_theta_ref=theta_true, subsample_rate=0.25),
}

# --------------------------------------------------------------
# Run samplers
# --------------------------------------------------------------
results = {}
for name, sampler in methods.items():
    print(f"\nRunning {name}...")
    results[name] = sampler.run(theta0=theta0, n_total=N_TOTAL, store_gp_ref=True, n_gp_update=N_BURNIN)

# --------------------------------------------------------------
# Analysis
# --------------------------------------------------------------
for name, res in results.items():
    chain = res["chain"]
    used_forward = res["used_forward"]
    accept_rate = res["accept_rate"]
    forward_frac = np.mean(used_forward)
    rmse = np.sqrt(np.mean((chain[-1] - theta_true)**2))
    
    print(f"\n{name} summary:")
    print(f"  Acceptance rate      : {accept_rate:.3f}")
    print(f"  Forward-call fraction: {forward_frac:.3f}")
    print(f"  RMSE vs theta_true   : {rmse:.5f}")
    
    # Surrogate prediction consistency
    if "gp_pred_ref" in res:
        y_preds, y_vars = zip(*res["gp_pred_ref"])
        y_preds = np.array(y_preds)
        mse_gp = np.mean((y_preds - fw(theta_true))**2)
        print(f"  GP MSE at theta_ref  : {mse_gp:.3e}")
    
    # Chain plots
    plot_chain_2d(
        chain=chain,
        used_forward=used_forward,
        theta_true=theta_true,
        names=("A", "f"),
        title=f"{name} chain"
    )
    
    # Surrogate prediction plot
    plot_prediction_at_theta(
        emul=emul,
        theta=theta_true,
        t=t,
        y_obs=y_obs,
        title=f"{name} surrogate prediction"
    )

# --------------------------------------------------------------
# Optional: Forward-call histogram over iterations
# --------------------------------------------------------------
plt.figure(figsize=(8,4))
for name, res in results.items():
    plt.plot(np.cumsum(res["used_forward"])/np.arange(1,N_TOTAL+1), label=name)
plt.xlabel("Iteration")
plt.ylabel("Cumulative Forward-call fraction")
plt.title("Forward-call fraction over iterations")
plt.legend()
plt.show()
