# =========================
# POD–GP surrogate demo
# =========================

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from gp_active_mcmc.utils import set_seed
from gp_active_mcmc.pod import POD
from gp_active_mcmc.gp_surrogate import GPSurrogate
from gp_active_mcmc.podgp_surrogate import PODGPSurrogate
from gp_active_mcmc.priors import GaussianPrior
from gp_active_mcmc.toy import toy_forward, make_timeline

from utils import (
    rmse,
    coverage,
    plot_pair_scatter_train_test,
    plot_pod_energy_curves,
    plot_pod_reconstruction_error_vs_rank,
    plot_error_vs_uncertainty,
    plot_prediction_at_theta,
    binned_reliability,
)

rng = set_seed(7)

# --------------------------------------------------------------
# Time discretization
# --------------------------------------------------------------
t = make_timeline(T=500, t_end=0.05)

# --------------------------------------------------------------
# Prior on parameters
# --------------------------------------------------------------
theta_mean = np.array([0.8, 150.0, 0.010])
theta_cov = np.diag([0.4**2, 10.0**2, 0.001**2])
prior = GaussianPrior(theta_mean, theta_cov)

# --------------------------------------------------------------
# Dataset generation
# --------------------------------------------------------------
N_SNAPSHOTS = 100
POD_RANK = 10
GP_KERNEL = "matern52"
USE_ARD = True

X = np.array([prior.sample(rng) for _ in range(N_SNAPSHOTS)])
Y = np.array([toy_forward(X[i], t) for i in range(N_SNAPSHOTS)])

# --------------------------------------------------------------
# Train–test split
# --------------------------------------------------------------
X_tr, X_te, Y_tr, Y_te = train_test_split(
    X, Y, test_size=0.25, random_state=0
)

# --------------------------------------------------------------
# POD + GP surrogate construction
# --------------------------------------------------------------
pod = POD(r=POD_RANK).fit(Y_tr)
A_tr = pod.project(Y_tr)

gps = [
    GPSurrogate(X_tr, A_tr[:, k], kernel=GP_KERNEL, ard=USE_ARD)
    for k in range(POD_RANK)
]

emul = PODGPSurrogate(pod=pod, gps=gps)

# --------------------------------------------------------------
# POD diagnostics
# --------------------------------------------------------------
plot_pod_energy_curves(
    Y_tr,
    r_max=50,
    center=True,
    thresholds=(0.90, 0.95, 0.99),
)

plot_pod_reconstruction_error_vs_rank(
    Y_tr,
    Y_te,
    r_list=range(100),
    center=True,
)

plot_pair_scatter_train_test(
    X_tr,
    X_te,
    names=("A", "f", "tau"),
)

# --------------------------------------------------------------
# Test-set performance and calibration
# --------------------------------------------------------------
z50, z90, z95 = 0.67449, 1.64485, 1.95996

n_test = X_te.shape[0]
test_rmse = np.zeros(n_test)
mean_pred_std = np.zeros(n_test)
cov50 = np.zeros(n_test)
cov90 = np.zeros(n_test)
cov95 = np.zeros(n_test)

for i in range(n_test):
    y_hat, y_std = emul.predict(X_te[i])
    y_true = Y_te[i]

    test_rmse[i] = rmse(y_hat, y_true)
    mean_pred_std[i] = np.mean(y_std)

    cov50[i] = coverage(y_true, y_hat, y_std, z50)
    cov90[i] = coverage(y_true, y_hat, y_std, z90)
    cov95[i] = coverage(y_true, y_hat, y_std, z95)

metrics = {
    "rmse_mean": test_rmse.mean(),
    "rmse_median": np.median(test_rmse),
    "mean_uncertainty": mean_pred_std.mean(),
    "coverage_50": cov50.mean(),
    "coverage_90": cov90.mean(),
    "coverage_95": cov95.mean(),
}

print(f"POD rank r = {POD_RANK}")
for k, v in metrics.items():
    print(f"{k:>18s} : {v:.6f}")

# --------------------------------------------------------------
# Representative cases
# --------------------------------------------------------------
idx_best = int(np.argmin(test_rmse))
idx_worst = int(np.argmax(test_rmse))
idx_median = int(np.argsort(test_rmse)[len(test_rmse) // 2])

# --------------------------------------------------------------
# Coefficient-level error analysis (worst case)
# --------------------------------------------------------------
theta_worst = X_te[idx_worst]
y_worst = Y_te[idx_worst]

a_true = pod.project(y_worst.reshape(1, -1))[0]
a_mu, a_var = emul.predict_coeffs(theta_worst)

abs_err = np.abs(a_mu - a_true)
k_sorted = np.argsort(abs_err)[::-1]

print("Worst-case POD coefficient errors (top 5):")
for k in k_sorted[:5]:
    print(
        f"k={k:2d}  "
        f"|err|={abs_err[k]:.4e}  "
        f"pred_std={np.sqrt(a_var[k]):.4e}  "
        f"true={a_true[k]:.4e}  "
        f"pred={a_mu[k]:.4e}"
    )

# --------------------------------------------------------------
# Reliability diagnostics
# --------------------------------------------------------------
plot_error_vs_uncertainty(mean_pred_std, test_rmse)
binned_reliability(mean_pred_std, test_rmse, n_bins=5)

# --------------------------------------------------------------
# Trajectory reconstructions
# --------------------------------------------------------------
for label, idx in [("best", idx_best), ("median", idx_median), ("worst", idx_worst)]:
    theta = X_te[idx]
    y_true = Y_te[idx]

    plot_prediction_at_theta(emul,theta,t,y_true)


