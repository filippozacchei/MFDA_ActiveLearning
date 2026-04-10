# %% [markdown]
# # Forward tritium: build and validate a POD-GP surrogate
#
# Same workflow as ``run_forward_toy.py``, applied to the Achlys tritium diffusion
# benchmark.  The HF model runs via UM-Bridge (Docker container).
#
# **Before running**, start the Docker server:
#
# ```bash
# docker run -it -p 4243:4243 linusseelinger/benchmark-achlys:latest
# ```
#
# We will:
#
# 1. sample parameters from the uniform prior,
# 2. evaluate the HF forward model (tritium flux vs. time),
# 3. fit a POD basis and a multi-output GP on POD coefficients,
# 4. evaluate prediction error and interval coverage on a held-out test set,
# 5. visualise representative predictions.

# %% Imports
from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split

from tritium import (
    make_forward_model,
    make_time_grid,
    sample_prior,
    PRIOR_BOUNDS,
    N_OUTPUT,
)
from gp_active_mcmc.diagnostics.pod import plot_pod_energy
from gp_active_mcmc.diagnostics.surrogate import (
    plot_error_vs_uncertainty,
    plot_prediction_at_theta,
)
from gp_active_mcmc.surrogates import POD, MultiOutputGP, PODGPSurrogate
from gp_active_mcmc.utils.metrics import coverage, rmse
from gp_active_mcmc.utils.rng import set_seed


# %% [markdown]
# ## Configuration

# %%
rng = set_seed(7)

# Time grid (for plotting only -- the server returns 500 output values)
t = make_time_grid(n_pts=N_OUTPUT)

# HF forward model via UM-Bridge
hf_forward = make_forward_model(url="http://localhost:4243", model_name="forward")

# Snapshot budget
N_SNAPSHOTS = 30
TEST_FRACTION = 0.25

# Surrogate config
POD_RANK = 20
GP_KERNEL = "matern52"
USE_ARD = True


# %% [markdown]
# ## Generate a snapshot dataset
#
# Draw parameters from the uniform prior and evaluate the HF model.
# Results are cached to ``snapshots.npz`` so the expensive Docker calls
# are only needed once.

# %%
import os
import time

CACHE_FILE = os.path.join(os.path.dirname(__file__) or ".", "snapshots.npz")

if os.path.exists(CACHE_FILE):
    print(f"Loading cached snapshots from {CACHE_FILE}")
    _data = np.load(CACHE_FILE)
    X, Y = _data["X"], _data["Y"]
else:
    X = sample_prior(rng, n=N_SNAPSHOTS)
    Y_list = []
    for i, theta in enumerate(X):
        t0 = time.perf_counter()
        print(f"  Evaluating snapshot {i+1}/{N_SNAPSHOTS} ...", end=" ", flush=True)
        y = hf_forward(theta)
        Y_list.append(y)
        dt = time.perf_counter() - t0
        print(f"done in {dt:.1f}s (max|y|={np.max(np.abs(y)):.3e})")
    Y = np.asarray(Y_list, dtype=float)
    np.savez(CACHE_FILE, X=X, Y=Y)
    print(f"Saved snapshots to {CACHE_FILE}")

print(f"Snapshot matrix shape: {Y.shape}")   # (N_SNAPSHOTS, 500)


# %% [markdown]
# ## Train-test split

# %%
X_tr, X_te, Y_tr, Y_te = train_test_split(X, Y, test_size=TEST_FRACTION, random_state=0)


# %% [markdown]
# ## Build a POD-GP surrogate

# %%
pod = POD(rank=POD_RANK).fit(Y_tr)
A_tr = pod.transform(Y_tr)[:, :POD_RANK]

gp = MultiOutputGP(
    X_train=X_tr,
    Y_train=A_tr,
    kernel=GP_KERNEL,
    ard=USE_ARD,
)

surrogate = PODGPSurrogate(pod=pod, gp=gp)


# %% [markdown]
# ## POD energy plot

# %%
plot_pod_energy(
    Y_tr,
    r_max=min(50, Y_tr.shape[0]),
    center=True,
    thresholds=(0.90, 0.95, 0.99),
    show=True,
)


# %% [markdown]
# ## Predictive accuracy on the test set

# %%
z50, z90, z95 = 0.67449, 1.64485, 1.95996

test_rmse: list[float] = []
mean_pred_std: list[float] = []
cov50: list[float] = []
cov90: list[float] = []
cov95: list[float] = []

for theta, y_true in zip(X_te, Y_te, strict=True):
    y_hat, y_var = surrogate.predict(theta)
    y_std = np.sqrt(y_var)

    test_rmse.append(rmse(y_hat, y_true))
    mean_pred_std.append(float(np.mean(y_std)))

    cov50.append(coverage(y_true, y_hat, y_std, z=z50))
    cov90.append(coverage(y_true, y_hat, y_std, z=z90))
    cov95.append(coverage(y_true, y_hat, y_std, z=z95))

metrics = {
    "POD rank": POD_RANK,
    "RMSE (mean)": float(np.mean(test_rmse)),
    "RMSE (median)": float(np.median(test_rmse)),
    "Mean predictive std": float(np.mean(mean_pred_std)),
    "Coverage 50%": float(np.mean(cov50)),
    "Coverage 90%": float(np.mean(cov90)),
    "Coverage 95%": float(np.mean(cov95)),
}
print(metrics)


# %% [markdown]
# ## Visual inspection at representative parameters

# %%
rmse_arr = np.array(test_rmse)
idx_best = int(np.argmin(rmse_arr))
idx_worst = int(np.argmax(rmse_arr))
idx_median = int(np.argsort(rmse_arr)[len(rmse_arr) // 2])

for label, idx in [("Best", idx_best), ("Median", idx_median), ("Worst", idx_worst)]:
    y_hat, y_var = surrogate.predict(X_te[idx])
    plot_prediction_at_theta(
        model=surrogate,
        theta=X_te[idx],
        t=t,
        y_obs=None,
        y_true=Y_te[idx],
        title=f"Tritium surrogate — {label} test case (RMSE={rmse_arr[idx]:.2e})",
        show=True,
    )
