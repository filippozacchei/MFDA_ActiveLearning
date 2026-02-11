# %% [markdown]
# # Forward toy: build and validate a POD-GP surrogate
#
# This notebook-style tutorial shows how to construct a **POD-GP surrogate** for the toy
# forward model and how to perform a minimal validation.
#
# The forward (surrogate-building) workflow is the foundation for the rest of the library:
# the low-fidelity model you build here can later be plugged into
# [`ActiveMCMCModel`][gp_active_mcmc.inference.model.ActiveMCMCModel] for active-learning MCMC.
#
# We will:
#
# 1. define a prior over parameters `theta`,
# 2. generate a dataset of trajectories `y(t)` from the forward model,
# 3. fit a POD basis (compression),
# 4. fit a multi-output GP on the POD coefficients (regression + uncertainty),
# 5. evaluate prediction error and interval coverage on a held-out test set,
# 6. visualise representative predictions.
#
# The numerical budgets are intentionally small so this tutorial runs quickly in
# documentation builds. Increase the dataset size and tune hyperparameters for real use.

# %% Imports
from __future__ import annotations

import numpy as np
from scipy.stats import multivariate_normal
from sklearn.model_selection import train_test_split

# Toy utilities.
#
# For documentation notebooks, prefer a *local* `toy.py` next to this file
# (e.g. `docs/tutorials/toy.py`) and import from there:
from toy_tutorial import make_timeline, toy_forward

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
#
# - `t` is the trajectory grid in observation space.
# - We generate `N_SNAPSHOTS` forward-model evaluations from the prior.
# - We keep `TEST_FRACTION` of them for validation.
# - POD rank controls the compression level.
# - GP kernel and ARD control the regression model used for POD coefficients.

# %%
rng = set_seed(7)

t = make_timeline(T=500, t_end=0.05)

N_SNAPSHOTS = 200
TEST_FRACTION = 0.25

POD_RANK = 20
GP_KERNEL = "matern52"
USE_ARD = True

# %% [markdown]
# ## Prior on parameters
#
# The toy problem has three parameters. We use a multivariate Normal prior:
#
# - `prior.rvs(...)` is used to sample parameter vectors,
# - in inverse problems the same prior can be reused inside `tinyDA.Posterior`.

# %%
prior_mean = np.array([0.8, 150.0, 0.01])
prior_cov = np.diag([0.4**2, 40.0**2, 0.01**2])
prior = multivariate_normal(mean=prior_mean, cov=prior_cov)

# %% [markdown]
# ## Generate a snapshot dataset
#
# We build:
#
# - `X`: parameters, shape `(N_SNAPSHOTS, n_dim)`
# - `Y`: trajectories, shape `(N_SNAPSHOTS, n_time)`
#
# Each row of `Y` corresponds to `toy_forward(theta_i, t)`.

# %%
X = np.asarray([prior.rvs(random_state=rng) for _ in range(N_SNAPSHOTS)], dtype=float)
Y = np.asarray([toy_forward(theta, t) for theta in X], dtype=float)

# %% [markdown]
# ## Train-test split
#
# We hold out a fraction of snapshots to evaluate predictive accuracy and uncertainty.

# %%
X_tr, X_te, Y_tr, Y_te = train_test_split(X, Y, test_size=TEST_FRACTION, random_state=0)

# %% [markdown]
# ## Build a POD-GP surrogate
#
# The surrogate is built in three conceptual steps:
#
# 1. **POD fit**: learn a reduced basis from training trajectories `Y_tr`.
# 2. **Projection**: convert training trajectories to coefficients `A_tr`.
# 3. **GP fit**: learn `theta -> coefficients` with uncertainty.
#
# Finally, `PODGPSurrogate` combines POD + GP and exposes `predict(theta)` returning:
#
# - `y_mean(t)` (trajectory mean),
# - `y_var(t)` (pointwise variance), which can be converted to standard deviation.

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
# ## POD sanity checks (optional but recommended)
#
# The energy plots help you understand whether your chosen POD rank is reasonable.
# If the cumulative energy saturates early, a smaller rank may be sufficient.

# %%
plot_pod_energy(
    Y_tr,
    r_max=min(50, Y_tr.shape[0]),
    center=True,
    thresholds=(0.90, 0.95, 0.99),
    show=True,
)

# %% [markdown]
# ## Predictive accuracy and uncertainty on the test set
#
# For each test parameter:
#
# - predict `y_hat(t)` and `y_var(t)`,
# - compute RMSE against the true trajectory,
# - compute empirical coverage of predictive intervals.
#
# Coverage is evaluated for three common Gaussian interval multipliers:
# - 50%: `z ≈ 0.674`
# - 90%: `z ≈ 1.645`
# - 95%: `z ≈ 1.960`

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
#
# Numerical summaries are useful, but plots often reveal failure modes quickly.
# We visualise three cases selected by trajectory RMSE:
#
# - best (lowest error),
# - median,
# - worst (highest error).
#
# The plot shows:
# - the true trajectory,
# - the surrogate mean,
# - an uncertainty band (±`z` standard deviations; see function default).

# %%
order = np.argsort(test_rmse)
idx_best = int(order[0])
idx_median = int(order[len(order) // 2])
idx_worst = int(order[-1])

for label, idx in [("best", idx_best), ("median", idx_median), ("worst", idx_worst)]:
    theta = X_te[idx]
    y_true = Y_te[idx]
    plot_prediction_at_theta(
        surrogate,
        theta,
        t,
        y_true,  # here we pass the *true* profile as y_obs
        y_true=y_true,  # also show it explicitly as the reference curve
        title=f"Surrogate prediction ({label} case)",
        show=True,
    )

# %% [markdown]
# ## Calibration diagnostic
#
# A quick check: does larger predicted uncertainty correlate with larger observed error?
# This is not a full calibration study, but it is a useful sanity check.

# %%
plot_error_vs_uncertainty(
    np.asarray(mean_pred_std),
    np.asarray(test_rmse),
    title="Error vs predicted uncertainty (test set)",
    show=True,
)
