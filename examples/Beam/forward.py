from __future__ import annotations

import numpy as np
from scipy.stats import multivariate_normal
from sklearn.model_selection import train_test_split

from beam import make_spatial_grid, make_forward_model
from gp_active_mcmc.utils.rng import set_seed


# ============================================================
# Configuration
# ============================================================

rng = set_seed(7)

x = make_spatial_grid(n_pts=31, length=1.0)
obs_idx = np.array([2, 5, 8, 11, 14, 17, 20, 23, 26, 29])

N_SNAPSHOTS = 200
TEST_FRACTION = 0.25

GP_KERNEL = "matern52"
USE_ARD = True

# ============================================================
# Prior on parameters
# ============================================================

prior_mean = np.array([10.0, 10.0, 10.0])
prior_cov = np.diag([2.0**2, 2.0**2, 2.0**2])

prior = multivariate_normal(mean=prior_mean, cov=prior_cov)

# ============================================================
# HF forward model on observed outputs
# ============================================================

hf_forward = make_forward_model(
    x=x,
    obs_idx=obs_idx,
    load_scale=-1.0,
    return_full_state=False,
)

# ============================================================
# Generate snapshot dataset
# ============================================================

X = np.asarray(
    [prior.rvs(random_state=rng) for _ in range(N_SNAPSHOTS)],
    dtype=float,
)

Y = np.asarray(
    [hf_forward(theta) for theta in X],
    dtype=float,
)


# ============================================================
# Train-test split
# ============================================================

X_tr, X_te, Y_tr, Y_te = train_test_split(
    X,
    Y,
    test_size=TEST_FRACTION,
    random_state=0,
)

print("X.shape    =", X.shape)
print("Y.shape    =", Y.shape)
print("X_tr.shape =", X_tr.shape)
print("Y_tr.shape =", Y_tr.shape)
print("X_te.shape =", X_te.shape)
print("Y_te.shape =", Y_te.shape)


# %% [markdown]
# ## Build a direct GP surrogate on observed outputs
#
# Since we already work in the reduced observation space, we skip POD.
# The surrogate learns directly:
#
#     theta -> y(theta)
#
# where y(theta) is the vector of observed beam displacements.
#
# The surrogate exposes:
#
# - `y_mean` : predictive mean
# - `y_var`  : predictive variance (componentwise)

# %%
from gp_active_mcmc.surrogates import MultiOutputGP


class DirectGPSurrogate:
    """
    Direct GP surrogate on observed outputs Y.
    """

    def __init__(self, gp):
        self.gp = gp

    def predict(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        theta = np.asarray(theta, dtype=float).reshape(-1)
        y_mean, y_var = self.gp.predict(theta)
        return np.asarray(y_mean, dtype=float), np.asarray(y_var, dtype=float)


gp = MultiOutputGP(
    X_train=X_tr,
    Y_train=Y_tr,
    kernel=GP_KERNEL,
    ard=USE_ARD,
)

surrogate = DirectGPSurrogate(gp=gp)




def rmse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

def coverage(y_true: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray, z: float = 1.96) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_mean = np.asarray(y_mean, dtype=float)
    y_std = np.asarray(y_std, dtype=float)

    lower = y_mean - z * y_std
    upper = y_mean + z * y_std

    inside = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(inside))

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
    "RMSE (mean)": float(np.mean(test_rmse)),
    "RMSE (median)": float(np.median(test_rmse)),
    "Mean predictive std": float(np.mean(mean_pred_std)),
    "Coverage 50%": float(np.mean(cov50)),
    "Coverage 90%": float(np.mean(cov90)),
    "Coverage 95%": float(np.mean(cov95)),
}
print(metrics)


import matplotlib.pyplot as plt

order = np.argsort(test_rmse)
idx_best = int(order[0])
idx_median = int(order[len(order) // 2])
idx_worst = int(order[-1])

x_obs = x[obs_idx]
z_plot = 1.96

for label, idx in [("best", idx_best), ("median", idx_median), ("worst", idx_worst)]:
    theta = X_te[idx]
    y_true = Y_te[idx]

    y_hat, y_var = surrogate.predict(theta)
    y_hat = np.asarray(y_hat, dtype=float).reshape(-1)
    y_var = np.asarray(y_var, dtype=float).reshape(-1)
    y_std = np.sqrt(y_var)

    plt.figure(figsize=(7, 4))
    plt.plot(x_obs, y_true, "o-", label="True output")
    plt.plot(x_obs, y_hat, "s--", label="Surrogate mean")
    plt.fill_between(
        x_obs,
        y_hat - z_plot * y_std,
        y_hat + z_plot * y_std,
        alpha=0.3,
        label="95% predictive interval",
    )
    plt.xlabel("x")
    plt.ylabel("Observed displacement")
    plt.title(f"Surrogate prediction ({label} case)")
    plt.legend()
    plt.grid(True)
    plt.show()