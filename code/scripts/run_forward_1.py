import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from gp_active_mcmc.utils import set_seed
from gp_active_mcmc.pod import POD
from gp_active_mcmc.gp_surrogate import GPSurrogate
from gp_active_mcmc.toy import toy_forward, make_timeline


def make_design_gaussian(rng, mean, cov, n):
    return rng.multivariate_normal(mean=mean, cov=cov, size=n)


def fit_pod_gp(X_tr, Y_tr, r):
    pod = POD(r=r).fit(Y_tr)
    A_tr = pod.project(Y_tr)  # (Ntr, r)
    gps = [GPSurrogate(X_tr, A_tr[:, k]) for k in range(r)]
    return pod, gps


def predict_coeffs(gps, theta):
    r = len(gps)
    mu = np.zeros(r)
    var = np.zeros(r)
    for k, gpk in enumerate(gps):
        mu_k, var_k = gpk.predict_loglike(theta)  # scalar mean/var
        mu[k] = mu_k
        var[k] = var_k
    return mu, var


def predict_series(pod, gps, theta):
    """
    Returns:
      y_hat: (T,)
      y_var: (T,) approx variance, assuming coefficient independence
    """
    mu_a, var_a = predict_coeffs(gps, theta)
    y_hat = pod.reconstruct(mu_a.reshape(1, -1))[0]
    Phi = pod.phi_  # (T, r)
    y_var = (Phi**2) @ var_a
    return y_hat, y_var


def rmse(y_hat, y_true):
    return float(np.sqrt(np.mean((y_hat - y_true) ** 2)))


def coverage(y_true, y_hat, y_std, z):
    lo = y_hat - z * y_std
    hi = y_hat + z * y_std
    return float(np.mean((y_true >= lo) & (y_true <= hi)))


def plot_pair_scatter(X, names=("A", "f", "tau"), title="Training points"):
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], s=15, alpha=0.7)
    plt.xlabel(names[0]); plt.ylabel(names[1]); plt.title(f"{title}: {names[0]} vs {names[1]}")
    plt.grid(True); plt.show()

    plt.figure()
    plt.scatter(X[:, 0], X[:, 2], s=15, alpha=0.7)
    plt.xlabel(names[0]); plt.ylabel(names[2]); plt.title(f"{title}: {names[0]} vs {names[2]}")
    plt.grid(True); plt.show()

    plt.figure()
    plt.scatter(X[:, 1], X[:, 2], s=15, alpha=0.7)
    plt.xlabel(names[1]); plt.ylabel(names[2]); plt.title(f"{title}: {names[1]} vs {names[2]}")
    plt.grid(True); plt.show()


def plot_uncertainty_slice(pod, gps, X_tr, theta_center, idx_x=0, idx_y=1, grid=60,
                           names=("A", "f", "tau")):
    """
    Heatmap of scalar uncertainty summary over a 2D slice:
    summary = mean_t std(t)
    """
    # ranges from training spread
    x_min, x_max = np.percentile(X_tr[:, idx_x], [1, 99])
    y_min, y_max = np.percentile(X_tr[:, idx_y], [1, 99])

    xs = np.linspace(x_min, x_max, grid)
    ys = np.linspace(y_min, y_max, grid)

    U = np.zeros((grid, grid))
    for i, xv in enumerate(xs):
        for j, yv in enumerate(ys):
            th = theta_center.copy()
            th[idx_x] = xv
            th[idx_y] = yv
            _, y_var = predict_series(pod, gps, th)
            y_std = np.sqrt(np.maximum(y_var, 1e-14))
            U[j, i] = np.mean(y_std)  # scalar summary

    plt.figure()
    plt.imshow(U, origin="lower", aspect="auto",
               extent=[x_min, x_max, y_min, y_max])
    plt.colorbar(label="mean_t predictive std")
    plt.scatter(X_tr[:, idx_x], X_tr[:, idx_y], s=15, c="k", alpha=0.5, label="train")
    plt.xlabel(names[idx_x]); plt.ylabel(names[idx_y])
    plt.title("Uncertainty map (2D slice) + training points")
    plt.grid(False)
    plt.legend()
    plt.show()


def main():
    rng = set_seed(7)

    t = make_timeline(T=500, t_end=0.05)

    theta_mean = np.array([0.8, 150.0, 0.010])
    theta_cov = np.diag([0.20**2, 30.0**2, 0.003**2])

    N = 100
    X = make_design_gaussian(rng, theta_mean, theta_cov, N)
    Y = np.array([toy_forward(X[i], t) for i in range(N)])

    X_tr, X_te, Y_tr, Y_te = train_test_split(X, Y, test_size=0.25, random_state=0)

    r = 10
    pod, gps = fit_pod_gp(X_tr, Y_tr, r)

    # 1) show where points are
    plot_pair_scatter(X_tr, title="Training points")
    plot_pair_scatter(X_te, title="Test points")

    # 2a) uncertainty map on a slice train
    plot_uncertainty_slice(pod, gps, X_tr, theta_center=X_tr.mean(axis=0), idx_x=0, idx_y=1)
    
    # 2b) uncertainty map on a slice test
    plot_uncertainty_slice(pod, gps, X_te, theta_center=X_te.mean(axis=0), idx_x=0, idx_y=1)

    # Evaluate forward error and uncertainty on test set
    test_rmse = []
    test_u = []       # scalar uncertainty summary
    cov50 = []
    cov90 = []
    cov95 = []

    z50, z90, z95 = 0.67449, 1.64485, 1.95996

    for i in range(X_te.shape[0]):
        y_hat, y_var = predict_series(pod, gps, X_te[i])
        y_true = Y_te[i]
        y_std = np.sqrt(np.maximum(y_var, 1e-14))

        test_rmse.append(rmse(y_hat, y_true))
        test_u.append(float(np.mean(y_std)))

        cov50.append(coverage(y_true, y_hat, y_std, z50))
        cov90.append(coverage(y_true, y_hat, y_std, z90))
        cov95.append(coverage(y_true, y_hat, y_std, z95))

    test_rmse = np.array(test_rmse)
    test_u = np.array(test_u)

    print(f"POD rank r = {r}")
    print(f"Test RMSE mean   : {test_rmse.mean():.6f}")
    print(f"Test RMSE median : {np.median(test_rmse):.6f}")
    print(f"Mean uncertainty : {test_u.mean():.6f}")
    print(f"Coverage 50% mean: {np.mean(cov50):.3f}")
    print(f"Coverage 90% mean: {np.mean(cov90):.3f}")
    print(f"Coverage 95% mean: {np.mean(cov95):.3f}")

    # 3) error vs uncertainty scatter
    plt.figure()
    plt.scatter(test_u, test_rmse, alpha=0.8)
    plt.xlabel("mean_t predictive std (summary)")
    plt.ylabel("RMSE of trajectory")
    plt.title("Error vs predicted uncertainty (should correlate)")
    plt.grid(True)
    plt.show()

    # 4) plot a few representative trajectories: best/median/worst
    idx_best = int(np.argmin(test_rmse))
    idx_worst = int(np.argmax(test_rmse))
    idx_med = int(np.argsort(test_rmse)[len(test_rmse)//2])

    for label, idx in [("best", idx_best), ("median", idx_med), ("worst", idx_worst)]:
        y_hat, y_var = predict_series(pod, gps, X_te[idx])
        y_true = Y_te[idx]
        y_std = np.sqrt(np.maximum(y_var, 1e-14))

        plt.figure()
        plt.plot(t, y_true, label="true")
        plt.plot(t, y_hat, label="POD+GP mean")
        plt.fill_between(t, y_hat - 2*y_std, y_hat + 2*y_std, alpha=0.2, label="±2 std (approx)")
        plt.title(f"Trajectory reconstruction ({label})")
        plt.grid(True)
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()
