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


def plot_pair_scatter_train_test(X_tr, X_te, names=("A", "f", "tau")):
    pairs = [(0, 1), (0, 2), (1, 2)]
    for i, j in pairs:
        plt.figure()
        plt.scatter(X_tr[:, i], X_tr[:, j], s=18, alpha=0.6, label="train")
        plt.scatter(X_te[:, i], X_te[:, j], s=22, alpha=0.8, label="test")
        plt.xlabel(names[i])
        plt.ylabel(names[j])
        plt.title(f"Design points: {names[i]} vs {names[j]}")
        plt.grid(True)
        plt.legend()
        plt.show()


def plot_uncertainty_slice_with_points(
    pod, gps,
    X_tr, X_te,
    theta_center,
    idx_x=0, idx_y=1,
    grid=80,
    names=("A", "f", "tau"),
    highlight=None,  # dict: {"best": theta, "median": theta, "worst": theta}
):
    """
    Heatmap of mean_t predictive std over a 2D slice (idx_x, idx_y), with
    train/test points and optional highlighted thetas.
    """
    X_all = np.vstack([X_tr, X_te])
    x_min, x_max = np.percentile(X_all[:, idx_x], [1, 99])
    y_min, y_max = np.percentile(X_all[:, idx_y], [1, 99])

    xs = np.linspace(x_min, x_max, grid)
    ys = np.linspace(y_min, y_max, grid)

    U = np.zeros((grid, grid), dtype=float)
    for ix, xv in enumerate(xs):
        th_base = theta_center.copy()
        th_base[idx_x] = xv
        for iy, yv in enumerate(ys):
            th = th_base.copy()
            th[idx_y] = yv
            _, y_var = predict_series(pod, gps, th)
            y_std = np.sqrt(np.maximum(y_var, 1e-14))
            U[iy, ix] = np.mean(y_std)

    plt.figure()
    plt.imshow(
        U, origin="lower", aspect="auto",
        extent=[x_min, x_max, y_min, y_max]
    )
    plt.colorbar(label="mean_t predictive std")
    plt.scatter(X_tr[:, idx_x], X_tr[:, idx_y], s=16, alpha=0.5, label="train")
    plt.scatter(X_te[:, idx_x], X_te[:, idx_y], s=20, alpha=0.7, label="test")

    if highlight is not None:
        for lab, th in highlight.items():
            plt.scatter(
                th[idx_x], th[idx_y],
                s=130, marker="X", linewidths=1.5,
                edgecolors="k",
                label=lab
            )

    plt.xlabel(names[idx_x])
    plt.ylabel(names[idx_y])
    plt.title(f"Uncertainty map slice: {names[idx_x]} vs {names[idx_y]}")
    plt.legend()
    plt.show()


def plot_error_vs_uncertainty(test_u, test_rmse):
    plt.figure()
    plt.scatter(test_u, test_rmse, alpha=0.85)
    plt.xlabel("mean_t predictive std (summary)")
    plt.ylabel("RMSE of trajectory")
    plt.title("Error vs predicted uncertainty")
    plt.grid(True)
    plt.show()

    corr = np.corrcoef(test_u, test_rmse)[0, 1] if len(test_u) > 1 else np.nan
    print("Corr(mean_std, RMSE) =", float(corr))


def binned_reliability(test_u, test_rmse, n_bins=5):
    """
    Bin by predicted uncertainty and report average RMSE per bin.
    Useful to assess monotonicity: higher predicted uncertainty -> higher error.
    """
    idx = np.argsort(test_u)
    u_sorted = test_u[idx]
    e_sorted = test_rmse[idx]

    bins = np.array_split(np.arange(len(test_u)), n_bins)
    u_bin = []
    e_bin = []
    for b in bins:
        u_bin.append(np.mean(u_sorted[b]))
        e_bin.append(np.mean(e_sorted[b]))

    plt.figure()
    plt.plot(u_bin, e_bin, marker="o")
    plt.xlabel("bin mean predicted uncertainty")
    plt.ylabel("bin mean RMSE")
    plt.title("Binned reliability: uncertainty vs error")
    plt.grid(True)
    plt.show()


def main():
    rng = set_seed(7)

    t = make_timeline(T=500, t_end=0.05)

    theta_mean = np.array([0.8, 150.0, 0.010])
    theta_cov = np.diag([0.20**2, 30.0**2, 0.003**2])

    N = 1000
    X = make_design_gaussian(rng, theta_mean, theta_cov, N)
    Y = np.array([toy_forward(X[i], t) for i in range(N)])

    X_tr, X_te, Y_tr, Y_te = train_test_split(X, Y, test_size=0.25, random_state=0)

    r = 10
    pod, gps = fit_pod_gp(X_tr, Y_tr, r)

    # 1) One set of pair plots with train/test overlaid
    plot_pair_scatter_train_test(X_tr, X_te, names=("A", "f", "tau"))

    # Evaluate test errors + uncertainty summaries + coverage
    z50, z90, z95 = 0.67449, 1.64485, 1.95996
    test_rmse = np.zeros(X_te.shape[0], dtype=float)
    test_u = np.zeros(X_te.shape[0], dtype=float)
    cov50 = np.zeros(X_te.shape[0], dtype=float)
    cov90 = np.zeros(X_te.shape[0], dtype=float)
    cov95 = np.zeros(X_te.shape[0], dtype=float)

    for i in range(X_te.shape[0]):
        y_hat, y_var = predict_series(pod, gps, X_te[i])
        y_true = Y_te[i]
        y_std = np.sqrt(np.maximum(y_var, 1e-14))

        test_rmse[i] = rmse(y_hat, y_true)
        test_u[i] = float(np.mean(y_std))

        cov50[i] = coverage(y_true, y_hat, y_std, z50)
        cov90[i] = coverage(y_true, y_hat, y_std, z90)
        cov95[i] = coverage(y_true, y_hat, y_std, z95)

    print(f"POD rank r = {r}")
    print(f"Test RMSE mean   : {test_rmse.mean():.6f}")
    print(f"Test RMSE median : {np.median(test_rmse):.6f}")
    print(f"Mean uncertainty : {test_u.mean():.6f}")
    print(f"Coverage 50% mean: {cov50.mean():.3f}")
    print(f"Coverage 90% mean: {cov90.mean():.3f}")
    print(f"Coverage 95% mean: {cov95.mean():.3f}")

    # best/median/worst indices (in test set)
    idx_best = int(np.argmin(test_rmse))
    idx_worst = int(np.argmax(test_rmse))
    idx_med = int(np.argsort(test_rmse)[len(test_rmse) // 2])

    for lab, idx in [("best", idx_best), ("median", idx_med), ("worst", idx_worst)]:
        print(lab, "theta=", X_te[idx], "RMSE=", test_rmse[idx], "mean_std=", test_u[idx])

    # 2) One uncertainty slice with train/test and highlighted cases
    highlight = {
        "best": X_te[idx_best],
        "median": X_te[idx_med],
        "worst": X_te[idx_worst],
    }
    plot_uncertainty_slice_with_points(
        pod, gps,
        X_tr=X_tr, X_te=X_te,
        theta_center=X_tr.mean(axis=0),
        idx_x=0, idx_y=1,
        grid=80,
        names=("A", "f", "tau"),
        highlight=highlight,
    )

    # 3) Error vs uncertainty + binned reliability
    plot_error_vs_uncertainty(test_u, test_rmse)
    binned_reliability(test_u, test_rmse, n_bins=5)

    # 4) Trajectories for best/median/worst (same points as highlighted)
    for label, idx in [("best", idx_best), ("median", idx_med), ("worst", idx_worst)]:
        y_hat, y_var = predict_series(pod, gps, X_te[idx])
        y_true = Y_te[idx]
        y_std = np.sqrt(np.maximum(y_var, 1e-14))

        plt.figure()
        plt.plot(t, y_true, label="true")
        plt.plot(t, y_hat, label="POD+GP mean")
        plt.fill_between(t, y_hat - 2 * y_std, y_hat + 2 * y_std, alpha=0.2, label="±2 std (approx)")
        plt.title(f"Trajectory reconstruction ({label})")
        plt.grid(True)
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()
