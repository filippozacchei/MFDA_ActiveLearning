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
    gps = [GPSurrogate(X_tr, A_tr[:, k], ard=True, kernel="matern32") for k in range(r)]
    return pod, gps

def predict_coeffs(gps, theta):
    r = len(gps)
    mu = np.zeros(r)
    var = np.zeros(r)
    for k, gpk in enumerate(gps):
        mu_k, var_k = gpk.predict(theta)  # scalar mean/var
        mu[k] = mu_k
        var[k] = var_k
    return mu, var

def predict_series(pod, gps, theta):
    """
    POD+GP prediction.
    Returns:
      y_hat: (T,)
      y_var: (T,) approx variance, assuming coefficient independence
    """
    mu_a, var_a = predict_coeffs(gps, theta)
    y_hat = pod.reconstruct(mu_a.reshape(1, -1))[0]
    Phi = pod.phi_  # (T, r)
    y_var = (Phi**2) @ var_a
    return y_hat, y_var

def pod_only_reconstruction(pod, y_true):
    """
    POD-only reconstruction using true coefficients (no GP).
    Returns reconstructed series.
    """
    a_true = pod.project(y_true.reshape(1, -1))  # (1, r)
    y_pod = pod.reconstruct(a_true)[0]
    return y_pod

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
    plt.imshow(U, origin="lower", aspect="auto", extent=[x_min, x_max, y_min, y_max])
    plt.colorbar(label="mean_t predictive std")
    plt.scatter(X_tr[:, idx_x], X_tr[:, idx_y], s=16, alpha=0.5, label="train")
    plt.scatter(X_te[:, idx_x], X_te[:, idx_y], s=20, alpha=0.7, label="test")

    if highlight is not None:
        for lab, th in highlight.items():
            plt.scatter(th[idx_x], th[idx_y], s=130, marker="X",
                        linewidths=1.5, edgecolors="k", label=lab)

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

def coeff_truth_from_pod(pod: POD, Y: np.ndarray) -> np.ndarray:
    """Return true POD coefficients A for a set of trajectories Y: (N,T) -> (N,r)."""
    return pod.project(Y)


def plot_coeff_surface(
    X_tr, X_te,
    A_tr_true, A_te_true,          # true coefficients from POD projection
    gps,
    k: int,
    theta_center: np.ndarray,
    idx_x: int = 0,
    idx_y: int = 1,
    grid: int = 80,
    names=("A", "f", "tau"),
    highlight=None,                # dict label->theta
):
    """
    Plot coefficient a_k surfaces on a 2D slice for:
      - true coefficient (interpolated via griddata is avoided; we show scatter + a binned map)
      - GP mean
      - GP std
    """

    # Slice domain from combined data
    X_all = np.vstack([X_tr, X_te])
    x_min, x_max = np.percentile(X_all[:, idx_x], [1, 99])
    y_min, y_max = np.percentile(X_all[:, idx_y], [1, 99])

    xs = np.linspace(x_min, x_max, grid)
    ys = np.linspace(y_min, y_max, grid)

    # --- GP mean/std on grid ---
    MU = np.zeros((grid, grid), dtype=float)
    SD = np.zeros((grid, grid), dtype=float)

    for ix, xv in enumerate(xs):
        th_base = theta_center.copy()
        th_base[idx_x] = xv
        for iy, yv in enumerate(ys):
            th = th_base.copy()
            th[idx_y] = yv
            mu_k, var_k = gps[k].predict_loglike(th)
            MU[iy, ix] = float(mu_k)
            SD[iy, ix] = float(np.sqrt(max(float(var_k), 1e-14)))

    # --- True coefficients shown as scatter (train/test) ---
    # (This avoids adding SciPy dependencies for interpolation; it’s still very informative.)
    def _scatter_truth(title, X, A_true, alpha, label_prefix):
        plt.scatter(
            X[:, idx_x], X[:, idx_y],
            c=A_true[:, k], s=28, alpha=alpha,
            label=f"{label_prefix} (colored by true a_{k})"
        )
        plt.xlabel(names[idx_x]); plt.ylabel(names[idx_y])
        plt.title(title)
        cb = plt.colorbar()
        cb.set_label(f"true a_{k}")

    # 1) True coeff scatter
    plt.figure()
    _scatter_truth(
        title=f"True POD coefficient a_{k}: scatter (train/test)",
        X=X_tr, A_true=A_tr_true, alpha=0.65, label_prefix="train"
    )
    plt.scatter(X_te[:, idx_x], X_te[:, idx_y], c=A_te_true[:, k], s=36, alpha=0.85, marker="^", label="test")
    if highlight is not None:
        for lab, th in highlight.items():
            plt.scatter(th[idx_x], th[idx_y], s=140, marker="X", edgecolors="k", linewidths=1.5, label=lab)
    plt.grid(True)
    plt.legend()
    plt.show()

    # 2) GP mean heatmap + points
    plt.figure()
    plt.imshow(MU, origin="lower", aspect="auto", extent=[x_min, x_max, y_min, y_max])
    plt.colorbar(label=f"GP mean μ[a_{k}]")
    plt.scatter(X_tr[:, idx_x], X_tr[:, idx_y], s=14, alpha=0.35, label="train")
    plt.scatter(X_te[:, idx_x], X_te[:, idx_y], s=18, alpha=0.55, label="test")
    if highlight is not None:
        for lab, th in highlight.items():
            plt.scatter(th[idx_x], th[idx_y], s=140, marker="X", edgecolors="k", linewidths=1.5, label=lab)
    plt.xlabel(names[idx_x]); plt.ylabel(names[idx_y])
    plt.title(f"GP mean surface for POD coeff a_{k} (slice {names[idx_x]} vs {names[idx_y]})")
    plt.legend()
    plt.show()

    # 3) GP std heatmap + points
    plt.figure()
    plt.imshow(SD, origin="lower", aspect="auto", extent=[x_min, x_max, y_min, y_max])
    plt.colorbar(label=f"GP std σ[a_{k}]")
    plt.scatter(X_tr[:, idx_x], X_tr[:, idx_y], s=14, alpha=0.35, label="train")
    plt.scatter(X_te[:, idx_x], X_te[:, idx_y], s=18, alpha=0.55, label="test")
    if highlight is not None:
        for lab, th in highlight.items():
            plt.scatter(th[idx_x], th[idx_y], s=140, marker="X", edgecolors="k", linewidths=1.5, label=lab)
    plt.xlabel(names[idx_x]); plt.ylabel(names[idx_y])
    plt.title(f"GP uncertainty surface for POD coeff a_{k} (slice {names[idx_x]} vs {names[idx_y]})")
    plt.legend()
    plt.show()
    
def pod_energy_from_snapshots(Y: np.ndarray, center: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute POD energy content from snapshots Y (N, T).
    Returns:
      energy_frac : (m,) fraction per mode
      energy_cum  : (m,) cumulative fraction
    Uses SVD on the snapshot matrix.
    """
    X = Y.copy()
    if center:
        X = X - X.mean(axis=0, keepdims=True)

    # SVD of (N,T): X = U S V^T
    # singular values S relate to energy via S^2
    _, S, _ = np.linalg.svd(X, full_matrices=False)
    lam = S**2
    energy_frac = lam / np.sum(lam)
    energy_cum = np.cumsum(energy_frac)
    return energy_frac, energy_cum


def plot_pod_energy_curves(Y_tr: np.ndarray, r_max: int = 50, center: bool = True, thresholds=(0.90, 0.95, 0.99)):
    """
    Plots:
      (i) per-mode energy (log scale)
      (ii) cumulative energy with threshold lines and suggested ranks
    """
    e_frac, e_cum = pod_energy_from_snapshots(Y_tr, center=center)
    m = min(len(e_frac), r_max)

    # Per-mode energy
    plt.figure()
    plt.semilogy(np.arange(1, m + 1), e_frac[:m], marker="o")
    plt.xlabel("mode index k")
    plt.ylabel("energy fraction (per mode)")
    plt.title("POD energy per mode")
    plt.grid(True)
    plt.show()

    # Cumulative energy
    plt.figure()
    plt.plot(np.arange(1, m + 1), e_cum[:m], marker="o")
    for th in thresholds:
        plt.axhline(th, linestyle="--")
        # smallest r with cum >= th
        r_th = int(np.searchsorted(e_cum, th) + 1)
        plt.axvline(r_th, linestyle=":")
        plt.text(r_th, th, f" r={r_th}", va="bottom", ha="left")
    plt.ylim(0.0, 1.01)
    plt.xlabel("rank r")
    plt.ylabel("cumulative energy")
    plt.title("Cumulative POD energy (rank selection)")
    plt.grid(True)
    plt.show()

    # Print recommended ranks
    print("Suggested ranks (cumulative energy):")
    for th in thresholds:
        r_th = int(np.searchsorted(e_cum, th) + 1)
        print(f"  {th:.0%}: r = {r_th}")


def plot_pod_reconstruction_error_vs_rank(Y_tr: np.ndarray, Y_te: np.ndarray, r_list=(1,2,3,5,10,15,20,30), center: bool = True):
    """
    POD-only reconstruction error vs rank (no GP).
    This is very useful to separate POD truncation error from surrogate error.
    """
    # Centering consistent with energy computation
    Ymean = Y_tr.mean(axis=0, keepdims=True) if center else 0.0

    # Compute basis once with SVD of centered training snapshots
    X = Y_tr - Ymean if center else Y_tr.copy()
    U, S, Vt = np.linalg.svd(X, full_matrices=False)  # Vt: (m,T)

    def pod_reconstruct(Y: np.ndarray, r: int) -> np.ndarray:
        Xy = Y - Ymean if center else Y
        Vr = Vt[:r, :].T                  # (T,r)
        A = Xy @ Vr                       # (N,r)
        Xhat = A @ Vr.T                   # (N,T)
        return Xhat + (Ymean if center else 0.0)

    def rmse_mat(Yhat, Ytrue):
        return float(np.sqrt(np.mean((Yhat - Ytrue) ** 2)))

    rms_tr = []
    rms_te = []
    for r in r_list:
        Yhat_tr = pod_reconstruct(Y_tr, r)
        Yhat_te = pod_reconstruct(Y_te, r)
        rms_tr.append(rmse_mat(Yhat_tr, Y_tr))
        rms_te.append(rmse_mat(Yhat_te, Y_te))

    plt.figure()
    plt.plot(r_list, rms_tr, marker="o", label="train POD-only RMSE")
    plt.plot(r_list, rms_te, marker="o", label="test  POD-only RMSE")
    plt.xlabel("rank r")
    plt.ylabel("RMSE (trajectory)")
    plt.title("POD-only reconstruction error vs rank")
    plt.grid(True)
    plt.legend()
    plt.show()
    
def main():
    rng = set_seed(7)

    t = make_timeline(T=500, t_end=0.05)

    theta_mean = np.array([0.8, 150.0, 0.010])
    theta_cov = np.diag([0.4**2, 25.0**2, 0.001**2])

    N = 100
    X = make_design_gaussian(rng, theta_mean, theta_cov, N)
    Y = np.array([toy_forward(X[i], t) for i in range(N)])

    X_tr, X_te, Y_tr, Y_te = train_test_split(X, Y, test_size=0.5, random_state=0)
    # --- POD rank selection diagnostics ---
    plot_pod_energy_curves(Y_tr, r_max=50, center=True, thresholds=(0.90, 0.95, 0.99))
    plot_pod_reconstruction_error_vs_rank(
        Y_tr, Y_te,
        r_list=(1, 2, 3, 5, 8, 10, 12, 15, 20, 30),
        center=True
    )
    
    r = 25
    pod, gps = fit_pod_gp(X_tr, Y_tr, r)

    # True POD coefficients for train/test (ground truth in coefficient space)
    A_tr_true = coeff_truth_from_pod(pod, Y_tr)  # (Ntr, r)
    A_te_true = coeff_truth_from_pod(pod, Y_te)  # (Nte, r)
    
    plot_pair_scatter_train_test(X_tr, X_te, names=("A", "f", "tau"))

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

    idx_best = int(np.argmin(test_rmse))
    idx_worst = int(np.argmax(test_rmse))
    idx_med = int(np.argsort(test_rmse)[len(test_rmse) // 2])
    
    theta_w = X_te[idx_worst]
    y_w = Y_te[idx_worst]

    a_true = pod.project(y_w.reshape(1, -1))[0]      # (r,)
    a_mu, a_var = predict_coeffs(gps, theta_w)       # (r,), (r,)

    abs_err = np.abs(a_mu - a_true)
    k_sorted = np.argsort(abs_err)[::-1]

    print("Worst-case coeff errors (top 5):")
    for kk in k_sorted[:5]:
        print(
            f"k={kk:2d}  abs_err={abs_err[kk]:.4e}  pred_std={np.sqrt(a_var[kk]):.4e}  "
            f"true={a_true[kk]:.4e}  pred={a_mu[kk]:.4e}"
        )

    for lab, idx in [("best", idx_best), ("median", idx_med), ("worst", idx_worst)]:
        print(lab, "theta=", X_te[idx], "RMSE=", test_rmse[idx], "mean_std=", test_u[idx])

    highlight = {"best": X_te[idx_best], "median": X_te[idx_med], "worst": X_te[idx_worst]}
    
        # --- coefficient surfaces: show a few coefficients (e.g., first 3 or the worst ones) ---
    # You can also choose coefficients based on worst-case coefficient error:
    # k_sorted already computed for worst case
    k_to_plot = list(k_sorted[:r])  # or list(k_sorted[:3])

    # for k in k_to_plot:
    #     plot_coeff_surface(
    #         X_tr=X_tr, X_te=X_te,
    #         A_tr_true=A_tr_true, A_te_true=A_te_true,
    #         gps=gps,
    #         k=k,
    #         theta_center=X_tr.mean(axis=0),
    #         idx_x=0, idx_y=1,      # slice A vs f
    #         grid=80,
    #         names=("A", "f", "tau"),
    #         highlight=highlight,
    #     )
        
    # plot_uncertainty_slice_with_points(
    #     pod, gps,
    #     X_tr=X_tr, X_te=X_te,
    #     theta_center=X_tr.mean(axis=0),
    #     idx_x=0, idx_y=1,
    #     grid=80,
    #     names=("A", "f", "tau"),
    #     highlight=highlight,
    # )

    plot_error_vs_uncertainty(test_u, test_rmse)
    binned_reliability(test_u, test_rmse, n_bins=5)

    # Trajectories: add POD-only reconstruction as requested
    for label, idx in [("best", idx_best), ("median", idx_med), ("worst", idx_worst)]:
        theta = X_te[idx]
        y_true = Y_te[idx]

        y_gp, y_var = predict_series(pod, gps, theta)
        y_std = np.sqrt(np.maximum(y_var, 1e-14))

        y_pod = pod_only_reconstruction(pod, y_true)

        plt.figure()
        plt.plot(t, y_true, label="true")
        plt.plot(t, y_pod, label="POD-only (true coeffs)")
        plt.plot(t, y_gp, label="POD+GP mean")
        plt.fill_between(t, y_gp - 2 * y_std, y_gp + 2 * y_std, alpha=0.2, label="±2 std (approx)")
        plt.title(f"Trajectory reconstruction ({label})")
        plt.grid(True)
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()
