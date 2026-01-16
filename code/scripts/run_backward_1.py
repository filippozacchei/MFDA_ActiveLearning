from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

from gp_active_mcmc.utils import set_seed
from gp_active_mcmc.toy import toy_forward, make_timeline, make_observation
from gp_active_mcmc.pod import POD
from gp_active_mcmc.gp_surrogate import GPSurrogate
from gp_active_mcmc.algorithm1 import run_algorithm1_rwm
from gp_active_mcmc.priors import GaussianPrior


def loglike_gaussian_iid(y: np.ndarray, y_obs: np.ndarray, sigma: float) -> float:
    r = y - y_obs
    return float(-0.5 * np.sum((r / sigma) ** 2))


def make_design_gaussian(
    rng: np.random.Generator,
    mean: np.ndarray,
    cov: np.ndarray,
    n: int,
    tau_min: float = 1e-6
):
    X = rng.multivariate_normal(mean=mean, cov=cov, size=n)
    bad = X[:, 2] <= tau_min
    while np.any(bad):
        X[bad] = rng.multivariate_normal(mean=mean, cov=cov, size=int(np.sum(bad)))
        bad = X[:, 2] <= tau_min
    return X


class PODCoeffSurrogateAsLoglike:
    def __init__(
        self,
        pod: POD,
        gps: list[GPSurrogate],
        y_obs: np.ndarray,
        sigma_obs: float,
        rng: np.random.Generator,
        cache: dict,
        n_mc: int = 32,
        coeff_var_floor: float = 1e-12,
    ):
        self.pod = pod
        self.gps = gps
        self.y_obs = y_obs
        self.sigma_obs = float(sigma_obs)
        self.rng = rng
        self.cache = cache
        self.n_mc = int(n_mc)
        self.coeff_var_floor = float(coeff_var_floor)

    def _predict_coeffs(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r = len(self.gps)
        mu = np.zeros(r)
        var = np.zeros(r)
        for k, gpk in enumerate(self.gps):
            mu_k, var_k = gpk.predict_loglike(theta)
            mu[k] = float(mu_k)
            var[k] = float(var_k)
        return mu, var

    def predict_loglike(self, theta: np.ndarray) -> tuple[float, float]:
        mu_a, var_a = self._predict_coeffs(theta)
        std_a = np.sqrt(np.maximum(var_a, self.coeff_var_floor))

        Z = self.rng.normal(0.0, 1.0, size=(self.n_mc, mu_a.size))
        A_s = mu_a[None, :] + Z * std_a[None, :]
        Y_s = self.pod.reconstruct(A_s)

        logL = np.array([loglike_gaussian_iid(Y_s[i], self.y_obs, self.sigma_obs) for i in range(self.n_mc)])
        return float(logL.mean()), float(logL.var())

    def update(self, theta: np.ndarray, logL_true: float, gamma_L_ratio: float, n_retrain_max: int) -> None:
        y_true = None
        if "theta" in self.cache and np.allclose(self.cache["theta"], theta):
            y_true = self.cache.get("y", None)

        if y_true is None:
            y_true = self.cache["forward_fn"](theta)

        a_true = self.pod.project(y_true.reshape(1, -1))[0]

        for k, gpk in enumerate(self.gps):
            ak = float(a_true[k])
            gpk.update(theta, ak, gamma_L_ratio, n_retrain_max)


def least_squares_init(
    theta_guess: np.ndarray,
    y_obs: np.ndarray,
    t: np.ndarray,
    sigma_obs: float,
    tau_min: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      theta_ls: least-squares solution
      cov_ls  : Gauss–Newton covariance estimate at theta_ls
    """

    def residuals(theta: np.ndarray) -> np.ndarray:
        th = theta.copy()
        # enforce tau positive in the forward call (prevents exploding exp)
        th[2] = max(th[2], tau_min)
        y = toy_forward(th, t)
        return (y - y_obs) / sigma_obs

    # keep parameters in a sane region
    lb = np.array([0.0, 1.0, tau_min])
    ub = np.array([2.0, 500.0, 0.05])

    res = least_squares(residuals, x0=theta_guess, bounds=(lb, ub), method="trf")

    theta_ls = res.x.copy()

    # Gauss–Newton covariance: s^2 (J^T J)^(-1)
    J = res.jac  # (T, d)
    JTJ = J.T @ J
    d = theta_ls.size
    ndata = J.shape[0]

    # residual variance estimate
    rss = float(np.sum(res.fun ** 2))
    dof = max(ndata - d, 1)
    s2 = rss / dof

    cov_ls = s2 * np.linalg.pinv(JTJ)
    return theta_ls, cov_ls

import os


def surrogate_loglike_stats(gp_like, theta: np.ndarray) -> tuple[float, float]:
    """Convenience wrapper: returns (mu, var) from surrogate at theta."""
    mu, var = gp_like.predict_loglike(theta)
    return float(mu), float(var)


def plot_true_point_diagnostics(
    mu_hist: np.ndarray,
    var_hist: np.ndarray,
    title_prefix: str = "Surrogate at true θ"
):
    it = np.arange(len(mu_hist))

    plt.figure()
    plt.plot(it, mu_hist)
    plt.title(f"{title_prefix}: mean loglike")
    plt.xlabel("checkpoint index")
    plt.ylabel("mu_loglike")
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.semilogy(it, np.maximum(var_hist, 1e-18))
    plt.title(f"{title_prefix}: variance loglike")
    plt.xlabel("checkpoint index")
    plt.ylabel("var_loglike (log scale)")
    plt.grid(True)
    plt.show()


def plot_new_sampling_points_2d(
    X0: np.ndarray,
    chain: np.ndarray,
    forward_points: np.ndarray,
    idx_x: int = 0,
    idx_y: int = 1,
    names=("A", "f", "tau"),
    title="2D points: initial design, chain, and new forward calls",
):
    plt.figure()
    plt.scatter(X0[:, idx_x], X0[:, idx_y], s=14, alpha=0.25, label="initial design X0")
    plt.scatter(chain[:, idx_x], chain[:, idx_y], s=10, alpha=0.25, label="chain (thinned)")
    if forward_points.size > 0:
        plt.scatter(forward_points[:, idx_x], forward_points[:, idx_y], s=45, alpha=0.8, marker="x", label="new forward points")
    plt.xlabel(names[idx_x]); plt.ylabel(names[idx_y])
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()


def uncertainty_map_2d(
    gp_like,
    theta_center: np.ndarray,
    idx_x: int,
    idx_y: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    grid: int = 80,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      xs, ys, U  where U[iy,ix] = sqrt( Var(loglike(theta)) )
    """
    xs = np.linspace(x_min, x_max, grid)
    ys = np.linspace(y_min, y_max, grid)

    U = np.zeros((grid, grid), dtype=float)
    for ix, xv in enumerate(xs):
        th_base = theta_center.copy()
        th_base[idx_x] = xv
        for iy, yv in enumerate(ys):
            th = th_base.copy()
            th[idx_y] = yv
            _, var = gp_like.predict_loglike(th)
            U[iy, ix] = float(np.sqrt(max(float(var), 1e-18)))
    return xs, ys, U


def plot_uncertainty_map_with_chain(
    gp_like,
    theta_center: np.ndarray,
    X0: np.ndarray,
    chain: np.ndarray,
    forward_points: np.ndarray,
    idx_x: int = 0,
    idx_y: int = 1,
    grid: int = 80,
    names=("A", "f", "tau"),
    bounds_from: str = "X0",   # "X0" or "chain" or "both"
    p_low: float = 1.0,
    p_high: float = 99.0,
    title="Uncertainty map (sqrt Var loglike) + points",
):
    if bounds_from == "X0":
        Xref = X0
    elif bounds_from == "chain":
        Xref = chain
    else:
        Xref = np.vstack([X0, chain])

    x_min, x_max = np.percentile(Xref[:, idx_x], [p_low, p_high])
    y_min, y_max = np.percentile(Xref[:, idx_y], [p_low, p_high])

    xs, ys, U = uncertainty_map_2d(
        gp_like=gp_like,
        theta_center=theta_center,
        idx_x=idx_x, idx_y=idx_y,
        x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
        grid=grid,
    )

    plt.figure()
    plt.imshow(
        U, origin="lower", aspect="auto",
        extent=[x_min, x_max, y_min, y_max]
    )
    plt.colorbar(label="sqrt(Var surrogate loglike)")
    plt.scatter(X0[:, idx_x], X0[:, idx_y], s=12, alpha=0.25, label="X0")
    plt.scatter(chain[:, idx_x], chain[:, idx_y], s=10, alpha=0.25, label="chain")
    if forward_points.size > 0:
        plt.scatter(forward_points[:, idx_x], forward_points[:, idx_y], s=45, alpha=0.9, marker="x", label="new forward")
    plt.xlabel(names[idx_x]); plt.ylabel(names[idx_y])
    plt.title(title)
    plt.legend()
    plt.show()


def save_gif_frames_uncertainty(
    gp_like,
    theta_center: np.ndarray,
    X0: np.ndarray,
    chain_full: np.ndarray,
    forward_points_full: np.ndarray,
    idx_x: int = 0,
    idx_y: int = 1,
    grid: int = 80,
    names=("A", "f", "tau"),
    out_dir: str = "gif_frames",
    every: int = 500,
    max_frames: int = 60,
    p_low: float = 1.0,
    p_high: float = 99.0,
):
    """
    Saves PNG frames showing uncertainty map + chain up to time n.
    You can stitch afterwards:
      magick -delay 10 -loop 0 gif_frames/frame_*.png out.gif
    """
    os.makedirs(out_dir, exist_ok=True)

    # bounds from X0 + chain for stable axis across frames
    Xref = np.vstack([X0, chain_full])
    x_min, x_max = np.percentile(Xref[:, idx_x], [p_low, p_high])
    y_min, y_max = np.percentile(Xref[:, idx_y], [p_low, p_high])

    frames = 0
    for n in range(0, chain_full.shape[0], every):
        if frames >= max_frames:
            break

        chain_n = chain_full[: n + 1]
        # forward points available up to n (approx)
        fwd_n = forward_points_full[forward_points_full[:, 3] <= n][:, :3] if forward_points_full.size > 0 else np.empty((0, 3))

        _, _, U = uncertainty_map_2d(
            gp_like=gp_like,
            theta_center=theta_center,
            idx_x=idx_x, idx_y=idx_y,
            x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
            grid=grid,
        )

        plt.figure()
        plt.imshow(
            U, origin="lower", aspect="auto",
            extent=[x_min, x_max, y_min, y_max]
        )
        plt.colorbar(label="sqrt(Var surrogate loglike)")
        plt.scatter(X0[:, idx_x], X0[:, idx_y], s=10, alpha=0.2, label="X0")
        plt.scatter(chain_n[:, idx_x], chain_n[:, idx_y], s=10, alpha=0.3, label="chain")
        if fwd_n.size > 0:
            plt.scatter(fwd_n[:, idx_x], fwd_n[:, idx_y], s=45, alpha=0.9, marker="x", label="new forward")
        plt.xlabel(names[idx_x]); plt.ylabel(names[idx_y])
        plt.title(f"n = {n}")
        plt.legend(loc="best")
        path = os.path.join(out_dir, f"frame_{frames:03d}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        frames += 1

    print(f"Saved {frames} frames in: {out_dir}")

def main():
    make_plots = True
    make_gif_frames = True
    
    rng = set_seed(7)

    # Timeline + truth
    t = make_timeline(T=500, t_end=0.05)
    theta_true = np.array([0.8, 150.0, 0.010])
    sigma_obs = 0.02
    y_obs = make_observation(rng, theta_true, t, sigma_obs)

    # Prior for inversion
    prior_mean = np.array([0.8, 150.0, 0.010])
    prior_cov = np.diag([0.25**2, 40.0**2, 0.004**2])
    prior = GaussianPrior(prior_mean, prior_cov)

    # Initial training data for POD+GP surrogate
    N0 = 100
    X0 = make_design_gaussian(rng, prior_mean, prior_cov, N0)
    Y0 = np.array([toy_forward(X0[i], t) for i in range(N0)])

    # Fit POD and coefficient GPs
    r = 5
    pod = POD(r=r).fit(Y0)
    A0 = pod.project(Y0)

    gps = []
    for k in range(r):
        gps.append(GPSurrogate(X0, A0[:, k], kernel="matern52", ard=True))

    # Cache to avoid recomputing forward in update()
    cache = {"forward_fn": lambda th: toy_forward(th, t)}

    def loglike_true_fn(theta: np.ndarray) -> float:
        y = toy_forward(theta, t)
        ll = loglike_gaussian_iid(y, y_obs, sigma_obs)
        cache["theta"] = theta.copy()
        cache["y"] = y
        cache["ll"] = ll
        return ll

    gp_like = PODCoeffSurrogateAsLoglike(
        pod=pod,
        gps=gps,
        y_obs=y_obs,
        sigma_obs=sigma_obs,
        rng=rng,
        cache=cache,
        n_mc=32,
    )

    # --------- Least-squares initialization (x0 and cov) ----------
    theta_guess = prior_mean.copy()
    theta0, cov_ls = least_squares_init(theta_guess, y_obs, t, sigma_obs, tau_min=1e-6)

    # Proposal covariance: LS covariance + jitter
    cov_prop = cov_ls + 1e-8 * np.eye(3)

    print("Least-squares theta0:", theta0)
    print("LS covariance diag  :", np.diag(cov_prop))

    # Hard constraint
    constraint_fn = lambda th: (th[2] > 1e-6)

    # Run Algorithm 1
    out = run_algorithm1_rwm(
        rng=rng,
        theta0=theta0,
        cov=cov_prop,
        n_total=30000,
        gamma_var=0.5,
        gamma_L_ratio=2.5,
        n_retrain_max=50,
        step_scale=1.0,    # NOTE: since cov_prop already scaled, start at 1.0
        gp=gp_like,
        loglike_true_fn=loglike_true_fn,
        prior=prior,
        constraint_fn=constraint_fn,
        verbose=True,
        print_every=200,
    )

    burnin = 10000
    Nt = 5
    chain = out["chain"][burnin:-1:Nt]
    used_forward = out["used_forward"]

    print("Acceptance rate:", out["accept_rate"])
    print("Forward eval fraction:", float(np.mean(used_forward)))

    # Diagnostics
    burn = int(0.3 * len(chain))
    names = ["A", "f", "tau"]

    for j, nm in enumerate(names):
        plt.figure()
        plt.plot(chain[:, j])
        plt.axhline(theta_true[j], linestyle="--", label="true")
        plt.title(f"Trace: {nm}")
        plt.grid(True)
        plt.legend()
        plt.show()

        plt.figure()
        plt.hist(chain[burn:, j], bins=40, density=True)
        plt.axvline(theta_true[j], linestyle="--", label="true")
        plt.title(f"Posterior (approx): {nm}")
        plt.grid(True)
        plt.legend()
        plt.show()

    theta_hat = chain[burn:].mean(axis=0)
    y_hat = toy_forward(theta_hat, t)

    plt.figure()
    plt.plot(t, y_obs, label="obs")
    plt.plot(t, y_hat, label="forward(posterior mean)")
    plt.title("Data fit at posterior mean (rough)")
    plt.grid(True)
    plt.legend()
    plt.show()
    
        # --- Keep full chain (for plots / frames) ---
    chain_full = out["chain"]              # (n_total+1, d)
    used_forward = out["used_forward"]     # (n_total,)
    accept_rate = out["accept_rate"]

    print("Acceptance rate:", accept_rate)
    print("Forward eval fraction:", float(np.mean(used_forward)))

    # Identify new forward points (where the surrogate was updated)
    # We store also the iteration index for frame selection.
    forward_idx = np.where(used_forward)[0]  # iteration n in [0, n_total-1]
    if forward_idx.size > 0:
        forward_points = chain_full[forward_idx + 1]  # theta at step n+1
        forward_points_full = np.hstack([forward_points, forward_idx.reshape(-1, 1)])  # (nfwd, 4)
    else:
        forward_points = np.empty((0, 3))
        forward_points_full = np.empty((0, 4))

    # True-point diagnostics: check if surrogate improves at theta_true
    # Evaluate at checkpoints along the chain
    if make_plots:
        checkpoints = np.unique(np.linspace(0, chain_full.shape[0] - 1, 30, dtype=int))
        mu_hist = np.zeros(checkpoints.size)
        var_hist = np.zeros(checkpoints.size)

        # NOTE: your surrogate has been updated online during MCMC.
        # We just record its current prediction at theta_true over time checkpoints.
        # (If you want strict "online" history, you'd need to snapshot model state, expensive.)
        for i, _ in enumerate(checkpoints):
            mu_hist[i], var_hist[i] = surrogate_loglike_stats(gp_like, theta_true)

        plot_true_point_diagnostics(mu_hist, var_hist, title_prefix="Surrogate at true θ (current state)")

        # 2D points plot
        plot_new_sampling_points_2d(
            X0=X0,
            chain=chain_full[::50],          # thin for visualization
            forward_points=forward_points,
            idx_x=0, idx_y=1,
            names=("A", "f", "tau"),
            title="Points in (A,f): X0, chain, and new forward evaluations"
        )

        # 2D uncertainty map + chain
        plot_uncertainty_map_with_chain(
            gp_like=gp_like,
            theta_center=theta0.copy(),
            X0=X0,
            chain=chain_full[::50],
            forward_points=forward_points,
            idx_x=0, idx_y=1,
            grid=90,
            names=("A", "f", "tau"),
            bounds_from="both",
            title="Uncertainty map (A,f slice) + X0 + chain + new forward points"
        )

    if make_gif_frames:
        save_gif_frames_uncertainty(
            gp_like=gp_like,
            theta_center=theta0.copy(),
            X0=X0,
            chain_full=chain_full,
            forward_points_full=forward_points_full,
            idx_x=0, idx_y=1,
            grid=80,
            names=("A", "f", "tau"),
            out_dir="gif_frames_uncertainty",
            every=800,          # adjust
            max_frames=60,
        )


if __name__ == "__main__":
    main()
