"""
Robust UM-Bridge muq-beam runner for the MFDA_ActiveLearning / gp_active_mcmc
adaptive delayed-acceptance algorithm.

Main fixes compared with the first runner:
- the GP surrogate is trained on a clipped + standardised log-posterior target;
- catastrophic values such as -1e18 are clipped only inside the GP, never in the
  high-fidelity UM-Bridge posterior used by the fine correction;
- when --mode both is used, the active chain is seeded from the exact posterior
  tail and the GP is enriched with thinned exact warm-up samples;
- the active proposal covariance can be initialised from the exact posterior tail;
- surrogate validation is local/posterior by default, not global-prior.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import requests
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.stats import ks_2samp, norm, qmc, wasserstein_distance

FloatArray = NDArray[np.float64]

MUQ_DIM = 3
MUQ_PRIOR_MEAN = np.array([10.0, 10.0, 10.0], dtype=float)
MUQ_PRIOR_SD = np.array([2.0, 2.0, 2.0], dtype=float)


def as_jsonable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): as_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [as_jsonable(v) for v in obj]
    return obj


def finite_or_neginf(x: float) -> float:
    return float(x) if np.isfinite(x) else -np.inf


class UMBridgeLogPosterior:
    """Persistent-session wrapper around the UM-Bridge scalar log-posterior."""

    def __init__(self, url: str, model_name: str = "posterior") -> None:
        try:
            import umbridge  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Install umbridge with `python -m pip install umbridge`.") from exc

        self.url = url.rstrip("/")
        self.model_name = model_name
        self.model = umbridge.HTTPModel(self.url, model_name)
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=1, pool_maxsize=1, max_retries=0)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        # UM-Bridge's Python client calls the generic /Evaluate endpoint.
        # The model name is passed in the JSON payload. Do NOT call
        # /posterior/Evaluate: many UM-Bridge benchmark containers return 404 there.
        self.eval_url = self.url + "/Evaluate"
        self.n_calls = 0

    def check(self) -> dict[str, Any]:
        try:
            input_sizes = list(self.model.get_input_sizes())
            output_sizes = list(self.model.get_output_sizes())
        except Exception as exc:
            raise RuntimeError(
                f"Could not query UM-Bridge model at {self.url!r}. Is the Docker container running?"
            ) from exc
        if input_sizes != [MUQ_DIM] or output_sizes != [1]:
            raise RuntimeError(
                f"Unexpected UM-Bridge dimensions: input={input_sizes}, output={output_sizes}. "
                "Expected input [3] and output [1]."
            )
        return {"input_sizes": input_sizes, "output_sizes": output_sizes}

    def logp(self, theta: Any) -> float:
        th = np.asarray(theta, dtype=float).reshape(-1)
        if th.shape != (MUQ_DIM,):
            raise ValueError(f"theta must have shape ({MUQ_DIM},). Got {th.shape}.")
        if not np.all(np.isfinite(th)):
            return -np.inf

        last_exc: Exception | None = None
        for attempt in range(5):
            try:
                payload = {"name": self.model_name, "input": [th.tolist()], "config": {}}
                response = self.session.post(self.eval_url, json=payload, timeout=60.0)
                response.raise_for_status()
                out = response.json()
                # UM-Bridge protocol returns {"output": [[...]]}; keep a fallback for
                # older/minimal servers that may return the raw output list.
                if isinstance(out, dict):
                    if "error" in out and out["error"]:
                        raise RuntimeError(str(out["error"]))
                    out = out.get("output", out)
                value = float(np.asarray(out, dtype=float).reshape(-1)[0])
                self.n_calls += 1
                return finite_or_neginf(value)
            except Exception as exc:  # pragma: no cover - depends on local Docker/HTTP
                last_exc = exc
                wait = 0.25 * (2**attempt)
                print(
                    f"[retry] UM-Bridge evaluation attempt {attempt + 1}/5 failed; "
                    f"retrying in {wait:.2f}s... ({type(exc).__name__})"
                )
                time.sleep(wait)
        raise RuntimeError(f"UM-Bridge evaluation failed at theta={th.tolist()} after 5 attempts.") from last_exc

    def __call__(self, theta: Any) -> FloatArray:
        return np.array([self.logp(theta)], dtype=float)


@dataclass
class FlatPrior:
    """Improper flat prior for tinyDA because UM-Bridge already returns log posterior."""

    dim: int = MUQ_DIM
    rvs_center: FloatArray | None = None
    rvs_scale: FloatArray | None = None

    def logpdf(self, x: Any) -> float:
        arr = np.asarray(x, dtype=float).reshape(-1)
        if arr.shape != (self.dim,) or not np.all(np.isfinite(arr)):
            return -np.inf
        return 0.0

    def rvs(self, size: int | tuple[int, ...] | None = None, random_state: Any = None) -> FloatArray:
        rng = np.random.default_rng(random_state)
        center = MUQ_PRIOR_MEAN if self.rvs_center is None else np.asarray(self.rvs_center, dtype=float)
        scale = MUQ_PRIOR_SD if self.rvs_scale is None else np.asarray(self.rvs_scale, dtype=float)
        if size is None:
            return rng.normal(center, scale)
        if isinstance(size, tuple):
            return rng.normal(center, scale, size=(*size, self.dim))
        return rng.normal(center, scale, size=(int(size), self.dim))


class LogDensityLike:
    """tinyDA log-likelihood for models that output scalar log densities."""

    def __init__(self, variance_penalty: float = 0.0, floor: float | None = None) -> None:
        self.variance_penalty = float(variance_penalty)
        self.floor = None if floor is None else float(floor)
        self.n_calls = 0
        self.last_variance: float | None = None

    def loglike(self, y_pred: Any) -> float:
        self.n_calls += 1
        mean = np.asarray(y_pred, dtype=float).reshape(-1)
        if mean.size < 1 or not np.isfinite(mean[0]):
            return -np.inf
        val = float(mean[0])
        var_attr = getattr(y_pred, "variance", None)
        self.last_variance = None
        if var_attr is not None:
            var = np.asarray(var_attr, dtype=float).reshape(-1)
            if var.size and np.isfinite(var[0]) and var[0] >= 0.0:
                self.last_variance = float(var[0])
                val -= self.variance_penalty * math.sqrt(float(var[0]))
        if self.floor is not None:
            val = max(val, self.floor)
        return finite_or_neginf(val)


@dataclass(slots=True)
class LogpTransform:
    """Clip and standardise raw logp values for GP training."""

    clip_floor: float
    center: float
    scale: float
    raw_min: float
    raw_max: float
    clipped_min: float
    clipped_max: float
    n_fit: int
    n_clipped_fit: int

    @classmethod
    def fit(cls, y: Any, clip_drop: float = 500.0, min_scale: float = 1.0) -> "LogpTransform":
        arr = np.asarray(y, dtype=float).reshape(-1)
        finite = np.isfinite(arr)
        if not np.any(finite):
            raise RuntimeError("Cannot fit GP transform: all training logp values are non-finite.")
        yf = arr[finite]
        raw_max = float(np.max(yf))
        raw_min = float(np.min(yf))
        clip_floor = raw_max - float(clip_drop) if float(clip_drop) > 0.0 else -np.inf
        yc = np.maximum(yf, clip_floor)
        center = float(np.median(yc))
        scale = float(np.std(yc, ddof=1)) if yc.size > 1 else float(min_scale)
        if not np.isfinite(scale) or scale < float(min_scale):
            scale = float(min_scale)
        return cls(
            clip_floor=float(clip_floor),
            center=center,
            scale=scale,
            raw_min=raw_min,
            raw_max=raw_max,
            clipped_min=float(np.min(yc)),
            clipped_max=float(np.max(yc)),
            n_fit=int(yf.size),
            n_clipped_fit=int(np.sum(yf < clip_floor)) if np.isfinite(clip_floor) else 0,
        )

    def transform(self, y: Any) -> FloatArray:
        arr = np.asarray(y, dtype=float).reshape(-1)
        out = arr.copy()
        out[~np.isfinite(out)] = self.clip_floor
        if np.isfinite(self.clip_floor):
            out = np.maximum(out, self.clip_floor)
        return ((out - self.center) / self.scale).astype(float)

    def inverse_mean_var(self, mean_z: Any, var_z: Any) -> tuple[FloatArray, FloatArray]:
        mz = np.asarray(mean_z, dtype=float).reshape(-1)
        vz = np.asarray(var_z, dtype=float).reshape(-1)
        mean = self.center + self.scale * mz
        var = (self.scale**2) * np.maximum(vz, 0.0)
        return mean.astype(float), var.astype(float)

    def metadata(self) -> dict[str, Any]:
        return {
            "clip_floor": self.clip_floor,
            "center": self.center,
            "scale": self.scale,
            "raw_min": self.raw_min,
            "raw_max": self.raw_max,
            "clipped_min": self.clipped_min,
            "clipped_max": self.clipped_max,
            "n_fit": self.n_fit,
            "n_clipped_fit": self.n_clipped_fit,
        }


class ScalarGPSurrogate:
    """Adapter: transformed scalar MultiOutputGP -> ActiveMCMCModel surrogate protocol."""

    def __init__(
        self,
        gp: Any,
        transform: LogpTransform,
        variance_floor: float = 1e-10,
        predict_clip: bool = True,
        training_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.gp = gp
        self.transform = transform
        self.variance_floor = float(variance_floor)
        self.predict_clip = bool(predict_clip)
        self.training_metadata = {} if training_metadata is None else dict(training_metadata)
        self.n_skipped_updates = 0

    def predict(self, theta: Any) -> tuple[FloatArray, FloatArray]:
        th = np.asarray(theta, dtype=float).reshape(1, -1)
        mean_z, var_z = self.gp.predict(th)
        mean1, var1 = self.transform.inverse_mean_var(mean_z, var_z)
        mean1 = mean1[:1]
        var1 = np.maximum(var1[:1], self.variance_floor)
        if self.predict_clip and np.isfinite(self.transform.clip_floor):
            mean1 = np.maximum(mean1, self.transform.clip_floor)
        return mean1, var1

    def update(self, theta: Any, y_hf: Any) -> None:
        th = np.asarray(theta, dtype=float).reshape(1, -1)
        y = np.asarray(y_hf, dtype=float).reshape(-1)
        if y.size < 1:
            raise ValueError("y_hf must contain one scalar log-posterior value.")
        if not np.isfinite(y[0]) and not np.isfinite(self.transform.clip_floor):
            self.n_skipped_updates += 1
            return
        z = self.transform.transform(np.array([float(y[0])], dtype=float))
        self.gp.update(th, z)

    @property
    def n_train(self) -> int | None:
        return getattr(self.gp, "n_train", None)


@dataclass
class ExactMCMCResult:
    samples: FloatArray
    logp: FloatArray
    accepted: NDArray[np.bool_]
    proposal_cov: FloatArray
    acceptance_rate: float


def latin_normal_design(
    rng: np.random.Generator,
    n: int,
    mean: FloatArray = MUQ_PRIOR_MEAN,
    sd: FloatArray = MUQ_PRIOR_SD,
    clip_sigma: float = 4.0,
) -> FloatArray:
    n = int(n)
    if n <= 0:
        return np.zeros((0, MUQ_DIM), dtype=float)
    sampler = qmc.LatinHypercube(d=MUQ_DIM, seed=rng)
    u = np.clip(sampler.random(n), np.finfo(float).eps, 1.0 - np.finfo(float).eps)
    x = norm.ppf(u, loc=mean, scale=sd)
    return np.clip(x, mean - clip_sigma * sd, mean + clip_sigma * sd).astype(float)


def eval_many(logpost: UMBridgeLogPosterior, X: FloatArray, label: str) -> FloatArray:
    y = np.empty(X.shape[0], dtype=float)
    t0 = time.time()
    for i, th in enumerate(X):
        y[i] = logpost.logp(th)
    print(f"[{label}] evaluated {len(X)} exact log-posterior calls in {time.time() - t0:.2f}s")
    return y


def find_map(logpost: UMBridgeLogPosterior, x0: FloatArray, maxiter: int = 250) -> tuple[FloatArray, float, dict[str, Any]]:
    def objective(x: FloatArray) -> float:
        lp = logpost.logp(x)
        if not np.isfinite(lp):
            return 1.0e100
        return -float(lp)

    result = minimize(
        objective,
        np.asarray(x0, dtype=float),
        method="Nelder-Mead",
        options={"maxiter": int(maxiter), "xatol": 1e-5, "fatol": 1e-5, "disp": False},
    )
    x_map = np.asarray(result.x, dtype=float)
    lp_map = logpost.logp(x_map)
    meta = {
        "success": bool(result.success),
        "message": str(result.message),
        "nit": int(getattr(result, "nit", -1)),
        "nfev": int(getattr(result, "nfev", -1)),
        "fun": float(getattr(result, "fun", np.nan)),
    }
    return x_map, float(lp_map), meta


def find_map_multistart(
    logpost: UMBridgeLogPosterior,
    starts: FloatArray,
    maxiter: int,
) -> tuple[FloatArray, float, dict[str, Any]]:
    best_x: FloatArray | None = None
    best_lp = -np.inf
    runs: list[dict[str, Any]] = []
    for k, x0 in enumerate(np.asarray(starts, dtype=float)):
        print(f"[map] start {k + 1}/{len(starts)} from theta={x0}")
        xk, lpk, meta = find_map(logpost, x0, maxiter=maxiter)
        meta = dict(meta)
        meta.update({"start_index": int(k), "theta": xk, "logp": float(lpk)})
        runs.append(meta)
        print(f"[map]   -> theta={xk}, logp={lpk:.6g}, success={meta['success']}")
        if np.isfinite(lpk) and lpk > best_lp:
            best_lp = float(lpk)
            best_x = xk.copy()
    if best_x is None:
        best_x = np.asarray(starts[0], dtype=float).copy()
        best_lp = float(logpost.logp(best_x))
    return best_x, best_lp, {"runs": runs, "n_starts": int(len(starts))}


def build_initial_training_data(
    logpost: UMBridgeLogPosterior,
    rng: np.random.Generator,
    n_init: int,
    n_local: int,
    local_sd: float,
    map_maxiter: int,
    skip_map: bool,
    n_map_starts: int,
) -> tuple[FloatArray, FloatArray, FloatArray, float, dict[str, Any]]:
    X0 = latin_normal_design(rng, n_init)
    y0 = eval_many(logpost, X0, label="initial-prior-design")

    finite_y0 = np.where(np.isfinite(y0), y0, -np.inf)
    order = np.argsort(finite_y0)[::-1]
    best_idx = int(order[0])
    x_best = X0[best_idx].copy()
    lp_best = float(y0[best_idx])
    map_meta: dict[str, Any] = {"used": False, "start": x_best, "start_logp": lp_best}

    if skip_map:
        x_map, lp_map = x_best, lp_best
    else:
        n_starts = max(1, min(int(n_map_starts), int(X0.shape[0])))
        starts = [X0[i].copy() for i in order[:n_starts]]
        starts.append(MUQ_PRIOR_MEAN.copy())
        starts_arr = np.vstack(starts)
        x_map, lp_map, opt_meta = find_map_multistart(logpost, starts_arr, maxiter=map_maxiter)
        map_meta.update({"used": True, "theta_map": x_map, "logp_map": lp_map, "optimizer": opt_meta})
        print(f"[map] selected theta_map={x_map}, logp={lp_map:.6g}")

    X_parts = [X0, x_map.reshape(1, -1)]
    y_parts = [y0, np.array([lp_map], dtype=float)]

    if n_local > 0:
        X_local = rng.normal(loc=x_map, scale=float(local_sd), size=(int(n_local), MUQ_DIM))
        y_local = eval_many(logpost, X_local, label="initial-local-design")
        X_parts.append(X_local)
        y_parts.append(y_local)

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    finite = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if not np.all(finite):
        print(f"[warning] dropping {int(np.sum(~finite))} non-finite initial evaluations")
        X, y = X[finite], y[finite]
    return X, y, x_map, lp_map, map_meta


def prepare_gp_training_data(
    X_train: FloatArray,
    y_train: FloatArray,
    args: argparse.Namespace,
) -> tuple[FloatArray, FloatArray, FloatArray, LogpTransform, dict[str, Any]]:
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float).reshape(-1)
    finite = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if not np.any(finite):
        raise RuntimeError("No finite log-posterior values available for GP training.")
    Xf = X[finite]
    yf = y[finite]

    transform = LogpTransform.fit(yf, clip_drop=args.gp_logp_drop, min_scale=args.gp_min_y_std)

    if np.isfinite(transform.clip_floor):
        informative = yf > transform.clip_floor
        clipped = ~informative
    else:
        informative = np.ones_like(yf, dtype=bool)
        clipped = np.zeros_like(yf, dtype=bool)

    keep = np.where(informative)[0].tolist()
    clipped_idx = np.where(clipped)[0]
    max_clipped = int(args.max_clipped_train)
    if max_clipped < 0:
        keep.extend(clipped_idx.tolist())
    elif clipped_idx.size > 0 and max_clipped > 0:
        order = np.argsort(yf[clipped_idx])
        keep.extend(clipped_idx[order[-max_clipped:]].tolist())

    keep_arr = np.array(sorted(set(keep)), dtype=int)
    if keep_arr.size < MUQ_DIM + 2:
        order = np.argsort(yf)
        keep_arr = np.array(sorted(set(order[-min(yf.size, max(MUQ_DIM + 2, 8)) :].tolist())), dtype=int)

    X_gp = Xf[keep_arr]
    y_gp_raw = yf[keep_arr]
    y_gp_z = transform.transform(y_gp_raw)
    meta = {
        "n_input": int(X_train.shape[0]),
        "n_finite": int(Xf.shape[0]),
        "n_used_for_gp": int(X_gp.shape[0]),
        "n_clipped_available": int(np.sum(clipped)),
        "n_clipped_used": int(np.sum(y_gp_raw <= transform.clip_floor)) if np.isfinite(transform.clip_floor) else 0,
        "transform": transform.metadata(),
        "z_min": float(np.min(y_gp_z)),
        "z_max": float(np.max(y_gp_z)),
        "z_std": float(np.std(y_gp_z, ddof=1)) if y_gp_z.size > 1 else 0.0,
    }
    return X_gp, y_gp_raw, y_gp_z, transform, meta


def rw_metropolis_fixed(
    logp_fn: Callable[[FloatArray], float],
    theta0: FloatArray,
    n_steps: int,
    proposal_cov: FloatArray,
    rng: np.random.Generator,
) -> ExactMCMCResult:
    d = theta0.size
    cov = np.asarray(proposal_cov, dtype=float)
    cov = 0.5 * (cov + cov.T) + 1e-12 * np.eye(d)
    samples = np.empty((int(n_steps), d), dtype=float)
    logp_vals = np.empty(int(n_steps), dtype=float)
    accepted = np.zeros(int(n_steps), dtype=bool)

    cur = np.asarray(theta0, dtype=float).copy()
    cur_lp = float(logp_fn(cur))
    if not np.isfinite(cur_lp):
        raise RuntimeError(f"Initial theta has non-finite log posterior: theta0={theta0}, logp={cur_lp}")

    for i in range(int(n_steps)):
        prop = rng.multivariate_normal(cur, cov)
        prop_lp = float(logp_fn(prop))
        log_alpha = prop_lp - cur_lp
        if np.isfinite(prop_lp) and np.log(rng.uniform()) < log_alpha:
            cur = prop
            cur_lp = prop_lp
            accepted[i] = True
        samples[i] = cur
        logp_vals[i] = cur_lp

    return ExactMCMCResult(samples, logp_vals, accepted, cov, float(np.mean(accepted)))


def proposal_cov_from_samples(samples: FloatArray, fallback_scale: float = 0.03, scale: float = 1.0) -> FloatArray:
    s = np.asarray(samples, dtype=float).reshape(-1, MUQ_DIM)
    if s.shape[0] <= MUQ_DIM + 2:
        return (float(fallback_scale) ** 2) * np.eye(MUQ_DIM)
    cov = np.cov(s.T)
    if cov.shape != (MUQ_DIM, MUQ_DIM) or not np.all(np.isfinite(cov)) or np.max(np.diag(cov)) <= 0:
        return (float(fallback_scale) ** 2) * np.eye(MUQ_DIM)
    cov = 0.5 * (cov + cov.T)
    vals, vecs = np.linalg.eigh(cov)
    floor = max(float(np.max(vals)) * 1e-8, 1e-12)
    vals = np.maximum(vals, floor)
    target_cov = (vecs * vals) @ vecs.T
    rw_cov = (2.38**2 / MUQ_DIM) * float(scale) * target_cov
    return 0.5 * (rw_cov + rw_cov.T) + 1e-12 * np.eye(MUQ_DIM)


def tune_then_sample_exact(
    logpost: UMBridgeLogPosterior,
    theta0: FloatArray,
    rng: np.random.Generator,
    n_tune: int,
    n_exact: int,
    proposal_scale: float,
) -> tuple[ExactMCMCResult, ExactMCMCResult | None]:
    d = theta0.size
    init_cov = (float(proposal_scale) ** 2) * np.eye(d)
    tune_result: ExactMCMCResult | None = None
    start = np.asarray(theta0, dtype=float)
    final_cov = init_cov

    if int(n_tune) > 0:
        print(f"[exact-tune] running {n_tune} RW-MH prep steps with initial scale {proposal_scale}")
        tune_result = rw_metropolis_fixed(logpost.logp, start, int(n_tune), init_cov, rng)
        tail = tune_result.samples[max(0, int(0.5 * n_tune)) :]
        fallback = float(proposal_scale) * (0.25 if tune_result.acceptance_rate < 0.05 else 1.0)
        final_cov = proposal_cov_from_samples(tail, fallback_scale=fallback)
        start = tune_result.samples[-1]
        print(
            f"[exact-tune] acceptance={tune_result.acceptance_rate:.3f}; "
            f"using proposal diag={np.diag(final_cov)}"
        )

    print(f"[exact] running {n_exact} fixed-kernel RW-MH reference steps")
    exact = rw_metropolis_fixed(logpost.logp, start, int(n_exact), final_cov, rng)
    print(f"[exact] acceptance={exact.acceptance_rate:.3f}")
    return exact, tune_result


def posterior_tail_arrays(
    samples: FloatArray,
    logp: FloatArray | None = None,
    burn_in: int = 0,
    tail_frac: float = 0.5,
) -> tuple[FloatArray, FloatArray | None, int]:
    s = np.asarray(samples, dtype=float)
    n = s.shape[0]
    if n == 0:
        return s, None if logp is None else np.asarray(logp, dtype=float), 0
    b = max(0, min(int(burn_in), n - 1))
    frac = min(max(float(tail_frac), 1e-6), 1.0)
    start_frac = int(math.floor((1.0 - frac) * n))
    start = max(b, start_frac)
    tail_s = s[start:]
    tail_lp = None if logp is None else np.asarray(logp, dtype=float)[start:]
    return tail_s, tail_lp, start


def choose_best_sample(samples: FloatArray, logp: FloatArray | None, fallback: FloatArray) -> FloatArray:
    s = np.asarray(samples, dtype=float)
    if s.size == 0:
        return np.asarray(fallback, dtype=float).copy()
    if logp is not None and np.any(np.isfinite(logp)):
        return s[int(np.nanargmax(logp))].copy()
    return np.mean(s, axis=0)


def evenly_spaced_indices(n: int, k: int) -> NDArray[np.int_]:
    if n <= 0 or k <= 0:
        return np.zeros((0,), dtype=int)
    if k >= n:
        return np.arange(n, dtype=int)
    return np.unique(np.linspace(0, n - 1, num=int(k), dtype=int))


def add_exact_tail_to_training(
    X_train: FloatArray,
    y_train: FloatArray,
    exact_result: ExactMCMCResult,
    args: argparse.Namespace,
) -> tuple[FloatArray, FloatArray, dict[str, Any], FloatArray, FloatArray | None]:
    tail_s, tail_lp, tail_start = posterior_tail_arrays(
        exact_result.samples,
        exact_result.logp,
        burn_in=args.burn_exact,
        tail_frac=args.posterior_tail_frac,
    )
    idx = evenly_spaced_indices(tail_s.shape[0], int(args.active_add_exact_to_gp))
    if idx.size > 0:
        X_aug = np.vstack([X_train, tail_s[idx]])
        y_aug = np.concatenate([y_train, np.asarray(tail_lp, dtype=float)[idx]])  # type: ignore[index]
    else:
        X_aug, y_aug = X_train, y_train
    meta = {
        "tail_start": int(tail_start),
        "tail_n": int(tail_s.shape[0]),
        "n_added_to_gp": int(idx.size),
    }
    return X_aug, y_aug, meta, tail_s, tail_lp


def run_active_da_mcmc(
    logpost: UMBridgeLogPosterior,
    X_train: FloatArray,
    y_train: FloatArray,
    theta0: FloatArray,
    args: argparse.Namespace,
    proposal_cov_override: FloatArray | None = None,
) -> tuple[Any, Any, ScalarGPSurrogate]:
    try:
        import tinyDA as tda  # type: ignore
        from gp_active_mcmc.inference import (
            ActiveMCMCModel,
            AdaptiveMetropolisShared,
            AdaptiveSubchain,
            AdaptiveSubchainControl,
            AdaptiveSubchainState,
            ChunkedMCMCConfig,
            sample_adaptive_active_chain,
        )
        from gp_active_mcmc.surrogates import MultiOutputGP
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Could not import gp_active_mcmc/tinyDA. Run from the MFDA_ActiveLearning "
            "environment after `python -m pip install -e .`."
        ) from exc

    X_gp, y_gp_raw, y_gp_z, transform, gp_meta = prepare_gp_training_data(X_train, y_train, args)
    print(
        "[active] fitting scalar MultiOutputGP on "
        f"{X_gp.shape[0]}/{X_train.shape[0]} points; "
        f"raw logp {np.min(y_gp_raw):.3g}..{np.max(y_gp_raw):.3g}; "
        f"clip_floor={transform.clip_floor:.3g}; z {np.min(y_gp_z):.3g}..{np.max(y_gp_z):.3g}"
    )
    gp = MultiOutputGP(
        X_train=X_gp,
        Y_train=y_gp_z.reshape(-1, 1),
        kernel=args.gp_kernel,
        ard=not args.no_ard,
        noise_variance=float(args.gp_noise),
        update_every=int(args.gp_update_every),
        n_retrain_max=int(args.gp_retrain_max),
    )
    surrogate = ScalarGPSurrogate(
        gp=gp,
        transform=transform,
        variance_floor=float(args.gp_var_floor),
        predict_clip=not bool(args.no_gp_predict_clip),
        training_metadata=gp_meta,
    )

    adaptive_policy = AdaptiveSubchain(
        state=AdaptiveSubchainState(subchain_length=int(args.initial_subchain)),
        control=AdaptiveSubchainControl(
            update_every=int(args.subchain_update_every),
            target_error=float(args.target_error),
            min_subchain=int(args.min_subchain),
            max_subchain=int(args.max_subchain),
            grow_factor=float(args.grow_factor),
            shrink_factor=float(args.shrink_factor),
        ),
    )
    model = ActiveMCMCModel(
        lf_model=surrogate,
        hf_model=logpost,
        gamma_threshold=float(args.gamma),
        adaptive=adaptive_policy,
    )

    flat_prior = FlatPrior(dim=MUQ_DIM, rvs_center=MUQ_PRIOR_MEAN, rvs_scale=MUQ_PRIOR_SD)
    coarse_floor = transform.clip_floor if args.coarse_like_floor else None
    coarse_like = LogDensityLike(variance_penalty=float(args.coarse_var_penalty), floor=coarse_floor)
    fine_like = LogDensityLike(variance_penalty=0.0, floor=None)
    posterior = [
        tda.Posterior(flat_prior, coarse_like, model.coarse),
        tda.Posterior(flat_prior, fine_like, model.fine),
    ]

    if proposal_cov_override is None:
        proposal_cov = (float(args.active_proposal_scale) ** 2) * np.eye(MUQ_DIM)
    else:
        proposal_cov = np.asarray(proposal_cov_override, dtype=float)
    proposal_cov = 0.5 * (proposal_cov + proposal_cov.T) + 1e-12 * np.eye(MUQ_DIM)
    print(f"[active] proposal diag={np.diag(proposal_cov)}")

    proposal = AdaptiveMetropolisShared(
        C0=proposal_cov,
        period=int(args.proposal_period),
        share_across_deepcopy=True,
        adaptive=True,
        sd=float(args.proposal_sd),
    )

    print(
        f"[active] running adaptive DA-MCMC with n_coarse_evals={args.n_active}, "
        f"chunk_size={args.active_chunk_size}, gamma={args.gamma}, chain_key={args.chain_key}"
    )
    result = sample_adaptive_active_chain(
        model=model,
        posterior=posterior,
        proposal=proposal,
        n_coarse_evals=int(args.n_active),
        initial_parameters=np.asarray(theta0, dtype=float),
        chain_key=str(args.chain_key),
        config=ChunkedMCMCConfig(chain_key=str(args.chain_key), chunk_size=int(args.active_chunk_size)),
        store_coarse_chain=not bool(args.no_store_coarse_chain),
        n_chains=1,
        force_sequential=True,
    )
    summary = result.chain.summary(burn_in=min(int(args.burn_active), result.chain.n_steps))
    print(f"[active] summary={summary}")
    return result, model, surrogate


def summarise_samples(samples: FloatArray, burn_in: int = 0) -> dict[str, Any]:
    b = max(0, min(int(burn_in), samples.shape[0] - 1)) if samples.shape[0] > 1 else 0
    s = np.asarray(samples[b:], dtype=float)
    return {
        "n": int(s.shape[0]),
        "burn_in": int(b),
        "mean": np.mean(s, axis=0),
        "std": np.std(s, axis=0, ddof=1) if s.shape[0] > 1 else np.zeros(s.shape[1]),
        "q025": np.quantile(s, 0.025, axis=0),
        "q500": np.quantile(s, 0.500, axis=0),
        "q975": np.quantile(s, 0.975, axis=0),
        "cov": np.cov(s.T) if s.shape[0] > 1 else np.zeros((s.shape[1], s.shape[1])),
    }


def compare_samples(
    exact_samples: FloatArray,
    active_samples: FloatArray,
    burn_exact: int,
    burn_active: int,
) -> dict[str, Any]:
    be = max(0, min(int(burn_exact), exact_samples.shape[0] - 1))
    ba = max(0, min(int(burn_active), active_samples.shape[0] - 1))
    e = exact_samples[be:]
    a = active_samples[ba:]
    exact_std = np.std(e, axis=0, ddof=1)
    mean_diff = np.mean(a, axis=0) - np.mean(e, axis=0)
    std_diff = np.std(a, axis=0, ddof=1) - exact_std
    out: dict[str, Any] = {
        "exact_n": int(e.shape[0]),
        "active_n": int(a.shape[0]),
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "mean_abs_diff_over_exact_std": np.abs(mean_diff) / np.maximum(exact_std, 1e-15),
        "mean_rmse": float(np.sqrt(np.mean(mean_diff**2))),
        "std_rmse": float(np.sqrt(np.mean(std_diff**2))),
        "per_dim": [],
    }
    for j in range(MUQ_DIM):
        ks = ks_2samp(e[:, j], a[:, j])
        out["per_dim"].append(
            {
                "dim": int(j),
                "wasserstein": float(wasserstein_distance(e[:, j], a[:, j])),
                "ks_statistic": float(ks.statistic),
                "ks_pvalue": float(ks.pvalue),
            }
        )
    return out


def validate_scalar_surrogate(
    logpost: UMBridgeLogPosterior,
    surrogate: ScalarGPSurrogate,
    rng: np.random.Generator,
    center: FloatArray,
    n_val: int,
    local_sd: float,
    source: str = "posterior",
    posterior_samples: FloatArray | None = None,
) -> tuple[dict[str, Any], FloatArray, FloatArray, FloatArray, FloatArray]:
    n_val = int(n_val)
    if n_val <= 0:
        empty = np.zeros((0, MUQ_DIM), dtype=float)
        return {"n_val": 0}, empty, np.array([]), np.array([]), np.array([])

    if source == "posterior" and posterior_samples is not None and posterior_samples.shape[0] > 0:
        idx = rng.choice(posterior_samples.shape[0], size=n_val, replace=posterior_samples.shape[0] < n_val)
        X_val = np.asarray(posterior_samples[idx], dtype=float)
        if float(local_sd) > 0.0:
            X_val = X_val + rng.normal(scale=float(local_sd), size=X_val.shape)
    elif source == "global":
        X_val = latin_normal_design(rng, n_val)
    elif source == "mixed":
        n_post = n_val // 2
        if posterior_samples is not None and posterior_samples.shape[0] > 0:
            idx = rng.choice(posterior_samples.shape[0], size=n_post, replace=posterior_samples.shape[0] < n_post)
            X_post = np.asarray(posterior_samples[idx], dtype=float)
            X_post = X_post + rng.normal(scale=float(local_sd), size=X_post.shape)
        else:
            X_post = rng.normal(loc=center, scale=float(local_sd), size=(n_post, MUQ_DIM))
        X_loc = rng.normal(loc=center, scale=float(local_sd), size=(n_val - n_post, MUQ_DIM))
        X_val = np.vstack([X_post, X_loc])
    else:
        X_val = rng.normal(loc=center, scale=float(local_sd), size=(n_val, MUQ_DIM))

    y_true = eval_many(logpost, X_val, label="surrogate-validation")
    y_mean = np.empty(n_val, dtype=float)
    y_var = np.empty(n_val, dtype=float)
    for i, th in enumerate(X_val):
        mu, var = surrogate.predict(th)
        y_mean[i] = float(mu[0])
        y_var[i] = max(float(var[0]), 0.0)

    finite = np.isfinite(y_true) & np.isfinite(y_mean)
    if not np.any(finite):
        metrics = {"n_val": int(n_val), "n_finite": 0}
    else:
        err = y_mean[finite] - y_true[finite]
        std = np.sqrt(np.maximum(y_var[finite], 0.0))
        corr = np.corrcoef(np.abs(err), std)[0, 1] if finite.sum() > 2 and np.std(std) > 0 else np.nan
        metrics = {
            "n_val": int(n_val),
            "n_finite": int(finite.sum()),
            "source": source,
            "rmse_logp": float(np.sqrt(np.mean(err**2))),
            "mae_logp": float(np.mean(np.abs(err))),
            "median_abs_error_logp": float(np.median(np.abs(err))),
            "mean_pred_std_logp": float(np.mean(std)),
            "coverage_1sigma": float(np.mean(np.abs(err) <= std)),
            "coverage_2sigma": float(np.mean(np.abs(err) <= 2.0 * std)),
            "corr_abs_error_pred_std": float(corr),
        }
    return metrics, X_val, y_true, y_mean, y_var


def make_plots(
    outdir: Path,
    exact_samples: FloatArray | None,
    active_samples: FloatArray | None,
    used_hf: NDArray[np.bool_] | None,
    val_true: FloatArray | None,
    val_mean: FloatArray | None,
    val_var: FloatArray | None,
    burn_exact: int,
    burn_active: int,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[plots] matplotlib not available, skipping plots: {exc}")
        return

    labels = [r"$m_1$", r"$m_2$", r"$m_3$"]

    if exact_samples is not None or active_samples is not None:
        fig, axes = plt.subplots(1, MUQ_DIM, figsize=(14, 3.5))
        for j, ax in enumerate(axes):
            if exact_samples is not None and exact_samples.shape[0] > 0:
                be = max(0, min(int(burn_exact), exact_samples.shape[0] - 1))
                ax.hist(exact_samples[be:, j], bins=45, density=True, alpha=0.45, label="exact")
            if active_samples is not None and active_samples.shape[0] > 0:
                ba = max(0, min(int(burn_active), active_samples.shape[0] - 1))
                ax.hist(active_samples[ba:, j], bins=45, density=True, alpha=0.45, label="active DA")
            ax.set_xlabel(labels[j])
            ax.set_ylabel("density")
            ax.legend()
        fig.suptitle("Marginal posterior comparison")
        fig.tight_layout()
        fig.savefig(outdir / "posterior_marginals.png", dpi=200)
        plt.close(fig)

    if exact_samples is not None and active_samples is not None:
        be = max(0, min(int(burn_exact), exact_samples.shape[0] - 1))
        ba = max(0, min(int(burn_active), active_samples.shape[0] - 1))
        e = exact_samples[be:]
        a = active_samples[ba:]
        e_plot = e[:: max(1, e.shape[0] // 3000)]
        a_plot = a[:: max(1, a.shape[0] // 3000)]
        pairs = [(0, 1), (0, 2), (1, 2)]
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for ax, (i, j) in zip(axes, pairs, strict=True):
            ax.scatter(e_plot[:, i], e_plot[:, j], s=5, alpha=0.25, label="exact")
            ax.scatter(a_plot[:, i], a_plot[:, j], s=5, alpha=0.25, label="active DA")
            ax.set_xlabel(labels[i])
            ax.set_ylabel(labels[j])
            ax.legend(markerscale=2)
        fig.suptitle("Pairwise posterior samples")
        fig.tight_layout()
        fig.savefig(outdir / "posterior_pairs.png", dpi=200)
        plt.close(fig)

    if active_samples is not None and active_samples.shape[0] > 0:
        fig, axes = plt.subplots(MUQ_DIM, 1, figsize=(12, 7), sharex=True)
        for j, ax in enumerate(axes):
            ax.plot(active_samples[:, j], linewidth=0.8)
            ax.set_ylabel(labels[j])
        axes[-1].set_xlabel("active DA sample index")
        fig.suptitle("Active DA-MCMC trace")
        fig.tight_layout()
        fig.savefig(outdir / "active_trace.png", dpi=200)
        plt.close(fig)

    if used_hf is not None and used_hf.size > 0:
        cum = np.cumsum(used_hf.astype(float)) / np.arange(1, used_hf.size + 1)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(cum, linewidth=1.5)
        ax.set_xlabel("coarse-chain step")
        ax.set_ylabel("cumulative HF fraction")
        ax.set_title("High-fidelity usage during active DA-MCMC")
        fig.tight_layout()
        fig.savefig(outdir / "hf_fraction.png", dpi=200)
        plt.close(fig)

    if val_true is not None and val_mean is not None and val_var is not None and val_true.size > 0:
        finite = np.isfinite(val_true) & np.isfinite(val_mean)
        if np.any(finite):
            vt = val_true[finite]
            vm = val_mean[finite]
            vv = val_var[finite]
            std = np.sqrt(np.maximum(vv, 0.0))
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.errorbar(vt, vm, yerr=2.0 * std, fmt="o", alpha=0.55, markersize=4)
            lo = float(np.nanmin([np.nanmin(vt), np.nanmin(vm)]))
            hi = float(np.nanmax([np.nanmax(vt), np.nanmax(vm)]))
            if not np.isclose(lo, hi):
                ax.plot([lo, hi], [lo, hi], linestyle="--")
            ax.set_xlabel("exact log posterior")
            ax.set_ylabel("GP predicted log posterior")
            ax.set_title("Scalar GP surrogate validation")
            fig.tight_layout()
            fig.savefig(outdir / "surrogate_validation.png", dpi=200)
            plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run UM-Bridge muq-beam with robust MFDA Active-DA-MCMC")
    p.add_argument("--url", default="http://localhost:4243")
    p.add_argument("--model-name", default="posterior")
    p.add_argument("--mode", choices=["both", "exact", "active"], default="both")
    p.add_argument("--outdir", default="results/muq_beam_mfda_fixed")
    p.add_argument("--seed", type=int, default=123)

    p.add_argument("--n-init", type=int, default=60)
    p.add_argument("--n-local", type=int, default=50)
    p.add_argument("--local-sd", type=float, default=0.12)
    p.add_argument("--map-maxiter", type=int, default=300)
    p.add_argument("--n-map-starts", type=int, default=4)
    p.add_argument("--skip-map", action="store_true")

    p.add_argument("--n-tune-exact", type=int, default=3000)
    p.add_argument("--n-exact", type=int, default=10000)
    p.add_argument("--exact-proposal-scale", type=float, default=0.04)
    p.add_argument("--burn-exact", type=int, default=0)
    p.add_argument("--posterior-tail-frac", type=float, default=0.5)

    p.add_argument("--n-active", type=int, default=6000)
    p.add_argument("--active-chunk-size", type=int, default=250)
    p.add_argument("--burn-active", type=int, default=500)
    p.add_argument("--gamma", type=float, default=3.0, help="HF trigger in raw logp std units")
    p.add_argument("--target-error", type=float, default=2.0)
    p.add_argument("--initial-subchain", type=int, default=10)
    p.add_argument("--subchain-update-every", type=int, default=5)
    p.add_argument("--min-subchain", type=int, default=1)
    p.add_argument("--max-subchain", type=int, default=250)
    p.add_argument("--grow-factor", type=float, default=1.5)
    p.add_argument("--shrink-factor", type=float, default=0.5)
    p.add_argument("--coarse-var-penalty", type=float, default=0.0)
    p.add_argument("--coarse-like-floor", action="store_true", default=True)
    p.add_argument("--no-coarse-like-floor", dest="coarse_like_floor", action="store_false")

    p.add_argument("--active-seed-from-exact", action="store_true", default=True)
    p.add_argument("--no-active-seed-from-exact", dest="active_seed_from_exact", action="store_false")
    p.add_argument("--active-add-exact-to-gp", type=int, default=200)
    p.add_argument("--active-use-exact-cov", action="store_true", default=True)
    p.add_argument("--no-active-use-exact-cov", dest="active_use_exact_cov", action="store_false")
    p.add_argument("--active-cov-scale", type=float, default=1.0)

    p.add_argument("--gp-kernel", choices=["rbf", "matern32", "matern52"], default="matern52")
    p.add_argument("--no-ard", action="store_true")
    p.add_argument("--gp-noise", type=float, default=1e-6)
    p.add_argument("--gp-update-every", type=int, default=25)
    p.add_argument("--gp-retrain-max", type=int, default=8)
    p.add_argument("--gp-var-floor", type=float, default=1e-8)
    p.add_argument("--gp-logp-drop", type=float, default=500.0)
    p.add_argument("--gp-min-y-std", type=float, default=1.0)
    p.add_argument("--max-clipped-train", type=int, default=20)
    p.add_argument("--no-gp-predict-clip", action="store_true")

    p.add_argument("--active-proposal-scale", type=float, default=0.03)
    p.add_argument("--proposal-period", type=int, default=100)
    p.add_argument("--proposal-sd", type=float, default=1.0)
    p.add_argument("--chain-key", default="chain_coarse_0")
    p.add_argument("--no-store-coarse-chain", action="store_true")

    p.add_argument("--n-val", type=int, default=80)
    p.add_argument("--val-local-sd", type=float, default=0.003)
    p.add_argument("--val-source", choices=["posterior", "local", "mixed", "global"], default="posterior")
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rng = np.random.default_rng(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=== UM-Bridge muq-beam + robust MFDA/gp_active_mcmc benchmark ===")
    print(f"outdir={outdir}")

    logpost = UMBridgeLogPosterior(args.url, args.model_name)
    model_info = logpost.check()
    print(f"[umbridge] connected to {args.url}/{args.model_name}: {model_info}")
    calls_start = logpost.n_calls

    X_initial, y_initial, theta_map, lp_map, map_meta = build_initial_training_data(
        logpost=logpost,
        rng=rng,
        n_init=args.n_init,
        n_local=args.n_local,
        local_sd=args.local_sd,
        map_maxiter=args.map_maxiter,
        skip_map=args.skip_map,
        n_map_starts=args.n_map_starts,
    )
    print(
        f"[training-initial] X={X_initial.shape}, best logp={float(np.max(y_initial)):.6g}, "
        f"theta_start={theta_map}"
    )

    exact_result: ExactMCMCResult | None = None
    tune_result: ExactMCMCResult | None = None
    active_result: Any | None = None
    active_model: Any | None = None
    surrogate: ScalarGPSurrogate | None = None
    tail_samples_for_validation: FloatArray | None = None
    active_init_meta: dict[str, Any] = {}

    if args.mode in {"both", "exact"}:
        calls_before = logpost.n_calls
        exact_result, tune_result = tune_then_sample_exact(
            logpost=logpost,
            theta0=theta_map,
            rng=rng,
            n_tune=args.n_tune_exact,
            n_exact=args.n_exact,
            proposal_scale=args.exact_proposal_scale,
        )
        print(f"[exact] exact baseline HF calls: {logpost.n_calls - calls_before}")

    X_active_train = X_initial.copy()
    y_active_train = y_initial.copy()
    theta_active0 = theta_map.copy()
    active_proposal_cov: FloatArray | None = None

    if args.mode in {"both", "active"}:
        if exact_result is not None:
            X_active_train, y_active_train, warm_meta, tail_s, tail_lp = add_exact_tail_to_training(
                X_initial, y_initial, exact_result, args
            )
            tail_samples_for_validation = tail_s
            active_init_meta.update(warm_meta)
            if bool(args.active_seed_from_exact):
                theta_active0 = choose_best_sample(tail_s, tail_lp, fallback=theta_map)
                active_init_meta["theta_active0_source"] = "best_exact_tail_logp"
            else:
                active_init_meta["theta_active0_source"] = "map"
            if bool(args.active_use_exact_cov):
                active_proposal_cov = proposal_cov_from_samples(
                    tail_s,
                    fallback_scale=float(args.active_proposal_scale),
                    scale=float(args.active_cov_scale),
                )
                active_init_meta["active_proposal_cov_source"] = "exact_tail"
                active_init_meta["active_proposal_cov_diag"] = np.diag(active_proposal_cov)
        else:
            active_init_meta["theta_active0_source"] = "map_no_exact_available"

        print(f"[active-init] theta_active0={theta_active0}")
        print(f"[active-training] X={X_active_train.shape}, best logp={float(np.max(y_active_train)):.6g}")
        calls_before = logpost.n_calls
        active_result, active_model, surrogate = run_active_da_mcmc(
            logpost=logpost,
            X_train=X_active_train,
            y_train=y_active_train,
            theta0=theta_active0,
            args=args,
            proposal_cov_override=active_proposal_cov,
        )
        print(f"[active] exact HF calls during active run: {logpost.n_calls - calls_before}")

    val_metrics: dict[str, Any] = {"n_val": 0}
    X_val = np.zeros((0, MUQ_DIM), dtype=float)
    y_val_true = np.array([], dtype=float)
    y_val_mean = np.array([], dtype=float)
    y_val_var = np.array([], dtype=float)
    if surrogate is not None and int(args.n_val) > 0:
        val_metrics, X_val, y_val_true, y_val_mean, y_val_var = validate_scalar_surrogate(
            logpost=logpost,
            surrogate=surrogate,
            rng=rng,
            center=theta_active0,
            n_val=args.n_val,
            local_sd=args.val_local_sd,
            source=args.val_source,
            posterior_samples=tail_samples_for_validation,
        )
        print(f"[surrogate-validation] {val_metrics}")

    active_samples = None if active_result is None else np.asarray(active_result.chain.samples, dtype=float)
    used_hf = None
    subchain_length = None
    if active_result is not None:
        used_hf = np.asarray(active_result.chain.extras.used_hf, dtype=bool)
        subchain_length = active_result.chain.extras.subchain_length

    metrics: dict[str, Any] = {
        "args": vars(args),
        "umbridge": model_info,
        "muq_prior_mean": MUQ_PRIOR_MEAN,
        "muq_prior_sd": MUQ_PRIOR_SD,
        "initial_training": {
            "n_train": int(X_initial.shape[0]),
            "theta_map_or_start": theta_map,
            "logp_map_or_start": float(lp_map),
            "map_meta": map_meta,
            "logp_train_min": float(np.min(y_initial)),
            "logp_train_max": float(np.max(y_initial)),
        },
        "active_initialisation": active_init_meta,
        "surrogate_validation": val_metrics,
        "hf_calls_total": int(logpost.n_calls - calls_start),
    }

    if exact_result is not None:
        metrics["exact"] = {
            "summary": summarise_samples(exact_result.samples, burn_in=args.burn_exact),
            "acceptance_rate": exact_result.acceptance_rate,
            "proposal_cov": exact_result.proposal_cov,
        }
        if tune_result is not None:
            metrics["exact_tune"] = {
                "acceptance_rate": tune_result.acceptance_rate,
                "proposal_cov": tune_result.proposal_cov,
            }

    if active_result is not None and active_samples is not None:
        metrics["active"] = {
            "summary": summarise_samples(active_samples, burn_in=args.burn_active),
            "chain_summary_raw": active_result.chain.summary(
                burn_in=min(int(args.burn_active), active_result.chain.n_steps)
            ),
            "metadata": active_result.metadata,
            "n_train_after_active": None if surrogate is None else surrogate.n_train,
            "gp_training_metadata": None if surrogate is None else surrogate.training_metadata,
            "n_skipped_gp_updates": None if surrogate is None else surrogate.n_skipped_updates,
            "hf_fraction_flagged": None if used_hf is None or used_hf.size == 0 else float(np.mean(used_hf)),
            "n_hf_calls_flagged": None if used_hf is None else int(np.sum(used_hf)),
        }
        if subchain_length is not None and len(subchain_length) > 0:
            metrics["active"]["subchain_length_mean"] = float(np.mean(subchain_length))
            metrics["active"]["subchain_length_last"] = int(subchain_length[-1])
        if active_model is not None and getattr(active_model, "adaptive", None) is not None:
            state = active_model.adaptive.state
            metrics["active"]["adaptive_hf_errors"] = np.asarray(state.hf_errors, dtype=float)

    if exact_result is not None and active_samples is not None:
        metrics["comparison_exact_vs_active"] = compare_samples(
            exact_result.samples,
            active_samples,
            burn_exact=args.burn_exact,
            burn_active=args.burn_active,
        )
        print(f"[comparison] {metrics['comparison_exact_vs_active']}")

    npz_path = outdir / "samples_and_diagnostics.npz"
    np.savez_compressed(
        npz_path,
        X_train_initial=X_initial,
        y_train_initial=y_initial,
        X_train_active=X_active_train,
        y_train_active=y_active_train,
        theta_map=theta_map,
        theta_active0=theta_active0,
        exact_samples=np.zeros((0, MUQ_DIM)) if exact_result is None else exact_result.samples,
        exact_logp=np.array([]) if exact_result is None else exact_result.logp,
        exact_accepted=np.array([]) if exact_result is None else exact_result.accepted,
        active_samples=np.zeros((0, MUQ_DIM)) if active_samples is None else active_samples,
        active_used_hf=np.array([]) if used_hf is None else used_hf,
        active_subchain_length=np.array([]) if subchain_length is None else subchain_length,
        X_val=X_val,
        y_val_true=y_val_true,
        y_val_mean=y_val_mean,
        y_val_var=y_val_var,
    )

    metrics_path = outdir / "metrics.json"
    metrics_path.write_text(json.dumps(as_jsonable(metrics), indent=2), encoding="utf-8")

    if not args.no_plots:
        make_plots(
            outdir=outdir,
            exact_samples=None if exact_result is None else exact_result.samples,
            active_samples=active_samples,
            used_hf=used_hf,
            val_true=y_val_true,
            val_mean=y_val_mean,
            val_var=y_val_var,
            burn_exact=args.burn_exact,
            burn_active=args.burn_active,
        )

    print("=== done ===")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved arrays:  {npz_path}")
    if not args.no_plots:
        print(f"Saved plots in: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
