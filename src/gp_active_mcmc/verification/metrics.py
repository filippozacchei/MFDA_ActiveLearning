"""Posterior-quality/cost metrics and R-hat/ESS convergence diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from gp_active_mcmc.inference import MCMCChain
from gp_active_mcmc.utils.mcmc import posterior_rmse
from gp_active_mcmc.verification.problem import Problem

FloatArray = NDArray[np.float64]

__all__ = [
    "adaptive_stm_coarse_eval_units",
    "adaptive_stm_cumulative_coarse_evals",
    "adaptive_stm_full_resolution_trace",
    "adaptive_stm_multichain_trace",
    "effective_burn_in",
    "find_burn_in_via_rhat",
    "gaussian_kl",
    "gaussian_wasserstein2",
    "multichain_diagnostics",
    "pooled_summarize",
    "prepare_trace_data",
    "summarize",
]


# effective_burn_in / adaptive_stm_* helpers: run_adaptive_stm's chain mixes two row
# currencies (one row per coarse eval during the adaptive phase, one row per DA block
# during frozen production), so these reconstruct burn-in and a common x-axis in
# coarse-evaluation units from its metadata dict (meta_adaptive_stm).


def effective_burn_in(method: str, *, meta_adaptive_stm: dict[str, Any] | None = None, default: int = 0) -> int:
    """Burn-in to use for posterior summaries/plots. `adaptive_stm`'s adaptive phase is
    non-stationary (the surrogate is still updating), so it's discarded in full via
    `meta_adaptive_stm["n_adapt_samples"]` (required in that case) regardless of
    `default`; every other method uses `default` as-is."""
    if method == "adaptive_stm":
        if meta_adaptive_stm is None:
            raise ValueError("meta_adaptive_stm is required to compute burn-in for 'adaptive_stm'.")
        return int(meta_adaptive_stm["n_adapt_samples"])
    return default


def adaptive_stm_coarse_eval_units(meta_adaptive_stm: dict[str, Any]) -> int:
    """True coarse-evaluation cost of `run_adaptive_stm`'s returned chain --
    `chain.n_steps` undercounts this during production, where each row is a whole DA
    block of `frozen_subsampling_rate` coarse evaluations, not one. Use as
    `summarize`'s/`pooled_summarize`'s `n_coarse_eval_units` override."""
    n_adapt_coarse_evals = int(meta_adaptive_stm["adapt_metadata"]["coarse_evals_used"])
    if meta_adaptive_stm["phase"] == "adapt_only":
        return n_adapt_coarse_evals
    production_iterations = int(meta_adaptive_stm["production_metadata"]["iterations"])
    frozen_rate = int(meta_adaptive_stm["frozen_subsampling_rate"])
    return n_adapt_coarse_evals + production_iterations * frozen_rate


def adaptive_stm_cumulative_coarse_evals(meta_adaptive_stm: dict[str, Any]) -> FloatArray:
    """Per-row cumulative coarse-evaluation count for `adaptive_stm`'s full chain --
    the x-axis a plot should use instead of raw row index, since each production row
    is a whole DA block, not one coarse eval (see `adaptive_stm_coarse_eval_units`)."""
    n_adapt = int(meta_adaptive_stm["n_adapt_samples"])
    adapt_x = np.arange(1, n_adapt + 1, dtype=float)
    if meta_adaptive_stm["phase"] == "adapt_only":
        return adapt_x
    n_adapt_coarse_evals = int(meta_adaptive_stm["adapt_metadata"]["coarse_evals_used"])
    frozen_rate = int(meta_adaptive_stm["frozen_subsampling_rate"])
    n_production = int(meta_adaptive_stm["n_production_samples"])
    production_x = n_adapt_coarse_evals + frozen_rate * np.arange(1, n_production + 1, dtype=float)
    return np.concatenate([adapt_x, production_x])


def adaptive_stm_full_resolution_trace(
    chain: MCMCChain, meta_adaptive_stm: dict[str, Any]
) -> tuple[FloatArray, FloatArray, NDArray[np.bool_]]:
    """Full-resolution ``(x, samples, used_hf)`` trajectory for `adaptive_stm`
    (chain + metadata from `run_adaptive_stm`), across both phases. `chain` alone is
    one row per DA block during production, hiding the cheap intra-block coarse
    steps; this stitches in the production phase's diagnostic chain
    (`meta_adaptive_stm["production_diagnostic_chain"]`) to recover the full-resolution
    path for a `used_hf`-colored trace plot. `x` shares units with
    `adaptive_stm_cumulative_coarse_evals`."""
    n_adapt = int(meta_adaptive_stm["n_adapt_samples"])
    adapt_samples = chain.samples[:n_adapt]
    adapt_used_hf = (
        chain.extras.used_hf[:n_adapt] if chain.extras.used_hf is not None else np.zeros(n_adapt, dtype=bool)
    )
    adapt_x = np.arange(1, n_adapt + 1, dtype=float)

    diagnostic_chain = meta_adaptive_stm.get("production_diagnostic_chain")
    if diagnostic_chain is None:
        return adapt_x, adapt_samples, adapt_used_hf

    n_adapt_coarse_evals = int(meta_adaptive_stm["adapt_metadata"]["coarse_evals_used"])
    prod_samples = diagnostic_chain.samples
    prod_used_hf = (
        diagnostic_chain.extras.used_hf
        if diagnostic_chain.extras.used_hf is not None
        else np.zeros(prod_samples.shape[0], dtype=bool)
    )
    prod_x = n_adapt_coarse_evals + np.arange(1, prod_samples.shape[0] + 1, dtype=float)

    x = np.concatenate([adapt_x, prod_x])
    samples = np.vstack([adapt_samples, prod_samples])
    used_hf = np.concatenate([adapt_used_hf, prod_used_hf])
    return x, samples, used_hf


def adaptive_stm_multichain_trace(
    adapt_chain: MCMCChain, adapt_meta: dict[str, Any], production_chain: MCMCChain, frozen_rate: int
) -> tuple[FloatArray, FloatArray, NDArray[np.bool_]]:
    """Stitched ``(x, samples, used_hf)`` trace for one `adaptive_stm` replicate from
    `run_convergence_driven_comparison` -- the multi-chain analog of
    `adaptive_stm_full_resolution_trace`. The production segment here is at DA-block
    resolution (one row per block, not per coarse eval): the chunked production loop
    doesn't request the intra-block diagnostic chain every round. `x` stays in
    coarse-evaluation-iteration units so it lines up with the other methods' traces.
    """
    n_adapt = adapt_chain.n_steps
    adapt_x = np.arange(1, n_adapt + 1, dtype=float)
    adapt_used_hf = (
        adapt_chain.extras.used_hf if adapt_chain.extras.used_hf is not None else np.zeros(n_adapt, dtype=bool)
    )

    n_adapt_coarse_evals = int(adapt_meta["adapt_coarse_evals_used"])
    n_production = production_chain.n_steps
    production_x = n_adapt_coarse_evals + frozen_rate * np.arange(1, n_production + 1, dtype=float)
    production_used_hf = (
        production_chain.extras.used_hf
        if production_chain.extras.used_hf is not None
        else np.zeros(n_production, dtype=bool)
    )

    x = np.concatenate([adapt_x, production_x])
    samples = np.vstack([adapt_chain.samples, production_chain.samples])
    used_hf = np.concatenate([adapt_used_hf, production_used_hf])
    return x, samples, used_hf


def prepare_trace_data(
    chains_by_method: dict[str, list[MCMCChain]],
    posterior: dict[str, Any],
    *,
    full_resolution_methods: tuple[str, ...],
    adaptive_stm_method: str = "adaptive_stm",
    adaptive_stm_adapt_key: str | None = None,
) -> tuple[dict[str, list[tuple[FloatArray, FloatArray, NDArray[np.bool_]]]], dict[str, float]]:
    """Per-replicate ``(x, samples, used_hf)`` traces and each method's R-hat-validated
    burn-in, both `chains_by_method`/`posterior` (as returned by
    `run_convergence_driven_comparison`), converted to a shared x-axis
    (coarse-evaluation-iteration units). `x` is row index for `full_resolution_methods`
    (already one row per coarse eval, e.g. `("hf_only", "adaptive_surrogate_mcmc")`);
    `adaptive_stm_method` is stitched via `adaptive_stm_multichain_trace` instead, with
    its burn-in converted using the shared adaptive phase's coarse-eval cost and the
    frozen `subsampling_rate`. `adaptive_stm_adapt_key` defaults to
    ``f"{adaptive_stm_method}_adapt"``.
    """
    if adaptive_stm_adapt_key is None:
        adaptive_stm_adapt_key = f"{adaptive_stm_method}_adapt"

    traces: dict[str, list[tuple[FloatArray, FloatArray, NDArray[np.bool_]]]] = {}
    burn_ins_x: dict[str, float] = {}
    for name in full_resolution_methods:
        traces[name] = [
            (
                np.arange(1, c.n_steps + 1, dtype=float),
                c.samples,
                c.extras.used_hf if c.extras.used_hf is not None else np.zeros(c.n_steps, dtype=bool),
            )
            for c in chains_by_method[name]
        ]
        burn_ins_x[name] = posterior[name]["burn_in"] or 0

    adapt_metas = posterior[adaptive_stm_method]["adapt_metas"]
    coarse_evals_per_chain = posterior[adaptive_stm_method]["coarse_evals_per_chain"]
    adaptive_stm_traces = []
    frozen_rate = 1
    for adapt_chain, adapt_meta, production_chain, production_cost in zip(
        chains_by_method[adaptive_stm_adapt_key],
        adapt_metas,
        chains_by_method[adaptive_stm_method],
        coarse_evals_per_chain,
        strict=True,
    ):
        # frozen_rate is exactly production_cost / (n genuine new production rows):
        # subsampling_rate is constant throughout production (frozen), and
        # coarse_evals_per_chain already sums iterations*subsampling_rate per round.
        frozen_rate = production_cost // (production_chain.n_steps - 1) if production_chain.n_steps > 1 else 1
        adaptive_stm_traces.append(adaptive_stm_multichain_trace(adapt_chain, adapt_meta, production_chain, frozen_rate))
    traces[adaptive_stm_method] = adaptive_stm_traces
    n_adapt_coarse_evals = int(adapt_metas[0]["adapt_coarse_evals_used"])
    adaptive_stm_burn_in_row = posterior[adaptive_stm_method]["burn_in"] or 0
    burn_ins_x[adaptive_stm_method] = n_adapt_coarse_evals + frozen_rate * adaptive_stm_burn_in_row

    return traces, burn_ins_x


# Distributional posterior-quality metrics, beyond RMSE-to-truth (which only checks
# the mean against a single point, saying nothing about spread/shape): compare a
# pooled posterior against a reference (typically the R-hat-validated HF-only
# posterior) under a Gaussian approximation of both.


def gaussian_wasserstein2(mean_a: FloatArray, cov_a: FloatArray, mean_b: FloatArray, cov_b: FloatArray) -> float:
    """Closed-form 2-Wasserstein distance between ``N(mean_a, cov_a)`` and
    ``N(mean_b, cov_b))``: symmetric and a true metric (unlike KL), capturing both a
    location mismatch (mean term) and a shape/spread mismatch (trace term), in the
    parameters' own units.

    .. math::

        W_2^2 = \\lVert \\mathrm{mean}_a - \\mathrm{mean}_b \\rVert^2
        + \\mathrm{tr}\\left(\\mathrm{cov}_a + \\mathrm{cov}_b
        - 2 \\left(\\mathrm{cov}_b^{1/2} \\mathrm{cov}_a \\mathrm{cov}_b^{1/2}\\right)^{1/2}\\right).
    """
    from scipy.linalg import sqrtm

    mean_a = np.asarray(mean_a, dtype=float)
    mean_b = np.asarray(mean_b, dtype=float)
    cov_a = np.asarray(cov_a, dtype=float)
    cov_b = np.asarray(cov_b, dtype=float)

    diff = mean_a - mean_b
    cov_b_sqrt = sqrtm(cov_b)
    cross_term = sqrtm(cov_b_sqrt @ cov_a @ cov_b_sqrt)
    cross_term = np.real(cross_term)  # discard negligible complex parts from numerical sqrtm

    w2_sq = float(diff @ diff + np.trace(cov_a + cov_b - 2.0 * cross_term))
    return float(np.sqrt(max(w2_sq, 0.0)))


def gaussian_kl(mean_a: FloatArray, cov_a: FloatArray, mean_b: FloatArray, cov_b: FloatArray) -> float:
    """Closed-form ``KL(N(mean_a, cov_a) || N(mean_b, cov_b))``. Asymmetric: unlike
    RMSE or even W2, it blows up if `mean_a`'s posterior is confidently wrong -- much
    narrower than the reference in some direction while also off-center -- rather
    than just overly diffuse, since a narrow-but-wrong posterior is a worse UQ
    failure than a wide-but-centered one."""
    mean_a = np.asarray(mean_a, dtype=float)
    mean_b = np.asarray(mean_b, dtype=float)
    cov_a = np.asarray(cov_a, dtype=float)
    cov_b = np.asarray(cov_b, dtype=float)

    k = mean_a.shape[0]
    cov_b_inv = np.linalg.inv(cov_b)
    diff = mean_b - mean_a

    term_trace = float(np.trace(cov_b_inv @ cov_a))
    term_quad = float(diff @ cov_b_inv @ diff)
    _sign_a, logdet_a = np.linalg.slogdet(cov_a)
    _sign_b, logdet_b = np.linalg.slogdet(cov_b)
    term_logdet = float(logdet_b - logdet_a)

    return 0.5 * (term_trace + term_quad - k + term_logdet)


def summarize(
    chain: MCMCChain,
    problem: Problem,
    *,
    burn_in: int,
    reference_mean: FloatArray | None = None,
    n_offline_hf: int = 0,
    n_coarse_eval_units: int | None = None,
) -> dict[str, Any]:
    """Summarizes a chain's accuracy and total HF cost. `n_offline_hf` (HF evals spent
    before this chain started, e.g. a shared seed design) is added to `n_hf_calls`
    since it's otherwise invisible in `chain.extras.used_hf`. `n_coarse_eval_units`
    overrides the HF-call-fraction denominator for methods where a chain row isn't one
    coarse eval (currently only `adaptive_stm`, via `adaptive_stm_coarse_eval_units`);
    `None` uses `chain.n_steps`. `reference_mean`, if given, adds
    `mean_abs_dev_from_reference` to the returned dict.
    """
    post = chain.burn_in(burn_in)
    n_hf_online = int(np.sum(chain.extras.used_hf)) if chain.extras.used_hf is not None else 0
    n_hf_total = n_offline_hf + n_hf_online
    steps_for_budget = chain.n_steps if n_coarse_eval_units is None else n_coarse_eval_units
    total_budget = n_offline_hf + steps_for_budget

    out: dict[str, Any] = {
        "n_steps": chain.n_steps,
        "posterior_mean": post.samples.mean(axis=0).tolist(),
        "posterior_std": post.samples.std(axis=0).tolist(),
        "rmse_to_truth": posterior_rmse(chain.samples, problem.theta_true, burn_in=burn_in),
        "n_offline_hf": n_offline_hf,
        "n_hf_calls_online": n_hf_online,
        "n_hf_calls": n_hf_total,
        "hf_call_fraction": (n_hf_total / total_budget) if total_budget > 0 else float("nan"),
    }
    if reference_mean is not None:
        out["mean_abs_dev_from_reference"] = np.abs(post.samples.mean(axis=0) - reference_mean).tolist()
    return out


def multichain_diagnostics(
    chains: list[MCMCChain], *, burn_in: int, param_names: tuple[str, ...]
) -> dict[str, Any]:
    """Gelman-Rubin R-hat and bulk-ESS across independent chains, which should share
    the same starting `theta0` so R-hat measures within-vs-between-chain mixing rather
    than sensitivity to the initial point. Chains of different length are truncated to
    the shortest, since arviz requires equal draw counts. Raises `ValueError` if fewer
    than 2 post-burn-in draws are available per chain."""
    import arviz as az

    posts = [c.burn_in(burn_in).samples for c in chains]
    min_len = min(p.shape[0] for p in posts)
    if min_len < 2:
        raise ValueError(f"Need at least 2 post-burn-in draws per chain; got min_len={min_len}.")
    stacked = np.stack([p[:min_len] for p in posts], axis=0)  # (n_chains, n_draws, n_dim)

    idata = az.from_dict(posterior={name: stacked[:, :, i] for i, name in enumerate(param_names)})
    rhat = az.rhat(idata)  # type: ignore[no-untyped-call]  # arviz.rhat has no annotations
    ess = az.ess(idata)  # type: ignore[no-untyped-call]  # arviz.ess has no annotations

    return {
        "n_chains": len(chains),
        "n_draws_per_chain": min_len,
        "rhat": {name: float(rhat[name].values) for name in param_names},
        "ess_bulk": {name: float(ess[name].values) for name in param_names},
    }


def find_burn_in_via_rhat(
    chains: list[MCMCChain],
    *,
    candidate_burn_ins: list[int] | None = None,
    rhat_threshold: float = 1.01,
    min_ess: float | None = None,
    min_burn_in: int = 0,
    patience: int = 5,
    param_names: tuple[str, ...],
) -> dict[str, Any]:
    """Finds the smallest burn-in at which R-hat drops to `rhat_threshold` *and*
    bulk-ESS is at least `min_ess` (if given), holding for `patience` consecutive
    larger candidates -- not just the first dip, which could be a fluke of a noisy
    R-hat estimate on few draws. `candidate_burn_ins` defaults to ~50 evenly-spaced
    candidates across the shortest chain; `min_burn_in` excludes candidates below a
    hard floor (e.g. an online-adapting surrogate's non-stationary opening `n`
    samples, which no amount of R-hat-based trimming inside that phase can fix).
    Returns `burn_in`/`rhat`/`ess_bulk`/`converged` for the chosen candidate plus the
    full `sweep` (useful for plotting R-hat/ESS vs. assumed burn-in).
    """
    n_total = min(c.n_steps for c in chains)
    if candidate_burn_ins is None:
        step = max(1, n_total // 50)
        candidate_burn_ins = list(range(0, n_total - 10, step))

    candidates = sorted(b for b in set(candidate_burn_ins) if b >= min_burn_in and n_total - b >= 10)

    sweep: list[dict[str, Any]] = []
    for b in candidates:
        try:
            diag = multichain_diagnostics(chains, burn_in=b, param_names=param_names)
        except ValueError:
            continue
        sweep.append(
            {
                "burn_in": b,
                "rhat": diag["rhat"],
                "ess_bulk": diag["ess_bulk"],
                "max_rhat": max(diag["rhat"].values()),
                "min_ess_bulk": min(diag["ess_bulk"].values()),
            }
        )

    def _passes(entry: dict[str, Any]) -> bool:
        if entry["max_rhat"] > rhat_threshold:
            return False
        return min_ess is None or entry["min_ess_bulk"] >= min_ess

    chosen = None
    for i, entry in enumerate(sweep):
        window = sweep[i : i + patience]
        if len(window) == patience and all(_passes(e) for e in window):
            chosen = entry
            break
    if chosen is None and sweep:
        # Never held converged for `patience` in a row within the tried range; report
        # the last (longest burn-in) point tried so the caller at least sees how close
        # it got.
        chosen = sweep[-1]

    return {
        "burn_in": chosen["burn_in"] if chosen else None,
        "rhat": chosen["rhat"] if chosen else None,
        "ess_bulk": chosen["ess_bulk"] if chosen else None,
        "converged": bool(chosen and _passes(chosen)),
        "sweep": sweep,
    }


def pooled_summarize(
    chains: list[MCMCChain],
    problem: Problem,
    *,
    burn_in: int,
    reference_mean: FloatArray | None = None,
    reference_cov: FloatArray | None = None,
    n_offline_hf: int | list[int] = 0,
    n_coarse_eval_units: int | list[int] | None = None,
) -> dict[str, Any]:
    """Like `summarize`, but pools post-burn-in samples across `chains` into one
    richer posterior estimate at a single shared `burn_in` (typically from
    `find_burn_in_via_rhat`). `n_offline_hf`/`n_coarse_eval_units` each take a single
    int for a one-time ensemble-wide cost (e.g. a shared seed design `deepcopy`'d into
    every replicate -- added once, not per chain) or a per-chain list for independently
    paid costs; raises `ValueError` if a list doesn't have one entry per chain.
    `reference_mean`+`reference_cov`, if both given, add `wasserstein2_to_reference`
    and `kl_to_reference` (typically vs. the R-hat-validated HF-only posterior).
    """
    if isinstance(n_offline_hf, int):
        # Shared, one-time cost -- reported per-chain below for transparency (each
        # replicate did inherit it), but added into totals exactly once, not
        # len(chains) times.
        n_offline_hf_list = [n_offline_hf] * len(chains)
        n_offline_hf_total = n_offline_hf
    else:
        n_offline_hf_list = list(n_offline_hf)
        n_offline_hf_total = sum(n_offline_hf_list)
    if len(n_offline_hf_list) != len(chains):
        raise ValueError("n_offline_hf list must have one entry per chain.")

    if n_coarse_eval_units is None:
        steps_list = [c.n_steps for c in chains]
    elif isinstance(n_coarse_eval_units, int):
        steps_list = [n_coarse_eval_units] * len(chains)
    else:
        steps_list = list(n_coarse_eval_units)
    if len(steps_list) != len(chains):
        raise ValueError("n_coarse_eval_units list must have one entry per chain.")

    posts = [c.burn_in(burn_in).samples for c in chains]
    pooled = np.concatenate(posts, axis=0)

    n_hf_online = sum(int(np.sum(c.extras.used_hf)) if c.extras.used_hf is not None else 0 for c in chains)
    n_hf_total = n_offline_hf_total + n_hf_online
    total_budget = n_offline_hf_total + sum(steps_list)

    posterior_mean = pooled.mean(axis=0)
    posterior_cov = np.cov(pooled, rowvar=False)
    out: dict[str, Any] = {
        "n_chains": len(chains),
        "n_pooled_samples": int(pooled.shape[0]),
        "posterior_mean": posterior_mean.tolist(),
        "posterior_std": pooled.std(axis=0).tolist(),
        "posterior_cov": posterior_cov.tolist(),
        "rmse_to_truth": float(np.mean(np.sqrt(np.sum((pooled - problem.theta_true[None, :]) ** 2, axis=1)))),
        "n_offline_hf": n_offline_hf_list,
        "n_offline_hf_total": n_offline_hf_total,
        "n_hf_calls_online": n_hf_online,
        "n_hf_calls": n_hf_total,
        "hf_call_fraction": (n_hf_total / total_budget) if total_budget > 0 else float("nan"),
    }
    if reference_mean is not None:
        out["mean_abs_dev_from_reference"] = np.abs(posterior_mean - reference_mean).tolist()
    if reference_mean is not None and reference_cov is not None:
        out["wasserstein2_to_reference"] = gaussian_wasserstein2(
            posterior_mean, posterior_cov, reference_mean, reference_cov
        )
        out["kl_to_reference"] = gaussian_kl(posterior_mean, posterior_cov, reference_mean, reference_cov)
    return out
