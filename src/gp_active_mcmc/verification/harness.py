"""Runs each comparison method to R-hat/ESS convergence and reports the cost to get
there. `n_chains` replicates advance in synchronized joblib-parallel rounds of
`chunk_size` coarse evals each, re-checking R-hat after every round.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import joblib
import numpy as np
from numpy.typing import NDArray

from gp_active_mcmc.inference import MCMCChain
from gp_active_mcmc.surrogates import PODGPSurrogate
from gp_active_mcmc.utils.rng import set_seed
from gp_active_mcmc.verification.design import DEFAULT_ONLINE_LEARNING, OnlineLearningConfig
from gp_active_mcmc.verification.methods import (
    _SEED_OFFSETS,
    _ChunkState,
    _init_adaptive_stm_production_state,
    _init_adaptive_stm_production_state_from_frozen,
    _init_adaptive_surrogate_mcmc_state,
    _init_hf_only_state,
    _make_adaptive_surrogate_mcmc_sync_hook,
)
from gp_active_mcmc.verification.metrics import pooled_summarize
from gp_active_mcmc.verification.problem import Problem
from gp_active_mcmc.verification.rounds import ConvergenceConfig, run_until_rhat_converged

FloatArray = NDArray[np.float64]

__all__ = [
    "print_convergence_driven_table",
    "run_convergence_driven_comparison",
    "run_hf_only_reference",
    "run_until_rhat_converged",
]


@dataclass(frozen=True)
class _PosteriorReference:
    """HF-only posterior summary used for method-to-reference metrics."""

    mean: FloatArray
    cov: FloatArray


def print_convergence_driven_table(results: dict[str, Any]) -> None:
    """Prints a cost-to-converge table (budget vs. accuracy per method) from
    `run_convergence_driven_comparison`'s `results`."""
    header = (
        f"{'method':<14}{'status':>10}{'rounds':>8}{'coarse_evals':>14}"
        f"{'n_hf':>8}{'w2_ref':>10}{'kl_ref':>10}{'rmse':>10}"
    )
    print(header)
    for name, r in results.items():
        pooled = r.get("pooled", {})
        rmse = pooled.get("rmse_to_truth", float("nan"))
        w2 = pooled.get("wasserstein2_to_reference")
        kl = pooled.get("kl_to_reference")
        w2_str = f"{w2:>10.4f}" if w2 is not None else f"{'--':>10}"
        kl_str = f"{kl:>10.4f}" if kl is not None else f"{'--':>10}"
        status = "converged" if r["converged"] else "capped"
        print(
            f"{name:<14}{status:>10}{r['rounds_run']:>8}{r['total_coarse_evals']:>14}"
            f"{pooled.get('n_hf_calls', 0):>8}{w2_str}{kl_str}{rmse:>10.4f}"
        )


def _run_monitored_replicates(
    init_fns: list[Callable[[], _ChunkState]],
    *,
    label: str,
    config: ConvergenceConfig,
    param_names: tuple[str, ...],
    max_hf_evals: int | None = None,
    post_round_hook: Callable[[list[_ChunkState], list[int]], tuple[list[_ChunkState], list[int]]] | None = None,
) -> dict[str, Any]:
    return run_until_rhat_converged(
        init_fns,
        config=config,
        param_names=param_names,
        max_hf_evals=max_hf_evals,
        label=label,
        post_round_hook=post_round_hook,
    )


def _take_monitored_outputs(run: dict[str, Any]) -> tuple[list[MCMCChain], list[_ChunkState]]:
    chains = cast(list[MCMCChain], run.pop("chains"))
    final_states = cast(list[_ChunkState], run.pop("final_states"))
    return chains, final_states


def _state_surrogates(states: list[_ChunkState]) -> list[PODGPSurrogate]:
    surrogates: list[PODGPSurrogate] = []
    for state in states:
        if state.model is None:
            raise ValueError("Cannot extract a surrogate from an HF-only chain state.")
        surrogates.append(cast(PODGPSurrogate, state.model.lf_model))
    return surrogates


def _attach_pooled_summary(
    run: dict[str, Any],
    chains: list[MCMCChain],
    problem: Problem,
    *,
    n_offline_hf: int | list[int],
    reference: _PosteriorReference | None = None,
    n_coarse_eval_units: int | list[int] | None = None,
) -> None:
    reference_mean = None if reference is None else reference.mean
    reference_cov = None if reference is None else reference.cov
    run["pooled"] = pooled_summarize(
        chains,
        problem,
        burn_in=run["burn_in"] or 0,
        n_offline_hf=n_offline_hf,
        n_coarse_eval_units=n_coarse_eval_units,
        reference_mean=reference_mean,
        reference_cov=reference_cov,
    )


def _reference_from_run(run: dict[str, Any]) -> _PosteriorReference:
    pooled = run["pooled"]
    return _PosteriorReference(
        mean=np.asarray(pooled["posterior_mean"]),
        cov=np.asarray(pooled["posterior_cov"]),
    )


def _run_method_and_summarize(
    label: str,
    init_fns: list[Callable[[], _ChunkState]],
    *,
    convergence: ConvergenceConfig,
    param_names: tuple[str, ...],
    problem: Problem,
    n_offline_hf: int | list[int],
    reference: _PosteriorReference | None = None,
    max_hf_evals: int | None = None,
    post_round_hook: Callable[[list[_ChunkState], list[int]], tuple[list[_ChunkState], list[int]]] | None = None,
    n_coarse_eval_units: int | list[int] | None = None,
    announce_start: bool = True,
) -> tuple[dict[str, Any], list[MCMCChain], list[_ChunkState]]:
    """Runs one method's monitored replicate loop and attaches its pooled summary --
    the tail every method block in `run_convergence_driven_comparison` shares
    (`adaptive_stm` excepted; its two-phase structure doesn't fit). `announce_start=False`
    skips the generic start line for a caller that already printed its own.
    """
    if announce_start:
        print(f"--- {label}: {len(init_fns)} chains ---")
    run = _run_monitored_replicates(
        init_fns, label=label, config=convergence, param_names=param_names,
        max_hf_evals=max_hf_evals, post_round_hook=post_round_hook,
    )
    chains, final_states = _take_monitored_outputs(run)
    _attach_pooled_summary(
        run, chains, problem, n_offline_hf=n_offline_hf, reference=reference,
        n_coarse_eval_units=n_coarse_eval_units,
    )
    print(f"--- {label} done: converged={run['converged']}, {run['rounds_run']} rounds ---")
    return run, chains, final_states


def run_hf_only_reference(
    problem: Problem,
    *,
    n_chains: int,
    convergence: ConvergenceConfig,
    seed_base: int = 0,
    theta0: FloatArray | None = None,
) -> tuple[dict[str, Any], list[MCMCChain]]:
    """Runs the `hf_only` reference alone: `n_chains` replicate MH chains, every step a
    real HF call (no surrogate at all), monitored to R-hat/ESS convergence exactly like
    `run_convergence_driven_comparison`'s own `hf_only` block -- because it *is* that
    block, factored out so it can be run, cached, and reused standalone.

    By a wide margin the most expensive part of a comparison run when `problem.hf_forward`
    is itself expensive (e.g. a real PDE solve): unlike every other method, it has no
    surrogate to make any step cheap. Its result depends on nothing else in the
    comparison -- not `gamma_threshold`, not `online_learning`, not any other method's
    outcome -- only on `problem`, `n_chains`, `convergence`, and `seed_base`/`theta0`. A
    caller re-running the same seed (e.g. while debugging a *different* method) can
    therefore compute this once, cache the returned `(hf_run, hf_chains)`, and pass it
    to `run_convergence_driven_comparison`'s `hf_only_reference` parameter on subsequent
    runs instead of recomputing it -- as long as `n_chains`/`convergence`/`seed_base`
    haven't changed (`run_convergence_driven_comparison` checks this itself).

    Returns
    -------
    hf_run, hf_chains
        `hf_run`: this run's summary dict (rounds, coarse evals, R-hat, pooled summary,
        ...), in the exact shape `run_convergence_driven_comparison`'s own `results["hf_only"]`
        has. `hf_chains`: one `MCMCChain` per replicate.
    """
    if theta0 is None:
        theta0 = np.asarray(problem.prior.rvs(random_state=set_seed(999 + seed_base)), dtype=float)
    param_names = problem.param_names
    theta0_fixed = theta0

    def _make_hf_only_init(i: int) -> Callable[[], _ChunkState]:
        return lambda: _init_hf_only_state(
            problem, seed=seed_base + _SEED_OFFSETS["hf_only"] + i, theta0=theta0_fixed
        )

    hf_init_fns: list[Callable[[], _ChunkState]] = [_make_hf_only_init(i) for i in range(n_chains)]
    hf_run, hf_chains, _hf_states = _run_method_and_summarize(
        "hf_only", hf_init_fns, convergence=convergence, param_names=param_names, problem=problem, n_offline_hf=0,
    )
    return hf_run, hf_chains


def _init_adaptive_stm_production_states_independent(
    problem: Problem,
    *,
    surrogate: PODGPSurrogate,
    gamma_threshold: float,
    max_adapt_coarse_evals: int,
    seed_base: int,
    n_chains: int,
    theta0: FloatArray,
    online_learning: OnlineLearningConfig,
    max_subchain: int,
    n_jobs: int | None,
    parallel_backend: str | None,
) -> list[tuple[_ChunkState, dict[str, Any], MCMCChain]]:
    """Runs `n_chains` independent `adaptive_stm` adaptive phases -- own seed, own
    `AdaptiveSubchain` state, own surrogate, own freeze point -- instead of one phase
    shared and deep-copied across replicates. The decentralized counterpart to
    `run_convergence_driven_comparison`'s `shared_adaptive_stm_phase=True` path.
    Parallelized like `run_until_rhat_converged`'s rounds (same `n_jobs`/
    `parallel_backend` semantics), since this repeats the (potentially expensive)
    adaptive phase once per replicate instead of once total.
    """
    effective_n_jobs = 1 if n_jobs is None else n_jobs
    parallel_kwargs: dict[str, Any] = {"n_jobs": effective_n_jobs}
    if parallel_backend is not None and effective_n_jobs != 1:
        parallel_kwargs["backend"] = parallel_backend

    def _one_adaptive_phase(i: int) -> tuple[_ChunkState, dict[str, Any], MCMCChain]:
        return _init_adaptive_stm_production_state(
            problem, surrogate=surrogate, gamma_threshold=gamma_threshold,
            max_adapt_coarse_evals=max_adapt_coarse_evals, seed=seed_base + i, theta0=theta0,
            online_learning=online_learning, max_subchain=max_subchain,
        )

    with joblib.Parallel(**parallel_kwargs) as parallel:
        results: list[tuple[_ChunkState, dict[str, Any], MCMCChain]] = parallel(
            joblib.delayed(_one_adaptive_phase)(i) for i in range(n_chains)
        )
    return results


def run_convergence_driven_comparison(
    problem: Problem,
    *,
    seed_X: FloatArray,
    seed_surrogate: PODGPSurrogate,
    n_chains: int,
    gamma_threshold: float,
    max_adapt_coarse_evals: int,
    convergence: ConvergenceConfig,
    adaptive_stm_adapt_coarse_evals: int | None = None,
    theta0: FloatArray | None = None,
    seed_base: int = 0,
    methods: tuple[str, ...] = ("hf_only", "adaptive_surrogate_mcmc", "adaptive_stm"),
    online_learning: OnlineLearningConfig = DEFAULT_ONLINE_LEARNING,
    max_subchain: int = 10_000,
    hf_only_reference: tuple[dict[str, Any], list[MCMCChain]] | None = None,
    sync_adaptive_surrogate_mcmc: bool = False,
    shared_adaptive_stm_phase: bool = False,
) -> tuple[dict[str, Any], dict[str, list[MCMCChain]], dict[str, list[PODGPSurrogate]]]:
    """Runs each method to R-hat/ESS convergence (or `convergence.max_total_coarse_evals`).

    `hf_only_reference`: an already-computed `(hf_run, hf_chains)` pair from a prior
    `run_hf_only_reference` call (e.g. loaded from a cache), reused as-is instead of
    recomputing `hf_only` -- see that function's docstring for when this is safe. Only
    used if its chain count matches `n_chains`; a mismatch is treated as a stale/wrong
    cache and silently ignored (`hf_only` is recomputed instead), rather than risking a
    result computed under a different replicate count.

    By default (`sync_adaptive_surrogate_mcmc=False`, `shared_adaptive_stm_phase=False`)
    every replicate of every surrogate-based method learns fully on its own: its own HF
    points, its own `PODGPSurrogate.refit_pod()` cadence, its own `adaptive_stm` freeze
    point -- chains only ever meet at the R-hat check `run_until_rhat_converged` does
    each round. This is what makes `adaptive_surrogate_mcmc` vs. `adaptive_stm` vs.
    `hf_only` an apples-to-apples test of whether independently-trained surrogates
    converge to the same posterior, rather than one method benefiting from
    cross-replicate pooling the other doesn't get.

    `sync_adaptive_surrogate_mcmc=True` restores the old behavior: every round, pool
    every replicate's acquired HF points and refit every replicate's surrogate on the
    pooled set (`_make_adaptive_surrogate_mcmc_sync_hook`) -- kept as an opt-in ablation,
    not the default, since it gives that method continuous access to `n_chains`x the
    training data the others never get.

    `shared_adaptive_stm_phase=True` restores the old `adaptive_stm` behavior: one
    adaptive phase (`_init_adaptive_stm_production_state`) is run once and its frozen
    surrogate/rate is deep-copied into every production replicate, instead of each
    replicate running (and paying for) its own independent adaptive phase. This was the
    more centralized of the two methods' old defaults -- all `n_chains` production
    chains started from bit-identical surrogate weights trained on a single chain's
    worth of exploration -- so it's kept opt-in too, for symmetry with
    `sync_adaptive_surrogate_mcmc`.
    """
    n_init = int(seed_X.shape[0])
    param_names = problem.param_names
    if theta0 is None:
        theta0 = np.asarray(problem.prior.rvs(random_state=set_seed(999 + n_init + seed_base)), dtype=float)

    results: dict[str, Any] = {}
    chains_by_method: dict[str, list[MCMCChain]] = {}
    surrogates_by_method: dict[str, list[PODGPSurrogate]] = {}

    def _make_state_init(state: _ChunkState) -> Callable[[], _ChunkState]:
        return lambda: state

    def _make_adaptive_surrogate_mcmc_init(seed_offset: int, i: int) -> Callable[[], _ChunkState]:
        return lambda: _init_adaptive_surrogate_mcmc_state(
            problem, surrogate=seed_surrogate, gamma_threshold=gamma_threshold,
            seed=seed_base + seed_offset + i, theta0=theta0, online_learning=online_learning,
        )

    if hf_only_reference is not None and len(hf_only_reference[1]) == n_chains:
        hf_run, hf_chains = hf_only_reference
    else:
        hf_run, hf_chains = run_hf_only_reference(
            problem, n_chains=n_chains, convergence=convergence, seed_base=seed_base, theta0=theta0,
        )
    chains_by_method["hf_only"] = hf_chains
    results["hf_only"] = hf_run
    hf_reference = _reference_from_run(hf_run)

    if "adaptive_surrogate_mcmc" in methods:
        # Default: no post_round_hook -- each replicate's surrogate refits on its own
        # PODGPSurrogate.pod_refit_every/pod_refit_max cadence, from its own HF history
        # only. sync_adaptive_surrogate_mcmc=True opts back into the old pooled-refit
        # ablation (see docstring above).
        sync_hook = _make_adaptive_surrogate_mcmc_sync_hook(n_init) if sync_adaptive_surrogate_mcmc else None
        asm_init_fns: list[Callable[[], _ChunkState]] = [
            _make_adaptive_surrogate_mcmc_init(_SEED_OFFSETS["adaptive_surrogate_mcmc"], i) for i in range(n_chains)
        ]
        asm_run, asm_chains, asm_final_states = _run_method_and_summarize(
            "adaptive_surrogate_mcmc", asm_init_fns, convergence=convergence, param_names=param_names,
            problem=problem, n_offline_hf=n_init, reference=hf_reference, max_hf_evals=max_adapt_coarse_evals,
            post_round_hook=sync_hook,
        )
        chains_by_method["adaptive_surrogate_mcmc"] = asm_chains
        surrogates_by_method["adaptive_surrogate_mcmc"] = _state_surrogates(asm_final_states)
        results["adaptive_surrogate_mcmc"] = asm_run

    if "adaptive_stm" not in methods:
        return results, chains_by_method, surrogates_by_method

    # Manual sequence: n_coarse_eval_units below needs the run's own
    # coarse_evals_per_chain, which _run_method_and_summarize can't provide.
    adaptive_stm_ceiling = max_adapt_coarse_evals if adaptive_stm_adapt_coarse_evals is None else adaptive_stm_adapt_coarse_evals

    if shared_adaptive_stm_phase:
        print(f"--- adaptive_stm: running 1 shared adaptive phase (coarse-eval ceiling={adaptive_stm_ceiling}) ---")
        shared_state, shared_adapt_meta, shared_adapt_chain = _init_adaptive_stm_production_state(
            problem, surrogate=seed_surrogate, gamma_threshold=gamma_threshold,
            max_adapt_coarse_evals=adaptive_stm_ceiling, seed=seed_base + _SEED_OFFSETS["adaptive_stm"], theta0=theta0,
            online_learning=online_learning, max_subchain=max_subchain,
        )
        assert shared_state.model is not None  # always set by _init_adaptive_stm_production_state
        shared_frozen_model = shared_state.model
        frozen_rate = shared_state.subsampling_rate
        theta_last = shared_state.theta_current
        print(f"--- adaptive_stm: adaptive phase done (frozen_rate={frozen_rate}), starting {n_chains} production chains ---")
        adaptive_stm_states = [
            _init_adaptive_stm_production_state_from_frozen(
                problem, frozen_model=shared_frozen_model, frozen_rate=frozen_rate, theta0=theta_last
            )
            for _ in range(n_chains)
        ]
        adaptive_stm_metas = [shared_adapt_meta] * n_chains
        adaptive_stm_adapt_chains = [shared_adapt_chain] * n_chains
    else:
        print(f"--- adaptive_stm: running {n_chains} independent adaptive phases (coarse-eval ceiling={adaptive_stm_ceiling}) ---")
        per_chain = _init_adaptive_stm_production_states_independent(
            problem, surrogate=seed_surrogate, gamma_threshold=gamma_threshold,
            max_adapt_coarse_evals=adaptive_stm_ceiling, seed_base=seed_base + _SEED_OFFSETS["adaptive_stm"],
            n_chains=n_chains, theta0=theta0, online_learning=online_learning, max_subchain=max_subchain,
            n_jobs=convergence.n_jobs, parallel_backend=convergence.parallel_backend,
        )
        adaptive_stm_states = [state for state, _meta, _chain in per_chain]
        adaptive_stm_metas = [meta for _state, meta, _chain in per_chain]
        adaptive_stm_adapt_chains = [chain for _state, _meta, chain in per_chain]
        frozen_rates = [s.subsampling_rate for s in adaptive_stm_states]
        print(f"--- adaptive_stm: adaptive phases done (frozen_rates={frozen_rates}), starting production ---")

    adaptive_stm_init_fns: list[Callable[[], _ChunkState]] = [_make_state_init(s) for s in adaptive_stm_states]
    adaptive_stm_run = _run_monitored_replicates(
        adaptive_stm_init_fns, label="adaptive_stm", config=convergence, param_names=param_names
    )
    adaptive_stm_chains, adaptive_stm_final_states = _take_monitored_outputs(adaptive_stm_run)
    chains_by_method["adaptive_stm"] = adaptive_stm_chains

    if shared_adaptive_stm_phase:
        # Every replicate's frozen surrogate is a deep copy of the same trained
        # weights (production never updates a frozen model) -- one is representative.
        surrogates_by_method["adaptive_stm"] = _state_surrogates(adaptive_stm_states[:1])
        # The adapt-phase cost was paid once, shared across all n_chains -- count it
        # on chain 0 only, not once per chain.
        adaptive_stm_production_coarse_evals = [
            (int(adaptive_stm_metas[0]["adapt_coarse_evals_used"]) if i == 0 else 0) + production_cost
            for i, production_cost in enumerate(adaptive_stm_run["coarse_evals_per_chain"])
        ]
    else:
        surrogates_by_method["adaptive_stm"] = _state_surrogates(adaptive_stm_final_states)
        # Each replicate paid for its own adapt phase -- add it in per chain.
        adaptive_stm_production_coarse_evals = [
            int(meta["adapt_coarse_evals_used"]) + production_cost
            for meta, production_cost in zip(
                adaptive_stm_metas, adaptive_stm_run["coarse_evals_per_chain"], strict=True
            )
        ]

    _attach_pooled_summary(
        adaptive_stm_run,
        adaptive_stm_chains,
        problem,
        n_offline_hf=n_init,
        n_coarse_eval_units=adaptive_stm_production_coarse_evals,
        reference=hf_reference,
    )
    adaptive_stm_run["adapt_metas"] = adaptive_stm_metas
    results["adaptive_stm"] = adaptive_stm_run
    chains_by_method["adaptive_stm_adapt"] = adaptive_stm_adapt_chains
    print(f"--- adaptive_stm done: converged={adaptive_stm_run['converged']}, {adaptive_stm_run['rounds_run']} rounds ---")

    return results, chains_by_method, surrogates_by_method
