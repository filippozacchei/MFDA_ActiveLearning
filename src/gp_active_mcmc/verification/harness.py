"""Runs each comparison method to R-hat/ESS convergence and reports the cost to get
there. `n_chains` replicates advance in synchronized joblib-parallel rounds of
`chunk_size` coarse evals each, re-checking R-hat after every round.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from gp_active_mcmc.inference import MCMCChain
from gp_active_mcmc.surrogates import PODGPSurrogate
from gp_active_mcmc.surrogates.gp import KernelName
from gp_active_mcmc.utils.rng import set_seed
from gp_active_mcmc.verification.design import DEFAULT_ONLINE_LEARNING, OnlineLearningConfig
from gp_active_mcmc.verification.methods import (
    _SEED_OFFSETS,
    _ChunkState,
    _init_adaptive_da_production_state,
    _init_adaptive_da_production_state_from_frozen,
    _init_hf_only_state,
    _init_online_active_state,
    _init_pretrained_state_from_surrogate,
    _make_online_active_sync_hook,
    _train_pretrained_surrogate,
)
from gp_active_mcmc.verification.metrics import pooled_summarize
from gp_active_mcmc.verification.problem import Problem
from gp_active_mcmc.verification.rounds import ConvergenceConfig, run_until_rhat_converged

FloatArray = NDArray[np.float64]

__all__ = [
    "print_convergence_driven_table",
    "run_convergence_driven_comparison",
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
    (`adaptive_da` excepted; its two-phase structure doesn't fit). `announce_start=False`
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


def run_convergence_driven_comparison(
    problem: Problem,
    *,
    pod_rank: int,
    kernel: KernelName,
    seed_X: FloatArray,
    seed_Y: FloatArray,
    seed_surrogate: PODGPSurrogate,
    n_chains: int,
    gamma_threshold: float,
    max_adapt_coarse_evals: int,
    convergence: ConvergenceConfig,
    adaptive_da_adapt_coarse_evals: int | None = None,
    theta0: FloatArray | None = None,
    seed_base: int = 0,
    methods: tuple[str, ...] = ("hf_only", "pretrained", "online_active", "adaptive_da"),
    online_learning: OnlineLearningConfig = DEFAULT_ONLINE_LEARNING,
    max_subchain: int = 10_000,
) -> tuple[dict[str, Any], dict[str, list[MCMCChain]], dict[str, list[PODGPSurrogate]]]:
    """Runs each method to R-hat/ESS convergence (or `convergence.max_total_coarse_evals`)
    and reports cost (HF calls, coarse evals) and accuracy for the comparison table.

    `pretrained`/`adaptive_da` train once and deep-copy into `n_chains` replicates
    instead of retraining each independently, so every replicate targets the same
    frozen posterior -- independent retraining would break R-hat's same-target
    assumption. `online_active` has no freeze point and keeps learning per replicate.

    `max_adapt_coarse_evals` doubles as `adaptive_da`'s adaptive-phase ceiling and
    `online_active`'s HF-call cap, keeping their online budgets comparable;
    `adaptive_da_adapt_coarse_evals` decouples them when needed -- coarse evals and HF
    calls aren't the same currency, since `adaptive_da` skips most coarse evals as HF.

    `methods` restricts which methods beyond the always-run `hf_only` reference
    actually run their `n_chains`-replicate loop.

    Returns
    -------
    results, chains_by_method, surrogates_by_method
        `results[method]` has `converged`, `rounds_run`, `total_coarse_evals`, `rhat`,
        `ess_bulk`, `burn_in`, `pooled` (`adaptive_da` also has `adapt_metas`).
        `chains_by_method` also has `"adaptive_da_adapt"`. `surrogates_by_method` omits
        `hf_only` (no surrogate).
    """
    n_init = int(seed_X.shape[0])
    param_names = problem.param_names
    if theta0 is None:
        theta0 = np.asarray(problem.prior.rvs(random_state=set_seed(999 + n_init + seed_base)), dtype=float)

    results: dict[str, Any] = {}
    chains_by_method: dict[str, list[MCMCChain]] = {}
    surrogates_by_method: dict[str, list[PODGPSurrogate]] = {}

    def _make_hf_only_init(i: int) -> Callable[[], _ChunkState]:
        return lambda: _init_hf_only_state(problem, seed=seed_base + _SEED_OFFSETS["hf_only"] + i, theta0=theta0)

    def _make_state_init(state: _ChunkState) -> Callable[[], _ChunkState]:
        return lambda: state

    def _make_online_active_init(seed_offset: int, i: int) -> Callable[[], _ChunkState]:
        return lambda: _init_online_active_state(
            problem, surrogate=seed_surrogate, gamma_threshold=gamma_threshold,
            seed=seed_base + seed_offset + i, theta0=theta0, online_learning=online_learning,
        )

    hf_init_fns: list[Callable[[], _ChunkState]] = [_make_hf_only_init(i) for i in range(n_chains)]
    hf_run, hf_chains, _hf_states = _run_method_and_summarize(
        "hf_only", hf_init_fns, convergence=convergence, param_names=param_names, problem=problem, n_offline_hf=0,
    )
    chains_by_method["hf_only"] = hf_chains
    results["hf_only"] = hf_run
    hf_reference = _reference_from_run(hf_run)

    if "pretrained" in methods:
        print("--- pretrained: training 1 shared offline design ---")
        pretrained_surrogate, n_hf_pretrained_shared = _train_pretrained_surrogate(
            problem, seed_X=seed_X, seed_Y=seed_Y, gamma_threshold=gamma_threshold, pod_rank=pod_rank,
            kernel=kernel, seed=seed_base + _SEED_OFFSETS["pretrained"],
        )
        n_hf_pretrained_list = [n_hf_pretrained_shared] + [0] * (n_chains - 1)
        print(f"--- pretrained: offline design done (n_hf={n_hf_pretrained_shared}), starting {n_chains} chains ---")
        pretrained_inits = [
            _init_pretrained_state_from_surrogate(
                problem, surrogate=pretrained_surrogate, seed=seed_base + _SEED_OFFSETS["pretrained"] + i,
                theta0=theta0,
            )
            for i in range(n_chains)
        ]
        pretrained_init_fns: list[Callable[[], _ChunkState]] = [_make_state_init(s) for s in pretrained_inits]
        pre_run, pretrained_chains, _pretrained_states = _run_method_and_summarize(
            "pretrained", pretrained_init_fns, convergence=convergence, param_names=param_names, problem=problem,
            n_offline_hf=n_hf_pretrained_list, reference=hf_reference, announce_start=False,
        )
        chains_by_method["pretrained"] = pretrained_chains
        surrogates_by_method["pretrained"] = [pretrained_surrogate]
        results["pretrained"] = pre_run

    if "online_active" in methods:
        online_active_init_fns: list[Callable[[], _ChunkState]] = [
            _make_online_active_init(_SEED_OFFSETS["online_active"], i) for i in range(n_chains)
        ]
        on_run, online_chains, online_final_states = _run_method_and_summarize(
            "online_active", online_active_init_fns, convergence=convergence, param_names=param_names,
            problem=problem, n_offline_hf=n_init, reference=hf_reference, max_hf_evals=max_adapt_coarse_evals,
        )
        chains_by_method["online_active"] = online_chains
        surrogates_by_method["online_active"] = _state_surrogates(online_final_states)
        results["online_active"] = on_run

    if "online_active_synced" in methods:
        sync_hook = _make_online_active_sync_hook(n_init)
        synced_init_fns: list[Callable[[], _ChunkState]] = [
            _make_online_active_init(_SEED_OFFSETS["online_active_synced"], i) for i in range(n_chains)
        ]
        oas_run, synced_chains, synced_final_states = _run_method_and_summarize(
            "online_active_synced", synced_init_fns, convergence=convergence, param_names=param_names,
            problem=problem, n_offline_hf=n_init, reference=hf_reference, max_hf_evals=max_adapt_coarse_evals,
            post_round_hook=sync_hook,
        )
        chains_by_method["online_active_synced"] = synced_chains
        surrogates_by_method["online_active_synced"] = _state_surrogates(synced_final_states)
        results["online_active_synced"] = oas_run

    if "adaptive_da" not in methods:
        return results, chains_by_method, surrogates_by_method

    # Manual sequence: n_coarse_eval_units below needs the run's own
    # coarse_evals_per_chain, which _run_method_and_summarize can't provide.
    adaptive_da_ceiling = max_adapt_coarse_evals if adaptive_da_adapt_coarse_evals is None else adaptive_da_adapt_coarse_evals
    print(f"--- adaptive_da: running 1 shared adaptive phase (coarse-eval ceiling={adaptive_da_ceiling}) ---")
    shared_state, shared_adapt_meta, shared_adapt_chain = _init_adaptive_da_production_state(
        problem, surrogate=seed_surrogate, gamma_threshold=gamma_threshold,
        max_adapt_coarse_evals=adaptive_da_ceiling, seed=seed_base + _SEED_OFFSETS["adaptive_da"], theta0=theta0,
        online_learning=online_learning, max_subchain=max_subchain,
    )
    assert shared_state.model is not None  # always set by _init_adaptive_da_production_state
    shared_frozen_model = shared_state.model
    frozen_rate = shared_state.subsampling_rate
    theta_last = shared_state.theta_current
    print(f"--- adaptive_da: adaptive phase done (frozen_rate={frozen_rate}), starting {n_chains} production chains ---")
    adaptive_da_inits = [
        _init_adaptive_da_production_state_from_frozen(
            problem, frozen_model=shared_frozen_model, frozen_rate=frozen_rate, theta0=theta_last
        )
        for _ in range(n_chains)
    ]
    adaptive_da_metas = [shared_adapt_meta] * n_chains
    adaptive_da_adapt_chains = [shared_adapt_chain] * n_chains
    adaptive_da_init_fns: list[Callable[[], _ChunkState]] = [_make_state_init(s) for s in adaptive_da_inits]
    adaptive_da_run = _run_monitored_replicates(
        adaptive_da_init_fns, label="adaptive_da", config=convergence, param_names=param_names
    )
    adaptive_da_chains, _adaptive_da_states = _take_monitored_outputs(adaptive_da_run)
    chains_by_method["adaptive_da"] = adaptive_da_chains
    surrogates_by_method["adaptive_da"] = [cast(PODGPSurrogate, shared_frozen_model.lf_model)]
    adaptive_da_production_coarse_evals = [
        (int(shared_adapt_meta["adapt_coarse_evals_used"]) if i == 0 else 0) + production_cost
        for i, production_cost in enumerate(adaptive_da_run["coarse_evals_per_chain"])
    ]
    _attach_pooled_summary(
        adaptive_da_run,
        adaptive_da_chains,
        problem,
        n_offline_hf=n_init,
        n_coarse_eval_units=adaptive_da_production_coarse_evals,
        reference=hf_reference,
    )
    adaptive_da_run["adapt_metas"] = adaptive_da_metas
    results["adaptive_da"] = adaptive_da_run
    chains_by_method["adaptive_da_adapt"] = adaptive_da_adapt_chains
    print(f"--- adaptive_da done: converged={adaptive_da_run['converged']}, {adaptive_da_run['rounds_run']} rounds ---")

    return results, chains_by_method, surrogates_by_method
