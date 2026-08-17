"""Convergence-driven multi-method comparison harness.

`run_convergence_driven_comparison` runs each method until its replicate chains agree
(R-hat/ESS converge) and reports the cost to get there -- the comparison that
demonstrates a computational-efficiency claim.

`n_chains` replicate chains advance in synchronized rounds: each round runs one
`chunk_size`-coarse-eval chunk per replicate in parallel (`joblib`), then re-checks
R-hat on the chains so far.
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
    """Cost-to-converge table: for each method, how much budget (coarse evals, HF
    calls) it needed to reach R-hat<=threshold and ESS>=min_ess (or "capped" if it
    never did), alongside the resulting accuracy.

    Parameters
    ----------
    results
        As returned by `run_convergence_driven_comparison` (its `results` element).
    """
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
    """Runs each method until its replicate chains agree (R-hat/ESS converge, per
    `convergence`) or `convergence.max_total_coarse_evals` is hit, and reports the
    cost (HF calls, coarse evals) to get there alongside the resulting accuracy.

    `pretrained` and `adaptive_da` each build their `n_chains` replicates from a
    single shared offline design / adaptive phase, deep-copied into each replicate's
    state, rather than an independently-retrained run per replicate -- independent
    retraining would make each replicate target a subtly different frozen posterior,
    breaking R-hat's same-target assumption. The shared one-time cost is charged to
    the first replicate only. `online_active` has no freeze point, so each replicate's
    surrogate keeps learning independently throughout.

    Parameters
    ----------
    problem
        Inverse-problem definition. `problem.param_names` is forwarded to
        `find_burn_in_via_rhat`; the shared offline design's size (`seed_X.shape[0]`)
        is used wherever `n_init` was previously passed separately.
    pod_rank
        POD basis rank.
    kernel
        GP kernel name.
    seed_X, seed_Y
        Shared initial training design.
    seed_surrogate
        Shared starting surrogate for `online_active`/`adaptive_da`.
    n_chains
        Number of independent replicate chains per method.
    gamma_threshold
        Predictive-variance/-standard-deviation trust threshold.
    max_adapt_coarse_evals
        Coarse-evaluation ceiling for `adaptive_da`'s adaptive phase, and the
        `max_hf_evals` cap for `online_active`/`online_active_synced` (so the two
        methods' online-learning HF budgets are directly comparable).
    convergence
        Convergence-loop settings shared by every method's monitored round loop.
    adaptive_da_adapt_coarse_evals
        Decouples `adaptive_da`'s adaptive-phase coarse-eval ceiling from
        `online_active`'s `max_hf_evals` cap, which otherwise share the single
        `max_adapt_coarse_evals` value. That sharing looks like "the same ceiling",
        but isn't the same currency: `max_hf_evals` bounds `online_active`'s real
        HF-call count directly, while `adaptive_da`'s adaptive phase is capped by
        total *coarse* evals, of which typically only a small fraction end up being
        real HF calls (`AdaptiveSubchain` grows trust and skips HF quickly once its
        own RMSE target is met). `None` (the default) means "same as
        `max_adapt_coarse_evals`". Note `AdaptiveSubchain.has_converged` can still stop
        the adaptive phase well short of any ceiling on its own signal -- raising this
        only helps if the phase was actually ceiling-bound, not convergence-bound.
    theta0
        Shared starting parameter vector. Drawn from `problem.prior` if not given.
    seed_base
        Base seed; each method/replicate uses a distinct offset from this.
    methods
        Restricts which surrogate-based methods (``"pretrained"``, ``"online_active"``,
        ``"online_active_synced"``, ``"adaptive_da"``) actually run their (expensive)
        `n_chains`-replicate monitored loop -- ``"hf_only"`` always runs regardless,
        since it supplies the HF-only reference every other method is compared
        against.
    online_learning
        POD/GP refit and rank settings for `online_active`/`online_active_synced`/
        `adaptive_da`'s surrogates.
    max_subchain
        Forwarded to `adaptive_da`'s `AdaptiveSubchainControl`.

    Returns
    -------
    results, chains_by_method, surrogates_by_method
        `results` is JSON-serializable, keyed by method name, each entry
        containing `converged`, `rounds_run`, `coarse_evals_per_chain`,
        `total_coarse_evals`, `rhat`, `ess_bulk`, `burn_in`, and `pooled`.
        `adaptive_da`'s entry additionally has `adapt_metas` (per-replicate
        adaptive-phase bookkeeping). `chains_by_method` also has
        ``"adaptive_da_adapt"``: each replicate's adaptive-phase chain.
        `surrogates_by_method` is keyed ``"pretrained"``/``"online_active"``/
        ``"online_active_synced"``/``"adaptive_da"`` only (`hf_only` has no surrogate).
    """
    n_init = int(seed_X.shape[0])
    param_names = problem.param_names
    if theta0 is None:
        theta0 = np.asarray(problem.prior.rvs(random_state=set_seed(999 + n_init + seed_base)), dtype=float)

    results: dict[str, Any] = {}
    chains_by_method: dict[str, list[MCMCChain]] = {}
    surrogates_by_method: dict[str, list[PODGPSurrogate]] = {}

    def _make_hf_only_init(i: int) -> Callable[[], _ChunkState]:
        return lambda: _init_hf_only_state(problem, seed=seed_base + 100 + i, theta0=theta0)

    def _make_state_init(state: _ChunkState) -> Callable[[], _ChunkState]:
        return lambda: state

    def _make_online_active_init(seed_offset: int, i: int) -> Callable[[], _ChunkState]:
        return lambda: _init_online_active_state(
            problem, surrogate=seed_surrogate, gamma_threshold=gamma_threshold,
            seed=seed_base + seed_offset + i, theta0=theta0, online_learning=online_learning,
        )

    print(f"--- hf_only: {n_chains} chains ---")
    hf_init_fns: list[Callable[[], _ChunkState]] = [_make_hf_only_init(i) for i in range(n_chains)]
    hf_run = _run_monitored_replicates(hf_init_fns, label="hf_only", config=convergence, param_names=param_names)
    hf_chains, _hf_states = _take_monitored_outputs(hf_run)
    chains_by_method["hf_only"] = hf_chains
    _attach_pooled_summary(hf_run, hf_chains, problem, n_offline_hf=0)
    results["hf_only"] = hf_run
    print(f"--- hf_only done: converged={hf_run['converged']}, {hf_run['rounds_run']} rounds ---")

    hf_reference = _reference_from_run(hf_run)

    if "pretrained" in methods:
        print("--- pretrained: training 1 shared offline design ---")
        pretrained_surrogate, n_hf_pretrained_shared = _train_pretrained_surrogate(
            problem, seed_X=seed_X, seed_Y=seed_Y, gamma_threshold=gamma_threshold, pod_rank=pod_rank,
            kernel=kernel, seed=seed_base + 200,
        )
        pretrained_inits = [
            _init_pretrained_state_from_surrogate(
                problem, surrogate=pretrained_surrogate, seed=seed_base + 200 + i, theta0=theta0
            )
            for i in range(n_chains)
        ]
        n_hf_pretrained_list = [n_hf_pretrained_shared] + [0] * (n_chains - 1)
        print(f"--- pretrained: offline design done (n_hf={n_hf_pretrained_shared}), starting {n_chains} chains ---")
        pretrained_init_fns: list[Callable[[], _ChunkState]] = [_make_state_init(s) for s in pretrained_inits]
        pre_run = _run_monitored_replicates(
            pretrained_init_fns, label="pretrained", config=convergence, param_names=param_names
        )
        pretrained_chains, _pretrained_states = _take_monitored_outputs(pre_run)
        chains_by_method["pretrained"] = pretrained_chains
        surrogates_by_method["pretrained"] = [pretrained_surrogate]
        _attach_pooled_summary(
            pre_run,
            pretrained_chains,
            problem,
            n_offline_hf=n_hf_pretrained_list,
            reference=hf_reference,
        )
        results["pretrained"] = pre_run
        print(f"--- pretrained done: converged={pre_run['converged']}, {pre_run['rounds_run']} rounds ---")

    if "online_active" in methods:
        print(f"--- online_active: {n_chains} chains ---")
        online_active_init_fns: list[Callable[[], _ChunkState]] = [
            _make_online_active_init(300, i) for i in range(n_chains)
        ]
        on_run = _run_monitored_replicates(
            online_active_init_fns,
            label="online_active",
            config=convergence,
            param_names=param_names,
            max_hf_evals=max_adapt_coarse_evals,
        )
        online_chains, online_final_states = _take_monitored_outputs(on_run)
        chains_by_method["online_active"] = online_chains
        surrogates_by_method["online_active"] = _state_surrogates(online_final_states)
        _attach_pooled_summary(
            on_run,
            online_chains,
            problem,
            n_offline_hf=n_init,
            reference=hf_reference,
        )
        results["online_active"] = on_run
        print(f"--- online_active done: converged={on_run['converged']}, {on_run['rounds_run']} rounds ---")

    if "online_active_synced" in methods:
        print(f"--- online_active_synced: {n_chains} chains ---")
        sync_hook = _make_online_active_sync_hook(n_init)
        synced_init_fns: list[Callable[[], _ChunkState]] = [
            _make_online_active_init(350, i) for i in range(n_chains)
        ]
        oas_run = _run_monitored_replicates(
            synced_init_fns,
            label="online_active_synced",
            config=convergence,
            param_names=param_names,
            max_hf_evals=max_adapt_coarse_evals,
            post_round_hook=sync_hook,
        )
        synced_chains, synced_final_states = _take_monitored_outputs(oas_run)
        chains_by_method["online_active_synced"] = synced_chains
        surrogates_by_method["online_active_synced"] = _state_surrogates(synced_final_states)
        _attach_pooled_summary(
            oas_run,
            synced_chains,
            problem,
            n_offline_hf=n_init,
            reference=hf_reference,
        )
        results["online_active_synced"] = oas_run
        print(f"--- online_active_synced done: converged={oas_run['converged']}, {oas_run['rounds_run']} rounds ---")

    if "adaptive_da" not in methods:
        return results, chains_by_method, surrogates_by_method

    adaptive_da_ceiling = max_adapt_coarse_evals if adaptive_da_adapt_coarse_evals is None else adaptive_da_adapt_coarse_evals
    print(f"--- adaptive_da: running 1 shared adaptive phase (coarse-eval ceiling={adaptive_da_ceiling}) ---")
    shared_state, shared_adapt_meta, shared_adapt_chain = _init_adaptive_da_production_state(
        problem, surrogate=seed_surrogate, gamma_threshold=gamma_threshold,
        max_adapt_coarse_evals=adaptive_da_ceiling, seed=seed_base + 400, theta0=theta0,
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
