"""Synchronized round runner for convergence-driven multi-chain checks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import joblib
import numpy as np
from numpy.typing import NDArray

from gp_active_mcmc.inference import MCMCChain
from gp_active_mcmc.verification.methods import _ChunkState, _suppress_tinyda_output
from gp_active_mcmc.verification.metrics import find_burn_in_via_rhat

__all__ = ["ConvergenceConfig", "run_until_rhat_converged"]


@dataclass(frozen=True)
class ConvergenceConfig:
    """Convergence-loop settings for one round-monitored method: how much budget to
    advance per round, when to declare convergence, and how to parallelize across
    replicates. Shared by every method in one `run_convergence_driven_comparison`
    call, so it's one value passed around instead of eight.
    """

    chunk_size: int  # coarse-evaluation-unit budget advanced per round, per replicate
    max_total_coarse_evals: int  # safety cap on total coarse-evaluation cost
    rhat_threshold: float = 1.01  # maximum R-hat to declare convergence
    min_ess: float | None = None  # minimum bulk-ESS to declare convergence; None skips the ESS check
    patience: int = 5  # consecutive passing rounds/candidates required before declaring convergence
    n_jobs: int | None = None  # parallel workers; None defaults to sequential (1) for portability
    parallel_backend: str | None = None  # joblib backend for n_jobs > 1 ("loky"/"threading"); None = joblib default
    burn_in_fraction: float | None = None  # fixed-candidate mode; see run_until_rhat_converged for full semantics


def _reset_model_log(model: Any | None) -> None:
    """Clear `model.log.used_hf` before each independent chunk run."""
    if model is not None and hasattr(model, "log"):
        model.log.used_hf.clear()


def _extract_used_hf_array(model: Any) -> NDArray[np.bool_]:
    """Read `model.log.used_hf` as a boolean array. Pairs with `_reset_model_log`
    (reset before a round's `advance()`, read after) -- see `_run_chunk`."""
    return np.asarray(model.log.used_hf, dtype=bool)


def _run_chunk(state: _ChunkState, chunk_size: int) -> tuple[_ChunkState, MCMCChain, int]:
    """Advance one replicate by one chunk of coarse-evaluation budget.

    Advances `state.chain_obj` -- a persistent `ResumableTdaChain` built once by the
    `_init_*_state` factory that created this replicate's state and carried forward
    round to round via `dataclasses.replace` (which this function uses to return the
    next round's state, and which leaves fields it doesn't override, `chain_obj`
    included, referencing the exact same object) -- instead of reconstructing a
    `tinyDA` chain fresh every round. See `ResumableTdaChain`'s docstring for why that
    distinction matters (proposal/model state that would otherwise be silently reset
    every round).
    """
    if state.chain_obj is None:
        raise ValueError("_ChunkState.chain_obj must be set (see the _init_*_state factories in methods.py).")

    iterations = max(1, chunk_size // state.subsampling_rate)
    coarse_evals_spent = iterations * state.subsampling_rate

    with _suppress_tinyda_output():
        if state.model is None:
            state.chain_obj.advance(iterations)
            samples = state.chain_obj.new_since_last(state.chain_key)
            used_hf = np.ones(samples.shape[0], dtype=bool)
        elif state.chain_key == "chain_fine_0":
            _reset_model_log(state.model)  # hygiene only: used_hf below never reads the log
            state.chain_obj.advance(iterations)
            samples = state.chain_obj.new_since_last(state.chain_key)
            # tinyDA's DA "fine" chain: one row per outer DA iteration, each backed by
            # exactly one genuine HF correction by construction of delayed acceptance.
            # model.log.used_hf instead counts per-*coarse*-step calls (subsampling_rate
            # of them per fine row), so it does NOT align 1:1 with samples here the way
            # it does for a single-posterior chain (the branch below).
            used_hf = np.ones(samples.shape[0], dtype=bool)
        else:
            # The bootstrap sample from chain_obj's own construction logged one
            # model evaluation before any advance() ever ran. It's included in
            # samples only on the round whose new_since_last() call is the very
            # first for this chain_key -- so the log must NOT be reset before that
            # one round (its used_hf needs the bootstrap's own entry too), but
            # should be reset before every later round (whose samples contain only
            # that round's own new evaluations).
            if not state.chain_obj.is_first_advance(state.chain_key):
                _reset_model_log(state.model)
            state.chain_obj.advance(iterations)
            samples = state.chain_obj.new_since_last(state.chain_key)
            used_hf = _extract_used_hf_array(state.model)
        chain = MCMCChain.from_arrays(samples=samples, used_hf=used_hf)

    return replace(state, theta_current=chain.samples[-1]), chain, coarse_evals_spent


def _concat_chain_blocks(blocks: list[MCMCChain]) -> MCMCChain:
    """Concatenate one replicate's per-round chunk blocks into its full chain so far.

    `state.chain_obj` (a `ResumableTdaChain`) is advanced, not reconstructed, each
    round -- see its docstring -- so the chain's one-time bootstrap sample (`theta0`,
    evaluated once at `chain_obj`'s own construction) appears only as `blocks[0]`'s
    first row; every later block contains only genuinely new proposals `advance()`
    generated, with no block-boundary duplicate left to drop here (unlike the old
    per-round-reconstruction design, where every block after the first re-evaluated
    the boundary point as its own first row).
    """
    samples = np.vstack([b.samples for b in blocks])

    def _used_hf(b: MCMCChain) -> NDArray[np.bool_]:
        return b.extras.used_hf if b.extras.used_hf is not None else np.zeros(b.n_steps, dtype=bool)

    used_hf = np.concatenate([_used_hf(b) for b in blocks])
    return MCMCChain.from_arrays(samples=samples, used_hf=used_hf)


def _validate_round_run_inputs(
    init_fns: list[Callable[[], _ChunkState]],
    *,
    chunk_size: int,
    max_total_coarse_evals: int,
    max_hf_evals: int | None,
) -> None:
    if not init_fns:
        raise ValueError("init_fns must contain at least one chain initializer.")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if max_total_coarse_evals <= 0:
        raise ValueError("max_total_coarse_evals must be positive.")
    if max_hf_evals is not None and max_hf_evals <= 0:
        raise ValueError("max_hf_evals must be positive when given.")


def _chain_hf_count(chain: MCMCChain) -> int:
    used_hf = chain.extras.used_hf
    return int(np.sum(used_hf)) if used_hf is not None else 0


def _check_round_convergence(
    chains: list[MCMCChain],
    *,
    rhat_threshold: float,
    min_ess: float | None,
    patience: int,
    param_names: tuple[str, ...],
    burn_in_fraction: float | None,
    stable_streak: int,
) -> tuple[dict[str, Any], int]:
    if burn_in_fraction is None:
        return (
            find_burn_in_via_rhat(
                chains,
                rhat_threshold=rhat_threshold,
                min_ess=min_ess,
                patience=patience,
                param_names=param_names,
            ),
            stable_streak,
        )

    n_total = min(c.n_steps for c in chains)
    fixed_candidate = int(burn_in_fraction * n_total)
    conv = find_burn_in_via_rhat(
        chains,
        candidate_burn_ins=[fixed_candidate],
        rhat_threshold=rhat_threshold,
        min_ess=min_ess,
        patience=1,
        param_names=param_names,
    )
    stable_streak = stable_streak + 1 if conv["converged"] else 0
    conv["converged"] = stable_streak >= patience
    return conv, stable_streak


def _stop_reason(*, converged: bool, hf_cap_hit: bool) -> str:
    if converged:
        return "converged"
    if hf_cap_hit:
        return "max_hf_evals"
    return "max_coarse_evals"


def _print_round_progress(
    *,
    label: str | None,
    rounds_run: int,
    total_coarse_evals: int,
    max_total_coarse_evals: int,
    total_hf_evals: int,
    max_hf_evals: int | None,
    conv: dict[str, Any],
    final: bool,
) -> None:
    prefix = f"[{label}] " if label else ""
    rhat_str = (
        ", ".join(f"{k}={v:.3f}" for k, v in conv["rhat"].items())
        if conv["rhat"] is not None
        else "n/a"
    )
    hf_str = f"  hf_evals={total_hf_evals}/{max_hf_evals}" if max_hf_evals is not None else ""
    line = (
        f"  {prefix}round {rounds_run}: coarse_evals={total_coarse_evals}/{max_total_coarse_evals}"
        f"{hf_str}  rhat=({rhat_str})  converged={conv['converged']}"
    )
    print(f"\033[K{line}", end=("\n" if final else "\r"), flush=True)


def run_until_rhat_converged(
    init_fns: list[Callable[[], _ChunkState]],
    *,
    config: ConvergenceConfig,
    param_names: tuple[str, ...],
    max_hf_evals: int | None = None,
    label: str | None = None,
    verbose: bool = True,
    post_round_hook: Callable[[list[_ChunkState], list[int]], tuple[list[_ChunkState], list[int]]] | None = None,
) -> dict[str, Any]:
    """Runs `len(init_fns)` independent replicate chains.
    """
    _validate_round_run_inputs(
        init_fns,
        chunk_size=config.chunk_size,
        max_total_coarse_evals=config.max_total_coarse_evals,
        max_hf_evals=max_hf_evals,
    )

    states = [f() for f in init_fns]
    n_chains = len(states)
    blocks: list[list[MCMCChain]] = [[] for _ in range(n_chains)]
    coarse_evals_per_chain = [0] * n_chains
    hf_evals_per_chain = [0] * n_chains
    rounds_run = 0
    stable_streak = 0

    # Sequential by default: loky's process backend can fail where semaphore queries
    # are blocked. Callers can still opt into n_jobs > 1 with an explicit backend.
    effective_n_jobs = 1 if config.n_jobs is None else config.n_jobs
    parallel_kwargs: dict[str, Any] = {"n_jobs": effective_n_jobs}
    if config.parallel_backend is not None and effective_n_jobs != 1:
        parallel_kwargs["backend"] = config.parallel_backend

    with joblib.Parallel(**parallel_kwargs) as parallel:
        while True:
            chunk_results = parallel(
                joblib.delayed(_run_chunk)(states[i], config.chunk_size) for i in range(n_chains)
            )
            for i, (new_state, block, coarse_evals_spent) in enumerate(chunk_results):
                states[i] = new_state
                blocks[i].append(block)
                coarse_evals_per_chain[i] += coarse_evals_spent
                hf_evals_per_chain[i] += _chain_hf_count(block)
            rounds_run += 1

            if post_round_hook is not None:
                states, hf_evals_per_chain = post_round_hook(states, hf_evals_per_chain)

            chains_so_far = [_concat_chain_blocks(b) for b in blocks]
            total_coarse_evals = max(coarse_evals_per_chain)
            total_hf_evals = max(hf_evals_per_chain)
            conv, stable_streak = _check_round_convergence(
                chains_so_far,
                rhat_threshold=config.rhat_threshold,
                min_ess=config.min_ess,
                patience=config.patience,
                param_names=param_names,
                burn_in_fraction=config.burn_in_fraction,
                stable_streak=stable_streak,
            )

            hf_cap_hit = max_hf_evals is not None and total_hf_evals >= max_hf_evals
            final = conv["converged"] or total_coarse_evals >= config.max_total_coarse_evals or hf_cap_hit

            if verbose:
                _print_round_progress(
                    label=label,
                    rounds_run=rounds_run,
                    total_coarse_evals=total_coarse_evals,
                    max_total_coarse_evals=config.max_total_coarse_evals,
                    total_hf_evals=total_hf_evals,
                    max_hf_evals=max_hf_evals,
                    conv=conv,
                    final=final,
                )

            if final:
                return {
                    "chains": chains_so_far,
                    "rounds_run": rounds_run,
                    "coarse_evals_per_chain": coarse_evals_per_chain,
                    "total_coarse_evals": total_coarse_evals,
                    "hf_evals_per_chain": hf_evals_per_chain,
                    "total_hf_evals": total_hf_evals,
                    "stop_reason": _stop_reason(converged=bool(conv["converged"]), hf_cap_hit=hf_cap_hit),
                    "final_states": states,
                    **conv,
                }
