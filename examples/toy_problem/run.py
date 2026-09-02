"""Runs the mass-spring-damper method comparison
(hf_only/pretrained/adaptive_surrogate_mcmc/adaptive_stm, see `msd_methods`) across
`--n-seeds` independent problem instances, appending one metrics row per seed to
`results/sweep_convergence_driven[_TAG].jsonl` and saving each seed's figures/artifact
bundle alongside it.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

import joblib

_MPLCONFIGDIR = Path(__file__).parent / "results" / ".matplotlib"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")

from msd_methods import (
    DEFAULT_ONLINE_LEARNING,
    GAMMA_THRESHOLD,
    KERNEL,
    MAX_ADAPT_COARSE_EVALS,
    MAX_SUBCHAIN,
    N_INIT,
    POD_REFIT_EVERY,
    RANK_ENERGY_THRESHOLD,
    RESULTS_DIR,
    ConvergenceConfig,
    OnlineLearningConfig,
    build_initial_surrogate,
    build_problem,
    prepare_trace_data,
    run_convergence_driven_comparison,
    run_training_cost_comparison,
)
from msd_plots import (
    plot_posterior_scatter,
    plot_surrogate_comparison,
    plot_traces,
    plot_training_points_scatter,
)

from gp_active_mcmc.utils.rng import set_seed

# Spaced well past the largest per-seed offset (methods._SEED_OFFSETS tops out at
# 600) so one seed's internal seeding never collides with the next seed's.
SEED_STRIDE = 10_000


def load_artifact(seed: int, *, tag: str = "") -> dict[str, Any]:
    """Loads one seed's full result bundle saved by `run_one_seed` (`seed_surrogate`,
    `seed_X`/`seed_Y`, `offline_surrogate`, `surrogates_by_method`, `chains_by_method`,
    `training_cost`, `posterior` -- the same dicts written to the `.jsonl` row), e.g.
    to remake a figure with different styling without re-running the sweep. Also
    reconstructs `problem` via `build_problem`, since `Problem.hf_forward` is a
    closure and isn't itself picklable."""
    path = _artifacts_dir(tag) / f"seed_{seed}.joblib"
    artifact = joblib.load(path)
    artifact["problem"] = build_problem(problem_seed=artifact["problem_seed"], sigma_obs=artifact["sigma_obs"])
    return artifact


def _artifacts_dir(tag: str) -> Path:
    return RESULTS_DIR / "sweep_artifacts" / (tag or "default")


def _figures_dir(tag: str) -> Path:
    base = RESULTS_DIR / "figures"
    return base / tag if tag else base


def _jsonl_path(tag: str) -> Path:
    suffix = f"_{tag}" if tag else ""
    return RESULTS_DIR / f"sweep_convergence_driven{suffix}.jsonl"


def _surrogate_stats(surrogates: list[Any]) -> dict[str, list[int]]:
    """POD rank and refit count for each of `surrogates`' replicates, after this
    seed's runs -- `pod_rank` is `surrogate.pod.rank` post-refit (`refit_pod` always
    replaces `.pod` wholesale, so this is never stale), `pod_refit_count` is
    `surrogate._pod_refit_count` (successful, non-no-op `refit_pod()` calls only;
    stops incrementing once `pod_refit_max` is spent). One list entry per replicate,
    not a single number, since replicates can diverge once unsynced."""
    return {
        "pod_rank": [int(s.pod.rank) for s in surrogates],
        "pod_refit_count": [int(s._pod_refit_count) for s in surrogates],
    }


def _save_figures(
    problem: Any,
    *,
    problem_seed: int,
    tag: str,
    seed_surrogate: Any,
    offline_surrogate: Any,
    chains_by_method: dict[str, list[Any]],
    surrogates_by_method: dict[str, list[Any]],
    posterior: dict[str, Any],
) -> None:
    """Saves this seed's four figures (surrogate comparison, posterior scatter, full
    and post-burn-in traces) under `figures/[tag/]`."""
    # Full per-chain surrogate lists -- plot_surrogate_comparison/plot_training_points_scatter
    # both overlay every chain's own surrogate, so agreement (or disagreement) between
    # independently-trained replicates is visible at a glance.
    surrogates_for_plot: dict[str, Any] = {
        "adaptive_surrogate_mcmc": surrogates_by_method["adaptive_surrogate_mcmc"],
        "adaptive_stm": surrogates_by_method["adaptive_stm"],
    }
    surrogate_methods = ("adaptive_surrogate_mcmc", "adaptive_stm")
    if offline_surrogate is not None:
        surrogates_for_plot["pretrained"] = offline_surrogate
        surrogate_methods = ("pretrained", *surrogate_methods)

    figures_dir = _figures_dir(tag)
    figures_dir.mkdir(parents=True, exist_ok=True)
    title_suffix = f" (seed {problem_seed})"

    fig_surrogate = plot_surrogate_comparison(
        problem, seed_surrogate, surrogates_for_plot, methods=surrogate_methods, title_suffix=title_suffix
    )
    fig_surrogate.savefig(figures_dir / f"surrogate_comparison_seed_{problem_seed}.png", bbox_inches="tight")

    fig_training_points = plot_training_points_scatter(
        problem, seed_surrogate, surrogates_for_plot, methods=surrogate_methods, title_suffix=title_suffix
    )
    fig_training_points.savefig(figures_dir / f"training_points_seed_{problem_seed}.png", bbox_inches="tight")

    posterior_methods = ("adaptive_surrogate_mcmc", "adaptive_stm")
    burn_ins = {name: posterior[name]["burn_in"] or 0 for name in ("hf_only", *posterior_methods)}
    fig_posterior = plot_posterior_scatter(
        problem, chains_by_method, burn_ins, methods=posterior_methods, title_suffix=title_suffix
    )
    fig_posterior.savefig(figures_dir / f"posterior_scatter_seed_{problem_seed}.png", bbox_inches="tight")

    traces, trace_burn_ins = prepare_trace_data(
        chains_by_method, posterior, full_resolution_methods=("hf_only", "adaptive_surrogate_mcmc"),
        adaptive_stm_method="adaptive_stm",
    )
    fig_trace_full = plot_traces(problem, traces, trace_burn_ins, mode="full", title_suffix=title_suffix)
    fig_trace_full.savefig(figures_dir / f"trace_full_seed_{problem_seed}.png", bbox_inches="tight")
    fig_trace_post = plot_traces(problem, traces, trace_burn_ins, mode="post_burn_in", title_suffix=title_suffix)
    fig_trace_post.savefig(figures_dir / f"trace_post_burn_in_seed_{problem_seed}.png", bbox_inches="tight")

    import matplotlib.pyplot as plt

    plt.close(fig_surrogate)
    plt.close(fig_training_points)
    plt.close(fig_posterior)
    plt.close(fig_trace_full)
    plt.close(fig_trace_post)


def _save_artifact(problem_seed: int, *, sigma_obs: float, tag: str, **bundle: Any) -> None:
    """Saves this seed's full result bundle to `sweep_artifacts/[tag/]seed_N.joblib`
    (`problem` itself excluded -- `Problem.hf_forward` is an unpicklable closure;
    `load_artifact` reconstructs it from the saved `problem_seed`/`sigma_obs` instead).
    A failure here is logged, not raised, so it can't take down the already-computed
    metrics row the caller still needs to write."""
    artifacts_dir = _artifacts_dir(tag)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    try:
        joblib.dump(
            {"problem_seed": problem_seed, "sigma_obs": sigma_obs, **bundle},
            artifacts_dir / f"seed_{problem_seed}.joblib",
            compress=3,
        )
    except Exception:  # noqa: BLE001 - keep the metrics row even if artifact serialization fails.
        print(f"[seed {problem_seed}] WARNING: failed to save artifact bundle:\n{traceback.format_exc()}")


def run_one_seed(
    problem_seed: int,
    *,
    sigma_obs: float,
    gamma_threshold: float,
    max_adapt_coarse_evals: int,
    n_chains: int,
    convergence: ConvergenceConfig,
    tag: str,
    n_init: int = N_INIT,
    skip_training_cost: bool = False,
    online_learning: OnlineLearningConfig = DEFAULT_ONLINE_LEARNING,
    adaptive_stm_adapt_coarse_evals: int | None = None,
    max_subchain: int = MAX_SUBCHAIN,
) -> dict[str, Any]:
    """Runs the training-cost and posterior-accuracy comparisons for one problem
    instance, saves its figures and full artifact bundle, and returns its `.jsonl`
    row. `skip_training_cost` skips the slow half (`pretrained`'s offline
    greedy-active-learning design in particular) for fast iteration on just the
    posterior/MCMC comparison."""
    problem = build_problem(problem_seed=problem_seed, sigma_obs=sigma_obs)
    seed_surrogate, seed_X, seed_Y = build_initial_surrogate(
        problem, set_seed(1_000 + problem_seed), n_init=n_init, kernel=KERNEL,
        rank_energy_threshold=online_learning.rank_energy_threshold, rank_max=online_learning.rank_max,
    )
    seed_base = problem_seed * SEED_STRIDE

    if skip_training_cost:
        training_cost, offline_surrogate = None, None
    else:
        training_cost, offline_surrogate = run_training_cost_comparison(
            problem, seed_X=seed_X, seed_Y=seed_Y, seed_surrogate=seed_surrogate,
            gamma_threshold=gamma_threshold, kernel=KERNEL,
            max_adapt_coarse_evals=max_adapt_coarse_evals, seed_base=seed_base,
            online_learning=online_learning, max_subchain=max_subchain,
        )

    # methods left at its default (hf_only, adaptive_surrogate_mcmc, adaptive_stm):
    # this sweep always compares exactly those three.
    posterior, chains_by_method, surrogates_by_method = run_convergence_driven_comparison(
        problem, seed_X=seed_X, seed_surrogate=seed_surrogate, n_chains=n_chains,
        gamma_threshold=gamma_threshold, max_adapt_coarse_evals=max_adapt_coarse_evals,
        adaptive_stm_adapt_coarse_evals=adaptive_stm_adapt_coarse_evals,
        convergence=convergence, seed_base=seed_base,
        online_learning=online_learning, max_subchain=max_subchain,
    )

    _save_figures(
        problem, problem_seed=problem_seed, tag=tag, seed_surrogate=seed_surrogate,
        offline_surrogate=offline_surrogate, chains_by_method=chains_by_method,
        surrogates_by_method=surrogates_by_method, posterior=posterior,
    )

    metrics_row = {
        "seed": problem_seed,
        "n_init": n_init,
        "sigma_obs": sigma_obs,
        "gamma_threshold": gamma_threshold,
        "max_adapt_coarse_evals": max_adapt_coarse_evals,
        "adaptive_stm_adapt_coarse_evals": adaptive_stm_adapt_coarse_evals,
        "rank_energy_threshold": online_learning.rank_energy_threshold,
        "rank_max": online_learning.rank_max,
        "pod_refit_every": online_learning.pod_refit_every,
        "pod_refit_max": online_learning.pod_refit_max,
        "max_subchain": max_subchain,
        "burn_in_fraction": convergence.burn_in_fraction,
        "training_cost": training_cost,
        "posterior": posterior,
        "surrogate_stats": {
            "seed_pod_rank": int(seed_surrogate.pod.rank),
            "pretrained_pod_rank": int(offline_surrogate.pod.rank) if offline_surrogate is not None else None,
            "adaptive_surrogate_mcmc": _surrogate_stats(surrogates_by_method["adaptive_surrogate_mcmc"]),
            "adaptive_stm": _surrogate_stats(surrogates_by_method["adaptive_stm"]),
        },
    }

    _save_artifact(
        problem_seed, sigma_obs=sigma_obs, tag=tag,
        seed_surrogate=seed_surrogate, seed_X=seed_X, seed_Y=seed_Y, offline_surrogate=offline_surrogate,
        surrogates_by_method=surrogates_by_method, chains_by_method=chains_by_method,
        training_cost=training_cost, posterior=posterior,
    )

    return metrics_row


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    sweep = parser.add_argument_group("sweep")
    sweep.add_argument("--n-seeds", type=int, default=25, help="Number of problem instances to run.")
    sweep.add_argument("--start-seed", type=int, default=0, help="First problem seed (for resuming a sweep).")
    sweep.add_argument("--tag", type=str, default="", help="Suffix/subdirectory for this run's outputs.")

    design = parser.add_argument_group("surrogate design")
    design.add_argument(
        "--n-init", type=int, default=N_INIT,
        help=f"Size of the shared offline seed design every method starts from (default {N_INIT}).",
    )
    design.add_argument(
        "--pod-refit-every", type=int, default=POD_REFIT_EVERY,
        help="Refit the POD basis and GP hyperparameters every N accumulated HF points during online "
        f"learning (default {POD_REFIT_EVERY}).",
    )
    design.add_argument(
        "--pod-refit-max", type=int, default=None,
        help="Cap on total refit_pod() calls per surrogate lifetime. Unbounded by default.",
    )
    design.add_argument(
        "--rank-energy-threshold", type=float, default=RANK_ENERGY_THRESHOLD,
        help="Cumulative-energy threshold used to adaptively derive the POD rank -- for the initial seed "
        f"surrogate, the offline pretrained design's every batch refit, and refit_pod()'s every online "
        f"refit alike; there's no fixed-rank option (default {RANK_ENERGY_THRESHOLD}).",
    )
    design.add_argument(
        "--rank-max", type=int, default=None,
        help="Upper bound on the adaptively-derived rank, wherever it's derived. 20 by default.",
    )

    problem = parser.add_argument_group("problem / methods")
    problem.add_argument("--sigma-obs", type=float, default=0.1, help="Observation-noise standard deviation.")
    problem.add_argument(
        "--gamma-threshold", type=float, default=GAMMA_THRESHOLD, help="Surrogate-trust threshold."
    )
    problem.add_argument(
        "--max-adapt-coarse-evals", type=int, default=MAX_ADAPT_COARSE_EVALS,
        help="Coarse-eval ceiling for adaptive_stm's adaptive phase, and the HF-call cap for "
        "adaptive_surrogate_mcmc.",
    )
    problem.add_argument(
        "--adaptive-stm-adapt-coarse-evals", type=int, default=None, dest="adaptive_stm_adapt_coarse_evals",
        help="Decouples adaptive_stm's adaptive-phase ceiling from --max-adapt-coarse-evals (they aren't the "
        "same currency: coarse evals vs. real HF calls). Shared with --max-adapt-coarse-evals by default.",
    )
    problem.add_argument(
        "--max-subchain", type=int, default=MAX_SUBCHAIN,
        help=f"Ceiling on adaptive_stm's coarse-to-fine subsampling rate (default {MAX_SUBCHAIN}).",
    )
    problem.add_argument(
        "--skip-training-cost", action="store_true",
        help="Skip the (slow) training-cost comparison and only run the posterior/MCMC comparison.",
    )

    convergence = parser.add_argument_group("convergence loop")
    convergence.add_argument("--n-chains", type=int, default=5, help="Replicate chains per method.")
    convergence.add_argument(
        "--chunk-size", type=int, default=100, help="Coarse evals advanced per round before re-checking convergence."
    )
    convergence.add_argument(
        "--max-total-coarse-evals", type=int, default=15_000, help="Safety cap on coarse-eval cost per method."
    )
    convergence.add_argument("--rhat-threshold", type=float, default=1.01, help="R-hat convergence threshold.")
    convergence.add_argument("--min-ess", type=float, default=400.0, help="Minimum bulk-ESS to declare convergence.")
    convergence.add_argument(
        "--burn-in-fraction", type=float, default=0.1,
        help="Fixed burn-in fraction checked each round (default: discard the first 10%% of each chain).",
    )
    convergence.add_argument(
        "--n-jobs", type=int, default=None,
        help="Parallel workers per replicate loop. Defaults to sequential; e.g. --n-jobs 5 for 5 chains.",
    )
    convergence.add_argument(
        "--parallel-backend", choices=("loky", "threading"), default=None,
        help="Joblib backend for --n-jobs > 1. Use 'threading' if process workers are blocked by the "
        "execution environment.",
    )

    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if not 0.0 <= args.burn_in_fraction < 1.0:
        raise SystemExit("--burn-in-fraction must be in [0, 1).")

    RESULTS_DIR.mkdir(exist_ok=True)
    jsonl_path = _jsonl_path(args.tag)

    # Built once and shared by every seed -- these settings don't vary seed to seed.
    convergence = ConvergenceConfig(
        chunk_size=args.chunk_size, max_total_coarse_evals=args.max_total_coarse_evals,
        rhat_threshold=args.rhat_threshold, min_ess=args.min_ess, n_jobs=args.n_jobs,
        parallel_backend=args.parallel_backend, burn_in_fraction=args.burn_in_fraction,
    )
    online_learning = OnlineLearningConfig(
        pod_refit_every=args.pod_refit_every, pod_refit_max=args.pod_refit_max,
        rank_energy_threshold=args.rank_energy_threshold, rank_max=args.rank_max,
    )

    mode = "a" if args.start_seed > 0 and jsonl_path.exists() else "w"
    with open(jsonl_path, mode) as f_json:
        for i in range(args.start_seed, args.start_seed + args.n_seeds):
            t0 = time.time()
            try:
                row = run_one_seed(
                    i,
                    sigma_obs=args.sigma_obs,
                    gamma_threshold=args.gamma_threshold,
                    max_adapt_coarse_evals=args.max_adapt_coarse_evals,
                    adaptive_stm_adapt_coarse_evals=args.adaptive_stm_adapt_coarse_evals,
                    n_chains=args.n_chains,
                    convergence=convergence,
                    tag=args.tag,
                    n_init=args.n_init,
                    skip_training_cost=args.skip_training_cost,
                    online_learning=online_learning,
                    max_subchain=args.max_subchain,
                )
            except Exception:  # noqa: BLE001 - keep later seeds running after an isolated seed failure.
                print(f"[seed {i}] FAILED:\n{traceback.format_exc()}")
                continue
            f_json.write(json.dumps(row) + "\n")
            f_json.flush()
            dt = time.time() - t0
            tc = row["training_cost"]
            training_cost_str = (
                f"offline_extra_hf={tc['offline']['n_hf_extra']}, online_extra_hf={tc['online']['n_hf_extra']}, "
                if tc is not None else ""
            )
            print(
                f"[seed {i}] done in {dt:.1f}s  ({training_cost_str}"
                f"posterior: hf_only={row['posterior']['hf_only']['converged']}, "
                f"adaptive_surrogate_mcmc={row['posterior']['adaptive_surrogate_mcmc']['converged']}, "
                f"adaptive_stm={row['posterior']['adaptive_stm']['converged']})"
            )

    print(f"\nSaved metrics to {jsonl_path}, figures to {_figures_dir(args.tag)}, artifacts to {_artifacts_dir(args.tag)}")


if __name__ == "__main__":
    main()
