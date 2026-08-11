"""Multi-seed sweep of the paper's two headline comparisons, at the fixed `N_INIT`/
`POD_RANK` (see `msd_methods.py`'s comments on those constants for how/why they were
chosen -- separately, not from the same sample, and why `N_INIT` is deliberately small
enough that active learning is actually necessary).

For each of `--n-seeds` independent problem instances (fresh `theta_true`/`y_obs` and a
freshly-drawn `n_init`-sized offline seed design each), computes:

1. **Training-cost comparison** (`run_training_cost_comparison`): online, MCMC-path-
   guided active learning (`ours`'s adaptive phase) vs. offline, global greedy-max-
   variance active learning (`pretrained`'s training procedure) -- HF calls and
   wall-clock to reach each one's own convergence criterion, no downstream MCMC
   production phase on either side. This is the comparison that isolates *training
   strategy* (see that function's docstring for why the posterior-accuracy comparison
   below deliberately excludes `pretrained`: once frozen and DA-corrected, a surrogate
   targets the true posterior regardless of how it was trained, so comparing posteriors
   wouldn't isolate training strategy at all).

2. **Posterior-accuracy comparison** (`run_convergence_driven_comparison`, restricted
   to `methods=("hf_only", "online_active", "ours")` -- `pretrained` is skipped
   entirely, not just excluded from the table, so its expensive replicate loop is never
   run): cost (HF calls, coarse evals) to reach R-hat<=`--rhat-threshold` and
   ESS>=`--min-ess` across `--n-chains` replicate chains, alongside the resulting
   accuracy (W2/KL/RMSE vs. the HF-only reference).

Both are appended, one JSON object per seed, to `results/sweep_convergence_driven.jsonl`
(one line each) so a partial run is still usable if interrupted -- re-running with
`--start-seed` picks up where a previous run left off.

Run from `examples/toy_problem/` (this is slow -- expect hours for `--n-seeds 25` at the
defaults; run in the background):
    python run_sweep_convergence_driven.py --n-seeds 25
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Any

from msd_methods import (
    GAMMA_THRESHOLD,
    KERNEL,
    MAX_ADAPT_COARSE_EVALS,
    N_INIT,
    POD_RANK,
    RESULTS_DIR,
    build_initial_surrogate,
    build_problem,
    run_convergence_driven_comparison,
    run_training_cost_comparison,
)

from gp_active_mcmc.utils.rng import set_seed

SIGMA_OBS = 0.1
SWEEP_JSONL = RESULTS_DIR / "sweep_convergence_driven.jsonl"

# Per-seed offsets are spaced 10_000 apart so a single seed's internal seeding (which
# uses offsets up to +600 -- see run_training_cost_comparison/run_convergence_driven_
# comparison's internals) never collides with the next seed's.
SEED_STRIDE = 10_000


def run_one_seed(
    problem_seed: int,
    *,
    n_chains: int,
    chunk_size: int,
    max_total_coarse_evals: int,
    rhat_threshold: float,
    min_ess: float,
    n_jobs: int | None,
) -> dict[str, Any]:
    problem = build_problem(problem_seed=problem_seed, sigma_obs=SIGMA_OBS)
    seed_surrogate, seed_X, seed_Y = build_initial_surrogate(
        problem, set_seed(1_000 + problem_seed), n_init=N_INIT, pod_rank=POD_RANK, kernel=KERNEL
    )

    seed_base = problem_seed * SEED_STRIDE

    t0 = time.time()
    training_cost = run_training_cost_comparison(
        problem, seed_X=seed_X, seed_Y=seed_Y, seed_surrogate=seed_surrogate,
        gamma_threshold=GAMMA_THRESHOLD, pod_rank=POD_RANK,
        max_adapt_coarse_evals=MAX_ADAPT_COARSE_EVALS, seed_base=seed_base,
    )
    training_cost_time = time.time() - t0

    t0 = time.time()
    posterior_results, _chains, _surrogates = run_convergence_driven_comparison(
        problem, n_init=N_INIT, pod_rank=POD_RANK, seed_X=seed_X, seed_Y=seed_Y,
        seed_surrogate=seed_surrogate, n_chains=n_chains, chunk_size=chunk_size,
        max_total_coarse_evals=max_total_coarse_evals, rhat_threshold=rhat_threshold,
        min_ess=min_ess, seed_base=seed_base, n_jobs=n_jobs,
        methods=("hf_only", "online_active", "ours"),
    )
    posterior_time = time.time() - t0

    return {
        "seed": problem_seed,
        "n_init": N_INIT,
        "pod_rank": POD_RANK,
        "training_cost": training_cost,
        "training_cost_wall_time_s": training_cost_time,
        "posterior": posterior_results,
        "posterior_wall_time_s": posterior_time,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=25)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--n-chains", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--max-total-coarse-evals", type=int, default=10_000)
    parser.add_argument("--rhat-threshold", type=float, default=1.01)
    parser.add_argument("--min-ess", type=float, default=400.0)
    parser.add_argument("--n-jobs", type=int, default=None)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)

    mode = "a" if args.start_seed > 0 and SWEEP_JSONL.exists() else "w"
    with open(SWEEP_JSONL, mode) as f_json:
        for i in range(args.start_seed, args.start_seed + args.n_seeds):
            t0 = time.time()
            try:
                row = run_one_seed(
                    i,
                    n_chains=args.n_chains,
                    chunk_size=args.chunk_size,
                    max_total_coarse_evals=args.max_total_coarse_evals,
                    rhat_threshold=args.rhat_threshold,
                    min_ess=args.min_ess,
                    n_jobs=args.n_jobs,
                )
            except Exception:
                print(f"[seed {i}] FAILED:\n{traceback.format_exc()}")
                continue
            f_json.write(json.dumps(row) + "\n")
            f_json.flush()
            dt = time.time() - t0
            tc = row["training_cost"]
            print(
                f"[seed {i}] done in {dt:.1f}s  "
                f"(offline_extra_hf={tc['offline']['n_hf_extra']}, "
                f"online_extra_hf={tc['online']['n_hf_extra']}, "
                f"online_converged={tc['online']['converged']}, "
                f"posterior: hf_only={row['posterior']['hf_only']['converged']}, "
                f"online_active={row['posterior']['online_active']['converged']}, "
                f"ours={row['posterior']['ours']['converged']})"
            )

    print(f"\nSaved to {SWEEP_JSONL}")


if __name__ == "__main__":
    main()
