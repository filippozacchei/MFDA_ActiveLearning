"""Multi-seed sweep producing the paper's two headline comparisons, at the fixed
`N_INIT`/`POD_RANK` (see `msd_methods.py`'s comments on those constants).

For each of `--n-seeds` independent problem instances (fresh `theta_true`/`y_obs` and a
freshly-drawn `N_INIT`-sized offline seed design each), computes:

1. **Training-cost comparison**: online, MCMC-path-guided active learning (`ours`'s
   adaptive phase) vs. offline, global greedy-max-variance active learning
   (`pretrained`'s training procedure) -- HF calls and wall-clock to reach each one's
   own convergence criterion.
2. **Posterior-accuracy comparison**: `hf_only`, `online_active`, and `ours` (methods
   argument to `run_convergence_driven_comparison` -- `pretrained` is skipped
   entirely). Cost to reach R-hat<=`--rhat-threshold`/ESS>=`--min-ess` across
   `--n-chains` replicate chains, alongside accuracy (W2/KL/RMSE vs. the `hf_only`
   reference).

Per-seed metrics are appended, one JSON object per line, to
`results/sweep_convergence_driven.jsonl` (partial runs are usable; `--start-seed`
resumes). Per-seed figures (surrogate-prediction comparison, posterior scatter) are
saved to `results/figures/`.

Run from `examples/toy_problem/` (slow -- expect roughly an hour for `--n-seeds 25` at
the defaults; run in the background):
    python run_sweep_convergence_driven.py --n-seeds 25
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from typing import Any

import matplotlib

matplotlib.use("Agg")

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
from msd_plots import plot_posterior_scatter, plot_surrogate_comparison

from gp_active_mcmc.utils.rng import set_seed

SIGMA_OBS = 0.1
SWEEP_JSONL = RESULTS_DIR / "sweep_convergence_driven.jsonl"
FIGURES_DIR = RESULTS_DIR / "figures"

# Per-seed offsets are spaced 10_000 apart so a single seed's internal seeding (which
# uses offsets up to +600) never collides with the next seed's.
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

    training_cost, offline_surrogate = run_training_cost_comparison(
        problem, seed_X=seed_X, seed_Y=seed_Y, seed_surrogate=seed_surrogate,
        gamma_threshold=GAMMA_THRESHOLD, pod_rank=POD_RANK,
        max_adapt_coarse_evals=MAX_ADAPT_COARSE_EVALS, seed_base=seed_base,
    )

    posterior, chains_by_method, surrogates_by_method = run_convergence_driven_comparison(
        problem, n_init=N_INIT, pod_rank=POD_RANK, seed_X=seed_X, seed_Y=seed_Y,
        seed_surrogate=seed_surrogate, n_chains=n_chains, chunk_size=chunk_size,
        max_total_coarse_evals=max_total_coarse_evals, rhat_threshold=rhat_threshold,
        min_ess=min_ess, seed_base=seed_base, n_jobs=n_jobs,
        methods=("hf_only", "online_active", "ours"),
    )

    fig_surrogate = plot_surrogate_comparison(
        problem, seed_surrogate,
        {
            "pretrained": offline_surrogate,
            "online_active": surrogates_by_method["online_active"][0],
            "ours": surrogates_by_method["ours"][0],
        },
        title_suffix=f" (seed {problem_seed})",
    )
    fig_surrogate.savefig(FIGURES_DIR / f"surrogate_comparison_seed_{problem_seed}.png", bbox_inches="tight")

    burn_ins = {name: posterior[name]["burn_in"] or 0 for name in ("hf_only", "online_active", "ours")}
    fig_posterior = plot_posterior_scatter(
        problem, chains_by_method, burn_ins, title_suffix=f" (seed {problem_seed})"
    )
    fig_posterior.savefig(FIGURES_DIR / f"posterior_scatter_seed_{problem_seed}.png", bbox_inches="tight")

    import matplotlib.pyplot as plt

    plt.close(fig_surrogate)
    plt.close(fig_posterior)

    return {
        "seed": problem_seed,
        "n_init": N_INIT,
        "pod_rank": POD_RANK,
        "training_cost": training_cost,
        "posterior": posterior,
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
    FIGURES_DIR.mkdir(exist_ok=True)

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
                f"posterior: hf_only={row['posterior']['hf_only']['converged']}, "
                f"online_active={row['posterior']['online_active']['converged']}, "
                f"ours={row['posterior']['ours']['converged']})"
            )

    print(f"\nSaved metrics to {SWEEP_JSONL}, figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
