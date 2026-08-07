"""Multi-seed x initial-design-size sweep on the mass-spring-damper benchmark.

For each seed, the HF-only reference chains are computed once (they do not depend on
the offline design), then methods 2-4 are run for each `n_init` setting in
`N_INIT_SETTINGS`. Results are appended row-by-row (one row per seed x n_init x method)
to a CSV/JSON pair in `results/`, so a partial run is still usable if interrupted.

Run from `examples/toy_problem/`:
    python run_sweep_msd.py --n-seeds 10
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Any

from msd_methods import (
    BURN_IN,
    GAMMA_THRESHOLD,
    KERNEL,
    MAX_ADAPT_COARSE_EVALS,
    N_COARSE_EVALS,
    N_INIT_SETTINGS,
    N_REF_ITERATIONS,
    POD_RANK,
    REF_BURN_IN,
    RESULTS_DIR,
    Problem,
    build_initial_surrogate,
    build_problem,
    effective_burn_in,
    ours_coarse_eval_units,
    run_hf_only,
    run_online_active,
    run_ours,
    run_pretrained,
    summarize,
)

from gp_active_mcmc.utils.rng import set_seed

SIGMA_OBS = 0.02

SWEEP_CSV = RESULTS_DIR / "sweep_results.csv"
SWEEP_JSON = RESULTS_DIR / "sweep_results.jsonl"


ROW_FIELDS = (
    "seed",
    "n_init",
    "method",
    "n_hf_calls",
    "hf_call_fraction",
    "rmse_to_truth",
    "posterior_mean",
    "posterior_std",
    "mean_abs_dev_from_reference",
    "converged",
    "frozen_subsampling_rate",
    "adapt_coarse_evals_used",
)


def _row(*, seed: int, n_init: int, method: str, summary: dict[str, Any], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    row: dict[str, Any] = dict.fromkeys(ROW_FIELDS)
    row.update(
        {
            "seed": seed,
            "n_init": n_init,
            "method": method,
            "n_hf_calls": summary.get("n_hf_calls"),
            "hf_call_fraction": summary.get("hf_call_fraction"),
            "rmse_to_truth": summary["rmse_to_truth"],
            "posterior_mean": summary["posterior_mean"],
            "posterior_std": summary["posterior_std"],
            "mean_abs_dev_from_reference": summary.get("mean_abs_dev_from_reference"),
        }
    )
    if extra:
        row.update(extra)
    return row


def run_one_seed(problem_seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    problem: Problem = build_problem(problem_seed=problem_seed, sigma_obs=SIGMA_OBS)

    # HF-only chains do not depend on n_init: compute once per seed.
    chain_ref = run_hf_only(problem, iterations=N_REF_ITERATIONS, seed=10_000 + problem_seed)
    ref_mean = chain_ref.burn_in(REF_BURN_IN).samples.mean(axis=0)
    rows.append(
        _row(
            seed=problem_seed,
            n_init=-1,
            method="hf_only_reference",
            summary=summarize(chain_ref, problem, burn_in=REF_BURN_IN),
        )
    )

    chain_hf_matched = run_hf_only(problem, iterations=N_COARSE_EVALS, seed=11_000 + problem_seed)
    rows.append(
        _row(
            seed=problem_seed,
            n_init=-1,
            method="hf_only_matched",
            summary=summarize(chain_hf_matched, problem, burn_in=BURN_IN, reference_mean=ref_mean),
        )
    )

    for n_init in N_INIT_SETTINGS:
        seed_surrogate, seed_X, seed_Y = build_initial_surrogate(
            problem, set_seed(1_000 + problem_seed * 100 + n_init), n_init=n_init, pod_rank=POD_RANK, kernel=KERNEL
        )

        chain_ours, meta_ours, _surrogate_ours = run_ours(
            problem,
            surrogate=seed_surrogate,
            gamma_threshold=GAMMA_THRESHOLD,
            n_coarse_evals=N_COARSE_EVALS,
            max_adapt_coarse_evals=MAX_ADAPT_COARSE_EVALS,
            seed=12_000 + problem_seed * 100 + n_init,
        )
        summary_ours = summarize(
            chain_ours,
            problem,
            burn_in=effective_burn_in("ours", meta_ours=meta_ours),
            reference_mean=ref_mean,
            n_offline_hf=n_init,
            n_coarse_eval_units=ours_coarse_eval_units(meta_ours),
        )
        rows.append(
            _row(
                seed=problem_seed,
                n_init=n_init,
                method="ours",
                summary=summary_ours,
                extra={
                    "converged": meta_ours.get("converged"),
                    "frozen_subsampling_rate": meta_ours.get("frozen_subsampling_rate"),
                    "adapt_coarse_evals_used": meta_ours.get("adapt_coarse_evals_used"),
                },
            )
        )

        chain_pretrained, n_hf_pretrained, _surrogate_pretrained = run_pretrained(
            problem,
            seed_X=seed_X,
            seed_Y=seed_Y,
            gamma_threshold=GAMMA_THRESHOLD,
            pod_rank=POD_RANK,
            kernel=KERNEL,
            iterations=N_COARSE_EVALS,
            seed=13_000 + problem_seed * 100 + n_init,
        )
        summary_pretrained = summarize(
            chain_pretrained, problem, burn_in=BURN_IN, reference_mean=ref_mean, n_offline_hf=n_hf_pretrained
        )
        rows.append(_row(seed=problem_seed, n_init=n_init, method="pretrained", summary=summary_pretrained))

        chain_online, _surrogate_online = run_online_active(
            problem,
            surrogate=seed_surrogate,
            gamma_threshold=GAMMA_THRESHOLD,
            iterations=N_COARSE_EVALS,
            seed=14_000 + problem_seed * 100 + n_init,
        )
        rows.append(
            _row(
                seed=problem_seed,
                n_init=n_init,
                method="online_active",
                summary=summarize(chain_online, problem, burn_in=BURN_IN, reference_mean=ref_mean, n_offline_hf=n_init),
            )
        )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--start-seed", type=int, default=0)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)

    import csv

    def _write_csv(rows: list[dict[str, Any]]) -> None:
        with open(SWEEP_CSV, "w", newline="") as f_csv:
            writer = csv.DictWriter(f_csv, fieldnames=list(ROW_FIELDS))
            writer.writeheader()
            for row in rows:
                flat = {k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in row.items()}
                writer.writerow(flat)

    all_rows: list[dict[str, Any]] = []
    with open(SWEEP_JSON, "w") as f_json:
        for i in range(args.start_seed, args.start_seed + args.n_seeds):
            t0 = time.time()
            try:
                rows = run_one_seed(i)
            except Exception:
                print(f"[seed {i}] FAILED:\n{traceback.format_exc()}")
                continue
            for row in rows:
                f_json.write(json.dumps(row) + "\n")
            f_json.flush()
            all_rows.extend(rows)
            _write_csv(all_rows)  # rewritten after every seed so progress is inspectable mid-run
            print(f"[seed {i}] done in {time.time() - t0:.1f}s ({len(rows)} rows, {len(all_rows)} total)")

    print(f"\nSaved {len(all_rows)} rows to {SWEEP_JSON} and {SWEEP_CSV}")


if __name__ == "__main__":
    main()
