"""Multi-chain convergence check: data-driven burn-in via R-hat, then pooled posterior.

For a single representative problem instance and n_init setting, launches N_CHAINS
independent chains *from the same theta0* per method, finds each method's burn-in via
a Gelman-Rubin R-hat sweep, and reports R-hat/ESS plus a pooled (across chains)
posterior summary at that burn-in.

The burn-in search doubles as an efficiency metric: it's literally "how many
iterations does this method need before its chains agree with each other", directly
comparable across methods.

For `ours`, the search never considers burn-ins shorter than the adaptive phase's
length (max across the replicate chains): the target is non-stationary during that
phase (the surrogate is still being learned), so no amount of burn-in trimmed from
*within* it can fix that -- the whole phase must be discarded regardless of what R-hat
says about it.

All the orchestration lives in `msd_methods.run_convergence_check` (shared with
`msd_benchmark.ipynb`, so the two can't drift out of sync).

Run from `examples/toy_problem/`:
    python check_convergence.py
"""

from __future__ import annotations

import json

from msd_methods import (
    KERNEL,
    MAX_ADAPT_COARSE_EVALS,
    PARAM_NAMES,
    POD_RANK,
    RESULTS_DIR,
    build_initial_surrogate,
    build_problem,
    print_convergence_table,
    run_convergence_check,
)

from gp_active_mcmc.utils.rng import set_seed

PROBLEM_SEED = 0
N_INIT = 20
N_CHAINS = 5
N_COARSE_EVALS_LONG = 10_000  # long horizon so R-hat has room to actually resolve
RHAT_THRESHOLD = 1.01


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    problem = build_problem(problem_seed=PROBLEM_SEED, sigma_obs=0.02)
    print("theta_true:", dict(zip(PARAM_NAMES, problem.theta_true.tolist(), strict=True)))
    print(f"N_CHAINS={N_CHAINS}, N_COARSE_EVALS_LONG={N_COARSE_EVALS_LONG}, MAX_ADAPT_COARSE_EVALS={MAX_ADAPT_COARSE_EVALS}")

    seed_surrogate, seed_X, seed_Y = build_initial_surrogate(
        problem, set_seed(1_000 + N_INIT), n_init=N_INIT, pod_rank=POD_RANK, kernel=KERNEL
    )

    results, _chains_by_method, _surrogates_by_method, _ours_metas = run_convergence_check(
        problem,
        n_init=N_INIT,
        pod_rank=POD_RANK,
        seed_X=seed_X,
        seed_Y=seed_Y,
        seed_surrogate=seed_surrogate,
        n_chains=N_CHAINS,
        n_coarse_evals=N_COARSE_EVALS_LONG,
        rhat_threshold=RHAT_THRESHOLD,
    )

    for name, r in results.items():
        status = "converged" if r["converged"] else "NOT stably converged in range tried"
        print(f"\n{name}: burn_in={r['burn_in']} ({status})")
        if r["rhat"] is not None:
            print(f"  rhat={r['rhat']}  ess_bulk={r['ess_bulk']}")
            print(f"  pooled posterior mean={r['pooled']['posterior_mean']}  rmse={r['pooled']['rmse_to_truth']:.4f}")
            print(f"  pooled hf_calls={r['pooled']['n_hf_calls']}  hf_fraction={r['pooled']['hf_call_fraction']:.3f}")

    out_path = RESULTS_DIR / "convergence_check.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out_path}")

    print()
    print_convergence_table(results)


if __name__ == "__main__":
    main()
