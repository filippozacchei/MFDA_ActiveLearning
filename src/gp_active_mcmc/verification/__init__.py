"""Problem-agnostic MCMC-comparison harness for surrogate-assisted Bayesian inversion:
four methods (`run_hf_only`, `run_pretrained`, `run_online_active`, `run_adaptive_da`)
plus R-hat-driven multi-chain orchestration, reusable by any `Problem`-shaped inverse
problem, extracted from the mass-spring-damper example for that reuse.
"""

from __future__ import annotations

from gp_active_mcmc.verification.design import (
    DEFAULT_ONLINE_LEARNING,
    OnlineLearningConfig,
    PODRankSelection,
    active_learning_offline_design,
    build_initial_surrogate,
    select_pod_rank_and_seed_design,
)
from gp_active_mcmc.verification.harness import (
    print_convergence_driven_table,
    run_convergence_driven_comparison,
)
from gp_active_mcmc.verification.methods import (
    run_adaptive_da,
    run_hf_only,
    run_online_active,
    run_pretrained,
    run_training_cost_comparison,
)
from gp_active_mcmc.verification.metrics import (
    adaptive_da_coarse_eval_units,
    adaptive_da_cumulative_coarse_evals,
    adaptive_da_full_resolution_trace,
    adaptive_da_multichain_trace,
    effective_burn_in,
    find_burn_in_via_rhat,
    gaussian_kl,
    gaussian_wasserstein2,
    multichain_diagnostics,
    pooled_summarize,
    prepare_trace_data,
    summarize,
)
from gp_active_mcmc.verification.problem import Problem
from gp_active_mcmc.verification.rounds import ConvergenceConfig, run_until_rhat_converged
from gp_active_mcmc.verification.sampling import make_proposal

__all__ = [
    "DEFAULT_ONLINE_LEARNING",
    "ConvergenceConfig",
    "OnlineLearningConfig",
    "PODRankSelection",
    "Problem",
    "active_learning_offline_design",
    "adaptive_da_coarse_eval_units",
    "adaptive_da_cumulative_coarse_evals",
    "adaptive_da_full_resolution_trace",
    "adaptive_da_multichain_trace",
    "build_initial_surrogate",
    "effective_burn_in",
    "find_burn_in_via_rhat",
    "gaussian_kl",
    "gaussian_wasserstein2",
    "make_proposal",
    "multichain_diagnostics",
    "pooled_summarize",
    "prepare_trace_data",
    "print_convergence_driven_table",
    "run_adaptive_da",
    "run_convergence_driven_comparison",
    "run_hf_only",
    "run_online_active",
    "run_pretrained",
    "run_training_cost_comparison",
    "run_until_rhat_converged",
    "select_pod_rank_and_seed_design",
    "summarize",
]
