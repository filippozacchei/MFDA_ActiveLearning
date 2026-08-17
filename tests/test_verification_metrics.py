from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from gp_active_mcmc.inference import MCMCChain
from gp_active_mcmc.verification.metrics import (
    adaptive_da_coarse_eval_units,
    adaptive_da_cumulative_coarse_evals,
    effective_burn_in,
    find_burn_in_via_rhat,
    gaussian_kl,
    gaussian_wasserstein2,
    multichain_diagnostics,
    pooled_summarize,
    prepare_trace_data,
    summarize,
)


@dataclass
class _DummyProblem:
    theta_true: np.ndarray


def _hand_rolled_kl(mean_a: np.ndarray, cov_a: np.ndarray, mean_b: np.ndarray, cov_b: np.ndarray) -> float:
    k = mean_a.shape[0]
    cov_b_inv = np.linalg.inv(cov_b)
    diff = mean_b - mean_a
    term_trace = np.trace(cov_b_inv @ cov_a)
    term_quad = diff @ cov_b_inv @ diff
    term_logdet = np.log(np.linalg.det(cov_b)) - np.log(np.linalg.det(cov_a))
    return float(0.5 * (term_trace + term_quad - k + term_logdet))


def _chain(samples: np.ndarray, *, used_hf: np.ndarray | None = None) -> MCMCChain:
    return MCMCChain.from_arrays(samples=samples, used_hf=used_hf)


# ---------------------------------------------------------------------------
# gaussian_wasserstein2 / gaussian_kl
# ---------------------------------------------------------------------------


def test_gaussian_wasserstein2_zero_for_identical_gaussians() -> None:
    mean = np.array([1.0, -2.0])
    cov = np.array([[2.0, 0.3], [0.3, 1.0]])
    assert gaussian_wasserstein2(mean, cov, mean, cov) == pytest.approx(0.0, abs=1e-5)


def test_gaussian_wasserstein2_matches_known_1d_formula() -> None:
    mean_a, mean_b = np.array([0.0]), np.array([3.0])
    cov_a, cov_b = np.array([[4.0]]), np.array([[1.0]])
    # 1-D closed form: (m_a - m_b)^2 + (sigma_a - sigma_b)^2
    expected = np.sqrt((0.0 - 3.0) ** 2 + (2.0 - 1.0) ** 2)
    assert gaussian_wasserstein2(mean_a, cov_a, mean_b, cov_b) == pytest.approx(expected, rel=1e-6)


def test_gaussian_kl_zero_for_identical_gaussians() -> None:
    mean = np.array([0.5, 1.5])
    cov = np.array([[1.0, 0.1], [0.1, 0.5]])
    assert gaussian_kl(mean, cov, mean, cov) == pytest.approx(0.0, abs=1e-8)


def test_gaussian_kl_matches_hand_rolled_reference() -> None:
    mean_a, mean_b = np.array([0.0, 0.0]), np.array([1.0, -1.0])
    cov_a = np.array([[1.0, 0.0], [0.0, 1.0]])
    cov_b = np.array([[2.0, 0.2], [0.2, 1.5]])
    assert gaussian_kl(mean_a, cov_a, mean_b, cov_b) == pytest.approx(
        _hand_rolled_kl(mean_a, cov_a, mean_b, cov_b), rel=1e-6
    )


# ---------------------------------------------------------------------------
# summarize / pooled_summarize
# ---------------------------------------------------------------------------


def test_summarize_computes_hf_fraction_and_rmse() -> None:
    problem = _DummyProblem(theta_true=np.array([1.0, 1.0]))
    samples = np.tile(np.array([1.0, 1.0]), (10, 1))
    used_hf = np.array([True, False, True, False, False, False, False, False, False, False])
    chain = _chain(samples, used_hf=used_hf)

    out = summarize(chain, problem, burn_in=0, n_offline_hf=5)
    assert out["n_offline_hf"] == 5
    assert out["n_hf_calls_online"] == 2
    assert out["n_hf_calls"] == 7
    assert out["hf_call_fraction"] == pytest.approx(7 / (5 + 10))
    assert out["rmse_to_truth"] == pytest.approx(0.0, abs=1e-8)


def test_pooled_summarize_shared_n_offline_hf_counted_once() -> None:
    # Regression test for a real, previously-fixed bug: an int n_offline_hf is a
    # *shared* one-time cost (e.g. a shared offline seed design), not one paid per
    # chain -- n_init=25 across 5 chains must total 25, not 125.
    problem = _DummyProblem(theta_true=np.array([0.0, 0.0]))
    chains = [_chain(np.zeros((10, 2))) for _ in range(5)]
    out = pooled_summarize(chains, problem, burn_in=0, n_offline_hf=25)
    assert out["n_offline_hf_total"] == 25
    assert out["n_hf_calls"] == 25


def test_pooled_summarize_per_chain_n_offline_hf_list_sums() -> None:
    problem = _DummyProblem(theta_true=np.array([0.0, 0.0]))
    chains = [_chain(np.zeros((10, 2))) for _ in range(3)]
    out = pooled_summarize(chains, problem, burn_in=0, n_offline_hf=[10, 20, 30])
    assert out["n_offline_hf_total"] == 60
    assert out["n_hf_calls"] == 60


def test_pooled_summarize_n_offline_hf_list_length_mismatch_raises() -> None:
    problem = _DummyProblem(theta_true=np.array([0.0, 0.0]))
    chains = [_chain(np.zeros((10, 2))) for _ in range(2)]
    with pytest.raises(ValueError):
        pooled_summarize(chains, problem, burn_in=0, n_offline_hf=[10, 20, 30])


# ---------------------------------------------------------------------------
# multichain_diagnostics / find_burn_in_via_rhat
# ---------------------------------------------------------------------------


def _well_mixed_chains(*, n_chains: int = 4, n_steps: int = 500, seed: int = 0) -> list[MCMCChain]:
    rng = np.random.default_rng(seed)
    return [_chain(rng.normal(size=(n_steps, 2))) for _ in range(n_chains)]


def test_multichain_diagnostics_requires_param_names_no_default() -> None:
    chains = _well_mixed_chains()
    with pytest.raises(TypeError):
        multichain_diagnostics(chains, burn_in=0)  # type: ignore[call-arg]


def test_multichain_diagnostics_rhat_near_one_for_identical_chains() -> None:
    rng = np.random.default_rng(0)
    samples = rng.normal(size=(500, 2))
    chains = [_chain(samples) for _ in range(4)]
    diag = multichain_diagnostics(chains, burn_in=0, param_names=("a", "b"))
    assert diag["n_chains"] == 4
    for name in ("a", "b"):
        assert diag["rhat"][name] == pytest.approx(1.0, abs=0.05)


def test_find_burn_in_via_rhat_finds_early_burn_in_for_well_mixed_chains() -> None:
    chains = _well_mixed_chains()
    result = find_burn_in_via_rhat(chains, param_names=("a", "b"))
    assert result["converged"]
    assert result["burn_in"] is not None
    assert result["burn_in"] < chains[0].n_steps // 2


def test_find_burn_in_via_rhat_respects_min_burn_in_floor() -> None:
    chains = _well_mixed_chains()
    floor = 200
    result = find_burn_in_via_rhat(chains, param_names=("a", "b"), min_burn_in=floor)
    assert result["burn_in"] is not None
    assert result["burn_in"] >= floor


# ---------------------------------------------------------------------------
# effective_burn_in / adaptive_da_* helpers
# ---------------------------------------------------------------------------


def test_effective_burn_in_requires_meta_for_adaptive_da() -> None:
    with pytest.raises(ValueError):
        effective_burn_in("adaptive_da", default=10)


def test_effective_burn_in_default_for_other_methods() -> None:
    assert effective_burn_in("online_active", default=42) == 42


def test_effective_burn_in_uses_n_adapt_samples_for_adaptive_da() -> None:
    meta = {"n_adapt_samples": 77}
    assert effective_burn_in("adaptive_da", meta_adaptive_da=meta, default=10) == 77


def test_adaptive_da_coarse_eval_units_adapt_only() -> None:
    meta: dict[str, Any] = {"phase": "adapt_only", "adapt_metadata": {"coarse_evals_used": 123}}
    assert adaptive_da_coarse_eval_units(meta) == 123


def test_adaptive_da_coarse_eval_units_with_production() -> None:
    meta: dict[str, Any] = {
        "phase": "adapt_then_production",
        "adapt_metadata": {"coarse_evals_used": 100},
        "production_metadata": {"iterations": 20},
        "frozen_subsampling_rate": 5,
    }
    assert adaptive_da_coarse_eval_units(meta) == 100 + 20 * 5


def test_adaptive_da_cumulative_coarse_evals_monotonic() -> None:
    meta: dict[str, Any] = {
        "n_adapt_samples": 10,
        "phase": "adapt_then_production",
        "adapt_metadata": {"coarse_evals_used": 100},
        "frozen_subsampling_rate": 5,
        "n_production_samples": 4,
    }
    x = adaptive_da_cumulative_coarse_evals(meta)
    assert x.shape == (14,)
    assert np.all(np.diff(x) > 0)
    assert x[-1] == 100 + 5 * 4


# ---------------------------------------------------------------------------
# prepare_trace_data
# ---------------------------------------------------------------------------


def test_prepare_trace_data_uses_configurable_method_names() -> None:
    n_steps = 5
    chains_by_method = {
        "method_a": [_chain(np.zeros((n_steps, 2)))],
        "c": [_chain(np.zeros((3, 2)))],
        "c_adapt": [_chain(np.zeros((2, 2)))],
    }
    posterior = {
        "method_a": {"burn_in": 1},
        "c": {
            "burn_in": 1,
            "adapt_metas": [{"adapt_coarse_evals_used": 10}],
            "coarse_evals_per_chain": [15],
        },
    }
    traces, burn_ins_x = prepare_trace_data(
        chains_by_method, posterior, full_resolution_methods=("method_a",),
        adaptive_da_method="c", adaptive_da_adapt_key="c_adapt",
    )
    # No hardcoded "hf_only"/"online_active"/"ours" literal survives: only the
    # caller-supplied names appear.
    assert set(traces.keys()) == {"method_a", "c"}
    assert set(burn_ins_x.keys()) == {"method_a", "c"}
    assert "hf_only" not in traces
    assert "ours" not in traces
