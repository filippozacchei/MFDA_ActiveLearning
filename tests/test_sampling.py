from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from gp_active_mcmc.inference.chain import MCMCChain, SamplingResult
from gp_active_mcmc.inference.sampling import ChunkedMCMCConfig, sample_active_chain


class _Link:
    def __init__(self, parameters: np.ndarray):
        self.parameters = parameters


class _Model:
    def __init__(self, used_hf: list[bool]):
        class _Log:
            def __init__(self, used_hf: list[bool]):
                self.used_hf = used_hf

        self.log = _Log(used_hf)


def test_sample_active_chain_builds_result(monkeypatch: pytest.MonkeyPatch) -> None:
    # fake tinyDA.sample output
    def _fake_sample(**kwargs):
        return {"chain_0": [_Link(np.array([0.0, 0.0])), _Link(np.array([1.0, 0.0]))]}

    import gp_active_mcmc.inference.sampling as sampling_mod

    monkeypatch.setattr(sampling_mod.tda, "sample", _fake_sample)

    model = _Model([False, True])
    result = sample_active_chain(
        model=model,
        posterior=[],
        proposal=object(),  # unused by fake
        iterations=2,
        initial_parameters=np.array([0.0, 0.0]),
        subsampling_rate=1,
        chain_key="chain_0",
    )

    assert result.chain.samples.shape == (2, 2)
    assert result.chain.extras.used_hf is not None
    assert result.chain.extras.used_hf.shape == (2,)
    assert bool(result.chain.extras.used_hf[1]) is True
    assert result.metadata["iterations"] == 2
    assert result.metadata["subsampling_rate"] == 1


class _FakeAdaptiveHook:
    """Minimal stand-in for AdaptiveSubchain: converges as soon as asked."""

    def __init__(self, *, subchain_length: int, converged: bool):
        self.state = type("_State", (), {"subchain_length": subchain_length})()
        self._converged = converged

    def has_converged(self) -> bool:
        return self._converged


def test_sample_adaptive_then_frozen_chain_orchestrates_and_stitches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gp_active_mcmc.inference.sampling as sampling_mod

    adapt_samples = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    adapt_used_hf = np.array([False, True, False, True])
    adapt_subchain = np.array([2, 2, 2, 2])
    adapt_chain = MCMCChain.from_arrays(
        samples=adapt_samples, used_hf=adapt_used_hf, subchain_length=adapt_subchain
    )
    adapt_metadata = {"coarse_evals_used": 8, "stopped_early": True}

    prod_samples = np.array([[4.0, 0.0], [5.0, 0.0]])
    prod_used_hf = np.array([True, False])
    prod_chain = MCMCChain.from_arrays(samples=prod_samples, used_hf=prod_used_hf)
    prod_metadata = {"iterations": 2, "subsampling_rate": 2}

    captured: dict[str, Any] = {}

    def _fake_adaptive(**kwargs: Any) -> SamplingResult:
        captured["adaptive_kwargs"] = kwargs
        return SamplingResult(chain=adapt_chain, metadata=adapt_metadata)

    def _fake_active(**kwargs: Any) -> SamplingResult:
        captured["active_kwargs"] = kwargs
        return SamplingResult(chain=prod_chain, metadata=prod_metadata)

    monkeypatch.setattr(sampling_mod, "sample_adaptive_active_chain", _fake_adaptive)
    monkeypatch.setattr(sampling_mod, "sample_active_chain", _fake_active)

    frozen_sentinel = object()

    class _FakeModel:
        def __init__(self) -> None:
            self.adaptive = _FakeAdaptiveHook(subchain_length=2, converged=True)

        def freeze(self) -> object:
            return frozen_sentinel

    model = _FakeModel()

    def posterior_factory(m: object) -> tuple[str, object]:
        return ("posterior_for", m)

    result = sampling_mod.sample_adaptive_then_frozen_chain(
        model=model,
        posterior_factory=posterior_factory,
        proposal=object(),
        n_coarse_evals=12,
        initial_parameters=np.array([0.0, 0.0]),
        chain_key="chain_coarse_0",
        config=ChunkedMCMCConfig(chain_key="chain_coarse_0", chunk_size=8),
    )

    # Adaptive phase gets the full budget (no cap given) and is wired to stop on convergence.
    assert captured["adaptive_kwargs"]["n_coarse_evals"] == 12
    stop_check = captured["adaptive_kwargs"]["stop_check"]
    assert stop_check.__func__ is model.adaptive.has_converged.__func__
    assert stop_check.__self__ is model.adaptive
    assert captured["adaptive_kwargs"]["posterior"] == ("posterior_for", model)

    # Production phase runs on the frozen model for the remaining budget (12-8=4 -> //2=2).
    assert captured["active_kwargs"]["model"] is frozen_sentinel
    assert captured["active_kwargs"]["posterior"] == ("posterior_for", frozen_sentinel)
    assert captured["active_kwargs"]["subsampling_rate"] == 2
    assert captured["active_kwargs"]["iterations"] == 2
    np.testing.assert_allclose(captured["active_kwargs"]["initial_parameters"], adapt_samples[-1])

    # Chain and metadata are stitched adapt-then-production.
    np.testing.assert_array_equal(result.chain.samples, np.vstack([adapt_samples, prod_samples]))
    assert result.chain.extras.used_hf is not None
    np.testing.assert_array_equal(
        result.chain.extras.used_hf, np.concatenate([adapt_used_hf, prod_used_hf])
    )
    assert result.chain.extras.subchain_length is not None
    np.testing.assert_array_equal(
        result.chain.extras.subchain_length,
        np.concatenate([adapt_subchain, np.full(2, 2, dtype=int)]),
    )

    assert result.metadata["phase"] == "adapt_then_production"
    assert result.metadata["converged"] is True
    assert result.metadata["n_adapt_samples"] == 4
    assert result.metadata["n_production_samples"] == 2
    assert result.metadata["frozen_subsampling_rate"] == 2
    assert result.metadata["adapt_metadata"] is adapt_metadata
    assert result.metadata["production_metadata"] is prod_metadata


def test_sample_adaptive_then_frozen_chain_adapt_only_when_budget_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gp_active_mcmc.inference.sampling as sampling_mod

    adapt_samples = np.array([[0.0, 0.0], [1.0, 0.0]])
    adapt_chain = MCMCChain.from_arrays(samples=adapt_samples, used_hf=np.array([False, True]))
    adapt_metadata = {"coarse_evals_used": 8, "stopped_early": False}

    def _fake_adaptive(**kwargs: Any) -> SamplingResult:
        return SamplingResult(chain=adapt_chain, metadata=adapt_metadata)

    def _fail_active(**kwargs: Any) -> SamplingResult:
        raise AssertionError("production phase should not run when budget is exhausted")

    monkeypatch.setattr(sampling_mod, "sample_adaptive_active_chain", _fake_adaptive)
    monkeypatch.setattr(sampling_mod, "sample_active_chain", _fail_active)

    class _FakeModel:
        def __init__(self) -> None:
            self.adaptive = _FakeAdaptiveHook(subchain_length=2, converged=False)

        def freeze(self) -> object:
            return object()

    result = sampling_mod.sample_adaptive_then_frozen_chain(
        model=_FakeModel(),
        posterior_factory=lambda m: [],
        proposal=object(),
        n_coarse_evals=8,  # fully consumed by the adaptive phase
        initial_parameters=np.array([0.0, 0.0]),
        chain_key="chain_coarse_0",
        config=ChunkedMCMCConfig(chain_key="chain_coarse_0", chunk_size=8),
    )

    assert result.metadata["phase"] == "adapt_only"
    assert result.metadata["n_production_samples"] == 0
    assert result.chain is adapt_chain


def test_sample_adaptive_then_frozen_chain_requires_convergence_hook() -> None:
    import gp_active_mcmc.inference.sampling as sampling_mod

    class _NoAdaptiveModel:
        adaptive = None

    with pytest.raises(ValueError, match="has_converged"):
        sampling_mod.sample_adaptive_then_frozen_chain(
            model=_NoAdaptiveModel(),
            posterior_factory=lambda m: [],
            proposal=object(),
            n_coarse_evals=10,
            initial_parameters=np.array([0.0, 0.0]),
            chain_key="chain_coarse_0",
            config=ChunkedMCMCConfig(chain_key="chain_coarse_0", chunk_size=8),
        )
