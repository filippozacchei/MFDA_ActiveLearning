from __future__ import annotations

import numpy as np
import pytest

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
        chain_key = kwargs.get("store_coarse_chain") and "chain_0" or "chain_0"
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
