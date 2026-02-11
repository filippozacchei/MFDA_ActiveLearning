from __future__ import annotations

import numpy as np
import pytest

from gp_active_mcmc.inference.chain import MCMCChain


def test_chain_validation_and_slicing() -> None:
    samples = np.arange(20, dtype=float).reshape(10, 2)
    used_hf = np.array([0, 1] * 5, dtype=bool)
    sub = np.arange(10, dtype=int)

    chain = MCMCChain.from_arrays(samples=samples, used_hf=used_hf, subchain_length=sub)
    assert chain.n_steps == 10
    assert chain.n_dim == 2

    b = chain.burn_in(3)
    assert b.samples.shape == (7, 2)
    assert b.extras.used_hf is not None
    assert b.extras.subchain_length is not None
    assert b.extras.used_hf.shape == (7,)
    assert b.extras.subchain_length.shape == (7,)
    assert b.extras.subchain_length[0] == 3

    t = chain.thin(2)
    assert t.samples.shape[0] == 5
    assert t.extras.subchain_length is not None
    assert t.extras.subchain_length[1] == 2

    with pytest.raises(ValueError):
        _ = chain.thin(0)


def test_summary_keys() -> None:
    rng = np.random.default_rng(0)
    samples = rng.normal(size=(30, 2))
    used_hf = rng.random(30) < 0.2

    chain = MCMCChain.from_arrays(samples=samples, used_hf=used_hf)
    s = chain.summary()
    assert "n_steps" in s and "n_dim" in s
    assert "hf_call_fraction" in s
    assert "move_fraction" in s  # no accepted flags provided
