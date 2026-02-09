"""gp_active_mcmc: Active learning / multi-fidelity MCMC with GP surrogates.

Public API
----------
High-level entrypoints are exposed at package level so users can:

- build surrogates (`gp_active_mcmc.surrogates`)
- run sampling (`gp_active_mcmc.sample_active_chain`, `gp_active_mcmc.sample_adaptive_active_chain`)
- access results (`gp_active_mcmc.SamplingResult`, `gp_active_mcmc.MCMCChain`)
"""

from __future__ import annotations

from gp_active_mcmc.inference.chain import ChainExtras, MCMCChain, SamplingResult
from gp_active_mcmc.inference.sampling import (
    ChunkedMCMCConfig,
    sample_active_chain,
    sample_adaptive_active_chain,
)

__all__ = [
    "ChainExtras",
    "MCMCChain",
    "SamplingResult",
    "ChunkedMCMCConfig",
    "sample_active_chain",
    "sample_adaptive_active_chain",
]
