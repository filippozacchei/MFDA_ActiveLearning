"""Inference components (models, likelihoods, proposals, sampling, results)."""

from __future__ import annotations

from gp_active_mcmc.inference.adaptive_subchain import (
    AdaptiveSubchain,
    AdaptiveSubchainControl,
    AdaptiveSubchainState,
)
from gp_active_mcmc.inference.chain import ChainExtras, MCMCChain, SamplingResult
from gp_active_mcmc.inference.coarse_output import CoarseOutput
from gp_active_mcmc.inference.likelihood import ActiveGPLogLike
from gp_active_mcmc.inference.model import ActiveMCMCModel, AdaptiveHook, EvaluationLog
from gp_active_mcmc.inference.proposal import AdaptiveMetropolisShared
from gp_active_mcmc.inference.sampling import (
    ChunkedMCMCConfig,
    sample_active_chain,
    sample_adaptive_active_chain,
    sample_adaptive_then_frozen_chain,
)

__all__ = [
    "ActiveGPLogLike",
    "ActiveMCMCModel",
    "AdaptiveHook",
    "AdaptiveMetropolisShared",
    "AdaptiveSubchain",
    "AdaptiveSubchainControl",
    "AdaptiveSubchainState",
    "ChainExtras",
    "ChunkedMCMCConfig",
    "CoarseOutput",
    "EvaluationLog",
    "MCMCChain",
    "SamplingResult",
    "sample_active_chain",
    "sample_adaptive_active_chain",
    "sample_adaptive_then_frozen_chain",
]
