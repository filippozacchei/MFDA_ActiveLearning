"""Inference components (models, likelihoods, proposals, sampling, results)."""

from __future__ import annotations

from gp_active_mcmc.inference.chain import ChainExtras, MCMCChain, SamplingResult
from gp_active_mcmc.inference.coarse_output import CoarseOutput
from gp_active_mcmc.inference.likelihood import GaussianLogLikeWithGP
from gp_active_mcmc.inference.model import ActiveMCMCModel, EvaluationLog
from gp_active_mcmc.inference.proposal import AdaptiveMetropolisShared
from gp_active_mcmc.inference.sampling import (
    ChunkedMCMCConfig,
    sample_active_chain,
    sample_adaptive_active_chain,
)

__all__ = [
    "ChainExtras",
    "MCMCChain",
    "SamplingResult",
    "CoarseOutput",
    "GaussianLogLikeWithGP",
    "EvaluationLog",
    "ActiveMCMCModel",
    "AdaptiveMetropolisShared",
    "ChunkedMCMCConfig",
    "sample_active_chain",
    "sample_adaptive_active_chain",
]
