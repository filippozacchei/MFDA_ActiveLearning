"""Shared MCMC proposal construction."""

from __future__ import annotations

from gp_active_mcmc.inference import AdaptiveMetropolisShared
from gp_active_mcmc.verification.problem import Problem

__all__ = ["make_proposal"]


def make_proposal(problem: Problem) -> AdaptiveMetropolisShared:
    """Adaptive random-walk Metropolis (Haario et al. 2001), shared by every method.    """
    return AdaptiveMetropolisShared(
        C0=problem.scale * problem.prior.cov,
        period=100,
        share_across_deepcopy=True,
        adaptive=True,
    )
