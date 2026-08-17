from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from gp_active_mcmc.verification.sampling import make_proposal


@dataclass
class _DummyProblem:
    prior: Any


class _DummyPrior:
    def __init__(self, cov: np.ndarray) -> None:
        self.cov = cov


def test_make_proposal_covariance_scales_with_prior_cov() -> None:
    cov = np.array([[4.0, 0.0], [0.0, 9.0]])
    problem = _DummyProblem(prior=_DummyPrior(cov))
    proposal = make_proposal(problem, scale=0.5)
    np.testing.assert_allclose(proposal.C, 0.5 * cov)


def test_make_proposal_default_scale() -> None:
    cov = np.eye(3)
    problem = _DummyProblem(prior=_DummyPrior(cov))
    proposal = make_proposal(problem)
    np.testing.assert_allclose(proposal.C, 0.05 * cov)
