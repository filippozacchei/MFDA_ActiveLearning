from __future__ import annotations

import copy
from typing import Any

import tinyDA as tda


class AdaptiveMetropolisShared(tda.AdaptiveMetropolis):
    """Adaptive Metropolis proposal with configurable deepcopy semantics.

    Motivation
    ----------
    Some MCMC workflows (e.g., multiple chains or chunked sampling) may deepcopy
    a proposal object. Depending on the application, you may want:

    - shared proposal state across deepcopies (default): adaptation history and
      covariance updates are shared, so all copies behave as one evolving proposal.
    - independent proposal state: deepcopies produce a separate object with a
      deep-copied internal state.

    Parameters
    ----------
    share_across_deepcopy
        If True, ``copy.deepcopy(proposal)`` returns the same instance.
        If False, ``deepcopy`` produces an independent clone.
    """
    def __init__(self, *args: Any, share_across_deepcopy: bool = True, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._share_across_deepcopy = bool(share_across_deepcopy)

    def __deepcopy__(self, memo: dict[int, object]) -> "AdaptiveMetropolisShared":
        if self._share_across_deepcopy:
            memo[id(self)] = self
            return self

        # Deep-copy attributes
        cls = self.__class__
        new_obj = cls.__new__(cls)  # type: ignore[misc]
        memo[id(self)] = new_obj
        for k, v in self.__dict__.items():
            setattr(new_obj, k, copy.deepcopy(v, memo))
        return new_obj
