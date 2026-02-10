from __future__ import annotations

import copy
from typing import Any

import tinyDA as tda


class AdaptiveMetropolisShared(tda.AdaptiveMetropolis):
    """Adaptive Metropolis proposal with controlled deep-copy semantics.

    This proposal extends `tinyDA.AdaptiveMetropolis` by
    defining what happens when the object is deep-copied.

    Why this matters
    ----------------
    Some sampling workflows deep-copy the proposal (explicitly or implicitly), for example:

    - **chunked sampling**, where the sampler is re-entered multiple times, and
    - certain multi-chain patterns.

    For adaptive proposals, deep-copying changes the algorithmic behaviour:

    - **Shared state**: adaptation continues across chunks as if there were a single proposal.
      This is typically what you want for chunked active sampling.
    - **Independent state**: each deepcopy adapts independently, which is appropriate only
      when you truly want independent proposals (e.g., fully independent chains).

    In this library, shared state is often the default because
    [`sample_adaptive_active_chain`][gp_active_mcmc.inference.sampling.sample_adaptive_active_chain]
    runs `tinyDA` repeatedly in chunks and we want a single evolving proposal.

    Parameters
    ----------
    *args, **kwargs
        Passed through to `tinyDA.AdaptiveMetropolis`.
    share_across_deepcopy
        Controls deepcopy behaviour:

        - True (default): ``copy.deepcopy(proposal)`` returns the same instance
          (proposal state is shared).
        - False: deepcopies produce independent clones with a deep-copied internal state.

    Notes
    -----
    Returning `self` from `__deepcopy__` is a deliberate deviation from standard Python
    semantics. It preserves adaptation history in workflows where deepcopies are an
    implementation detail rather than a user intent.

    See Also
    --------
    [`sample_adaptive_active_chain`][gp_active_mcmc.inference.sampling.sample_adaptive_active_chain]
        Chunked sampler where shared proposal state is typically desired.
    [`ChunkedMCMCConfig`][gp_active_mcmc.inference.sampling.ChunkedMCMCConfig]
        Configuration controlling chunk size for adaptive sampling.

    Examples
    --------
    Shared adaptation across chunks:

    >>> prop = AdaptiveMetropolisShared(C0=C0, period=100, share_across_deepcopy=True)

    Independent adaptation (useful for truly independent chains):

    >>> prop = AdaptiveMetropolisShared(C0=C0, period=100, share_across_deepcopy=False)
    """

    def __init__(
        self,
        *args: Any,
        share_across_deepcopy: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._share_across_deepcopy = bool(share_across_deepcopy)

    def __deepcopy__(self, memo: dict[int, object]) -> "AdaptiveMetropolisShared":
        """Deep-copy the proposal according to `share_across_deepcopy`.

        Parameters
        ----------
        memo
            Standard deepcopy memo dictionary used to preserve object identity and
            avoid infinite recursion.

        Returns
        -------
        proposal
            If `share_across_deepcopy=True`, returns `self` (shared state).
            Otherwise returns a deep-copied independent clone.
        """
        if self._share_across_deepcopy:
            memo[id(self)] = self
            return self

        cls = self.__class__
        new_obj = cls.__new__(cls)
        memo[id(self)] = new_obj
        for k, v in self.__dict__.items():
            setattr(new_obj, k, copy.deepcopy(v, memo))
        return new_obj
