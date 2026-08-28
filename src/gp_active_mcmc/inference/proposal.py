from __future__ import annotations

import copy
from collections import deque
from typing import Any

import tinyDA as tda


class AdaptiveMetropolisShared(tda.AdaptiveMetropolis):  # type: ignore[misc]
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
    min_scaling
        Lower bound enforced on `self.scaling` after every `adapt()` call (see `adapt`'s
        docstring for the mechanism this protects against). `0.0` opts out, restoring
        plain `tinyDA.GaussianRandomWalk`/`AdaptiveMetropolis` behaviour, where `scaling`
        has no floor at all. Default `0.05` was chosen empirically, not just picked as
        "small": a floor of `1e-3` was tried first and, reproducing an actually-observed
        stuck chain, was *not* enough -- by the time `scaling` reaches that range the
        proposal step is already too small in absolute terms to cross whatever region
        made the surrogate posterior wrong there, so the chain stayed stuck even pinned
        at that floor. `0.05` (roughly the un-adapted `scaling=1` proposal shrunk by up
        to 20x, still comparable to this package's own default `C0` scale factor in
        `make_proposal`) was verified directly to let that same reproduced chain escape
        and re-converge, with `scaling` then self-correcting back upward on its own once
        real acceptances resumed. It costs some efficiency in the legitimate case where a
        posterior's true width needs `scaling` below `0.05` to hit the target acceptance
        rate (MCMC correctness is unaffected either way, just mixing efficiency) -- pass
        a smaller value, or `0.0`, if that tradeoff doesn't fit a particular use case.

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
        min_scaling: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._share_across_deepcopy = bool(share_across_deepcopy)
        self.min_scaling = float(min_scaling)

    def __deepcopy__(self, memo: dict[int, object]) -> AdaptiveMetropolisShared:
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

    def setup_proposal(self, **kwargs: Any) -> None:
        """Initialise adaptive state, but only on this object's first call.

        `tinyDA.AdaptiveMetropolis.setup_proposal` (what `super()` resolves to)
        unconditionally rebuilds `self.AM_recursor` from scratch every time it's
        called -- a fresh `RecursiveSampleMoments` seeded with just the current
        `parameters` and a zero covariance, discarding whatever shape the running
        covariance estimate had already learned. That's fine for a proposal used in one
        continuous `tinyDA.Chain`/`DAChain`, which calls `setup_proposal` exactly once
        at construction -- but this library's chunked round-based harness
        (`rounds.py`'s `_run_chunk`) constructs a *new* `Chain`/`DAChain` every round,
        which means `setup_proposal` -- and this reset -- fires every round too, on the
        very same (never-deep-copied) proposal object each time. Left unguarded, the
        Haario covariance-shape estimate would never accumulate past one round's worth
        of samples before being thrown away, no matter how many rounds actually ran --
        undermining the entire point of `share_across_deepcopy`'s "adaptation continues
        across chunks as if there were a single proposal" contract (`__deepcopy__`
        above only protects that contract against the *deepcopy* reset path; this is
        the other, more common one in practice, since a chunk's proposal object is
        typically reused directly, never deep-copied, across its own rounds).

        Guarding on `AM_recursor` already existing makes `setup_proposal` a no-op for
        every call after the first on a given object -- exactly the semantics
        `share_across_deepcopy=True` already describes, just extended to cover this
        second reset path. `self.scaling`/`self.t`/`self.k` (the global-scaling
        adaptation `GaussianRandomWalk.adapt` tracks) live on this object and were
        never touched by `setup_proposal` itself -- but see `adapt` below for a
        related reset this same round-reconstruction pattern causes on *its* input.
        """
        if hasattr(self, "AM_recursor"):
            return
        super().setup_proposal(**kwargs)

    def adapt(self, **kwargs: Any) -> None:
        """Adapt the proposal, substituting a proposal-owned rolling window of recent
        accept/reject outcomes for `kwargs["accepted"]` before delegating.

        `GaussianRandomWalk.adapt` (reached via `super()`) computes its
        acceptance-rate-driven scaling update from `kwargs["accepted"][-self.period:]`.
        `kwargs["accepted"]` is `tinyDA.Chain`/`DAChain`'s own `accepted`/
        `accepted_coarse` list -- and, like `AM_recursor` above, it starts fresh (`[]`)
        every time a `Chain`/`DAChain` is constructed, i.e. every round in this
        library's chunked harness. Whenever a period boundary falls early in a new
        round, that slice draws on far fewer than `self.period` real outcomes, feeding
        a noisier-than-intended acceptance-rate estimate into the scaling update --
        the same class of bug as `setup_proposal`'s, just hitting the scalar `scaling`
        adaptation instead of the covariance-shape one.

        Maintaining the window here instead -- `self._recent_accepted`, capped at
        `self.period` and appended to (never replaced) every call -- keeps it
        continuous across round boundaries the same way `setup_proposal` now keeps
        `AM_recursor` continuous. Only engages when `self.adaptive` (matching
        `GaussianRandomWalk.adapt`'s own gating -- `self.period` isn't even set
        otherwise) and when the caller actually passed `accepted` (both
        `Chain.sample`/`DAChain._sample_coarse` always do, appending the latest
        outcome immediately before calling `adapt`, so `accepted[-1]` is always the
        one new decision this call is reporting).

        After delegating, also enforces `min_scaling` as a floor on `self.scaling`.
        `GaussianRandomWalk.adapt`'s update is
        `scaling = exp(log(scaling) + gamma**-k * (acceptance_rate - alpha_star))`:
        every period of 0% acceptance multiplies `scaling` by `exp(-gamma**-k * alpha_star)`,
        strictly less than 1, with no opposite pull unless acceptance recovers above
        `alpha_star`. Diminishing adaptation (`gamma**-k -> 0` as `k -> inf`) bounds the
        *total* possible shrinkage from repeated bad luck, but only asymptotically: with
        this proposal's usual `gamma=1.01`, the bound is already reached to within a
        rounding error after a few hundred periods, and by then `scaling` can be many
        orders of magnitude below anything that generates a proposal meaningfully
        different from the current point -- confirmed directly (`gamma=1.01`, 100
        straight 0%-acceptance periods: `scaling` collapses from `1` to `~2.7e-7`,
        matching this formula's closed form to 2 significant figures). Once `scaling` is
        that small, the random-walk step is small enough that acceptance can stay
        pinned near 0% indefinitely too -- confirmed the surrogate's local posterior
        surface can be confidently *wrong* (not merely uncertain) over a region much
        larger than a `~1e-7`-scaled step even at a perfectly healthy, non-degenerate
        fit -- so nothing in the unmodified algorithm ever pulls `scaling` back up: a
        chain that hits one unlucky stretch is trapped for the rest of the run. The
        floor below breaks that trap by guaranteeing the chain can always still propose
        a materially different point, without changing how `scaling` adapts anywhere
        above `min_scaling`.
        """
        accepted = kwargs.get("accepted")
        if self.adaptive and accepted:
            if not hasattr(self, "_recent_accepted"):
                self._recent_accepted: deque[bool] = deque(maxlen=self.period)
            self._recent_accepted.append(bool(accepted[-1]))
            kwargs = dict(kwargs, accepted=list(self._recent_accepted))
        super().adapt(**kwargs)
        if self.adaptive and self.min_scaling > 0.0:
            self.scaling = max(self.scaling, self.min_scaling)  # type: ignore[has-type]
