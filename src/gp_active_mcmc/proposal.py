from __future__ import annotations

import copy
from dataclasses import dataclass

import tinyDA as tda


@dataclass
class _AMSharedState:
    """Adaptive Metropolis state persisted across deepcopies."""

    scaling: float
    t: int
    k: int
    am_recursor: object | None = None


class AdaptiveMetropolisShared(tda.AdaptiveMetropolis):
    """
    AdaptiveMetropolis with shared adaptive state across deepcopies.

    This is designed for chunked sampling when tinyDA deepcopies the proposal
    internally. We share the adaptive state that must persist across chunk calls:
    - scaling
    - t
    - k
    - AM_recursor

    Implementation detail
    ---------------------
    We use guarded properties: during the base-class `__init__`, tinyDA sets
    attributes like `self.scaling = 1`. At that moment `_shared` does not exist,
    so the setters fall back to normal attribute assignment into `__dict__`.
    After `_shared` is created, reads/writes go through the shared state.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Read what tinyDA initialized (these should exist after super().__init__)
        scaling = float(self.__dict__.get("scaling", getattr(self, "scaling")))
        t = int(self.__dict__.get("t", getattr(self, "t")))
        k = int(self.__dict__.get("k", getattr(self, "k")))
        rec = self.__dict__.get("AM_recursor", None)

        self._shared = _AMSharedState(scaling=scaling, t=t, k=k, am_recursor=rec)

        # Optional: delete local shadow attributes to avoid confusion
        # (properties will serve subsequent access)
        for name in ("scaling", "t", "k", "AM_recursor"):
            if name in self.__dict__:
                del self.__dict__[name]

    # --- scaling -----------------------------------------------------------
    @property
    def scaling(self) -> float:
        if "_shared" not in self.__dict__:
            return float(self.__dict__.get("scaling", 1.0))
        return self._shared.scaling

    @scaling.setter
    def scaling(self, value: float) -> None:
        if "_shared" not in self.__dict__:
            self.__dict__["scaling"] = float(value)
            return
        self._shared.scaling = float(value)

    # --- t ----------------------------------------------------------------
    @property
    def t(self) -> int:
        if "_shared" not in self.__dict__:
            return int(self.__dict__.get("t", 0))
        return self._shared.t

    @t.setter
    def t(self, value: int) -> None:
        if "_shared" not in self.__dict__:
            self.__dict__["t"] = int(value)
            return
        self._shared.t = int(value)

    # --- k ----------------------------------------------------------------
    @property
    def k(self) -> int:
        if "_shared" not in self.__dict__:
            return int(self.__dict__.get("k", 0))
        return self._shared.k

    @k.setter
    def k(self, value: int) -> None:
        if "_shared" not in self.__dict__:
            self.__dict__["k"] = int(value)
            return
        self._shared.k = int(value)

    # --- AM_recursor -------------------------------------------------------
    @property
    def AM_recursor(self):
        if "_shared" not in self.__dict__:
            return self.__dict__.get("AM_recursor", None)
        return self._shared.am_recursor

    @AM_recursor.setter
    def AM_recursor(self, value) -> None:
        if "_shared" not in self.__dict__:
            self.__dict__["AM_recursor"] = value
            return
        self._shared.am_recursor = value

    # --- deepcopy ----------------------------------------------------------
    def __deepcopy__(self, memo):
        clone = type(self).__new__(type(self))
        memo[id(self)] = clone

        # Share adaptive state
        clone._shared = self._shared

        # Deepcopy everything else
        for key, val in self.__dict__.items():
            if key == "_shared":
                continue
            clone.__dict__[key] = copy.deepcopy(val, memo)

        return clone
