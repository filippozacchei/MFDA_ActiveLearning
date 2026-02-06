from __future__ import annotations

import copy
from dataclasses import dataclass

import tinyDA as tda


class AdaptiveMetropolisShared(tda.AdaptiveMetropolis):
    def __init__(self, *args, share_across_deepcopy: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self._share_across_deepcopy = share_across_deepcopy

    def __deepcopy__(self, memo):
        if self._share_across_deepcopy:
            memo[id(self)] = self
            return self
        return super().__deepcopy__(memo)  # only if base implements it
