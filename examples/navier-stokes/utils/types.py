# src/your_pkg/cfd/types.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


BoundaryName = Literal["fluid", "inlet", "outlet", "wall"]


@dataclass(frozen=True)
class BFSGeometry:
    h1: float = 0.10
    h2: float = 0.20
    L_up: float = 0.10
    L_down: float = 0.40


@dataclass(frozen=True)
class MeshOptions:
    lc_min: float = 1e-3
    lc_max: float = 1e-2
    gdim: int = 2
    order: int = 1
    recombine: bool = False
    algorithm: int = 8
    optimize: str | None = "Netgen"


@dataclass(frozen=True)
class BoundaryMarkers:
    fluid: int = 1
    inlet: int = 2
    outlet: int = 3
    wall: int = 4


@dataclass(frozen=True)
class OutletProfile:
    y: np.ndarray
    u_x: np.ndarray
