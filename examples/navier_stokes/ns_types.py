# examples/navier_stokes/utils/types.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

"""Lightweight domain types for the Navier-Stokes backward-facing step example.

This module defines *pure* dataclasses used across the example code.

Design goals
------------
- No dependency on FEniCS/dolfinx: these types are safe to import anywhere.
- Clear semantics: geometry parameters, meshing options, and boundary labelling.
- Minimal validation: catch common configuration mistakes early.

Notes
-----
Coordinate convention (typical for BFS benchmarks):
- x: streamwise direction (inlet -> outlet)
- y: wall-normal direction

All lengths are assumed to be in consistent units (e.g., meters).
"""


BoundaryName = Literal["fluid", "inlet", "outlet", "wall"]


@dataclass(frozen=True, slots=True)
class BFSGeometry:
    """Backward-facing step (BFS) geometry parameters.

    Parameters
    ----------
    h1
        Upstream channel height (before the step).
    h2
        Downstream channel height (after the step). Must satisfy `h2 > h1`.
    L_up
        Upstream channel length (inlet to step).
    L_down
        Downstream channel length (step to outlet).

    Notes
    -----
    A common BFS definition is:
    - inlet located at x = -L_up
    - step located at x = 0
    - outlet located at x = L_down
    - bottom wall at y = 0
    - top wall at y = h1 upstream, y = h2 downstream
    """

    h1: float = 0.10
    h2: float = 0.20
    L_up: float = 0.10
    L_down: float = 0.40

    def __post_init__(self) -> None:
        if self.h1 <= 0 or self.h2 <= 0:
            raise ValueError("h1 and h2 must be positive.")
        if self.h2 <= self.h1:
            raise ValueError(f"Expected h2 > h1 for a step. Got h1={self.h1}, h2={self.h2}.")
        if self.L_up <= 0 or self.L_down <= 0:
            raise ValueError("L_up and L_down must be positive.")

    @property
    def step_height(self) -> float:
        """Step height `h2 - h1`."""
        return float(self.h2 - self.h1)

    @property
    def L_total(self) -> float:
        """Total streamwise length `L_up + L_down`."""
        return float(self.L_up + self.L_down)


@dataclass(frozen=True, slots=True)
class MeshOptions:
    """Meshing options for generating a BFS mesh.

    Parameters
    ----------
    lc_min, lc_max
        Target minimum/maximum characteristic length (mesh size).
    gdim
        Geometric dimension (2 for 2D, 3 for 3D).
    order
        Polynomial order for geometry representation (when supported).
    recombine
        Whether to recombine triangles into quads (when supported).
    algorithm
        Meshing algorithm identifier (backend-dependent; e.g., gmsh).
    optimize
        Optional mesh optimization strategy name (backend-dependent).

    Notes
    -----
    These options are intentionally backend-agnostic: a mesh generator wrapper
    can interpret them according to gmsh/dolfinx conventions.
    """

    lc_min: float = 2.5e-2 #1e-3
    lc_max: float = 2.5e-2 #1e-2
    gdim: int = 2
    order: int = 1
    recombine: bool = False
    algorithm: int = 8
    optimize: str | None = None

    def __post_init__(self) -> None:
        if self.lc_min <= 0 or self.lc_max <= 0:
            raise ValueError("lc_min and lc_max must be positive.")
        if self.lc_min > self.lc_max:
            raise ValueError(f"Expected lc_min <= lc_max. Got {self.lc_min} > {self.lc_max}.")
        if self.gdim not in (2, 3):
            raise ValueError(f"gdim must be 2 or 3. Got {self.gdim}.")
        if self.order <= 0:
            raise ValueError("order must be a positive integer.")
        if self.algorithm <= 0:
            raise ValueError("algorithm must be a positive integer.")


@dataclass(frozen=True, slots=True)
class BoundaryMarkers:
    """Integer markers for physical boundaries.

    These IDs are used to tag facets/boundaries in a mesh and later apply
    boundary conditions or extract quantities of interest.

    Attributes
    ----------
    fluid
        Marker for the fluid domain (cells).
    inlet
        Marker for the inlet boundary.
    outlet
        Marker for the outlet boundary.
    wall
        Marker for all walls (including step surfaces).
    """

    fluid: int = 1
    inlet: int = 2
    outlet: int = 3
    wall: int = 4

    def __post_init__(self) -> None:
        vals = (self.fluid, self.inlet, self.outlet, self.wall)
        if any(v <= 0 for v in vals):
            raise ValueError("All markers must be positive integers.")
        if len(set(vals)) != len(vals):
            raise ValueError(f"Markers must be distinct. Got {vals}.")

    def as_dict(self) -> dict[str, int]:
        """Return markers as a plain dict (useful for serialisation/logging)."""
        return {
            "fluid": self.fluid,
            "inlet": self.inlet,
            "outlet": self.outlet,
            "wall": self.wall,
        }


@dataclass(frozen=True, slots=True)
class OutletProfile:
    """Velocity profile sampled at the outlet.

    Parameters
    ----------
    y
        Wall-normal coordinates along the outlet line, shape ``(n,)``.
    u_x
        Streamwise velocity evaluated at the same points, shape ``(n,)``.

    Notes
    -----
    This container is meant for:
    - plotting outlet profiles,
    - comparing to reference solutions,
    - building likelihoods from outlet measurements.

    The coordinates `y` are expected to be 1D and strictly increasing. This makes
    interpolation and integration well-defined and avoids subtle ordering bugs.
    """

    y: np.ndarray
    u_x: np.ndarray

    def __post_init__(self) -> None:
        y = np.asarray(self.y, dtype=float).ravel()
        u = np.asarray(self.u_x, dtype=float).ravel()

        if y.ndim != 1 or u.ndim != 1:
            raise ValueError("y and u_x must be 1D arrays.")
        if y.shape != u.shape:
            raise ValueError(f"y and u_x must have the same shape. Got {y.shape} vs {u.shape}.")
        if y.size == 0:
            raise ValueError("OutletProfile arrays must be non-empty.")
        if not np.all(np.isfinite(y)) or not np.all(np.isfinite(u)):
            raise ValueError("y and u_x must be finite.")
        if np.any(np.diff(y) <= 0):
            raise ValueError("y must be strictly increasing.")

        object.__setattr__(self, "y", y)
        object.__setattr__(self, "u_x", u)

    @property
    def n(self) -> int:
        """Number of sampled points."""
        return int(self.y.size)

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Return `(y, u_x)` as plain NumPy arrays."""
        return self.y, self.u_x
