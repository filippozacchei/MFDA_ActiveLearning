# examples/navier_stokes/utils/bfs_mesh.py
from __future__ import annotations

from dataclasses import dataclass

import gmsh
import numpy as np
from dolfinx.io import gmshio
from mpi4py import MPI

from .ns_types import BFSGeometry, BoundaryMarkers, MeshOptions

"""Mesh generation for the backward-facing step (BFS) benchmark.

This module provides a small wrapper around Gmsh + dolfinx to generate a
2D BFS geometry and export it as a dolfinx mesh with facet tags.

Design goals
------------
- Keep *all* meshing logic in one place.
- Return facet tags suitable for applying boundary conditions in dolfinx.
- Avoid side effects: Gmsh is initialised/finalised inside the function.

Notes
-----
The geometry is built as the union of two rectangles:

- upstream rectangle:  [0, L_up] x [0, h1]
- downstream rectangle: [L_up, L_up + L_down] x [0, h2]

The step is located at x = L_up.

Boundary classification is performed by checking the x-coordinate of the facet
center-of-mass:
- x ≈ 0         -> inlet
- x ≈ L_total   -> outlet
- otherwise     -> wall
"""


@dataclass(frozen=True, slots=True)
class BFSMesh:
    """Bundle returned by :func:`build_bfs_mesh`.

    Attributes
    ----------
    mesh
        dolfinx mesh of the BFS domain.
    facet_tags
        Facet markers (MeshTags) labelling boundaries according to `BoundaryMarkers`.
    L
        Total length, `L_up + L_down`.
    H
        Downstream height, `h2`.
    """

    mesh: object
    facet_tags: object
    L: float
    H: float


def _gmsh_reset() -> None:
    """Ensure Gmsh starts from a clean state."""
    if gmsh.is_initialized():
        gmsh.finalize()
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("bfs")


def build_bfs_mesh(
    *,
    geom: BFSGeometry,
    mesh_opts: MeshOptions | None = None,
    markers: BoundaryMarkers | None = None,
    comm: MPI.Comm = MPI.COMM_WORLD,
    model_rank: int = 0,
    atol: float = 1e-10,
) -> BFSMesh:
    """Build a 2D backward-facing step mesh with facet tags via Gmsh.

    Parameters
    ----------
    geom
        BFS geometry parameters.
    mesh_opts
        Meshing options (characteristic lengths, algorithm, element order, ...).
    markers
        Integer markers for domain and boundaries.
    comm
        MPI communicator used by dolfinx.
    model_rank
        Rank responsible for constructing the Gmsh model (typically 0).
    atol
        Absolute tolerance used to classify inlet/outlet facets using the
        x-coordinate of facet centers-of-mass.

    Returns
    -------
    out
        :class:`BFSMesh` bundle: `(mesh, facet_tags, L, H)`.

    Raises
    ------
    ValueError
        If `mesh_opts.gdim` is not 2 for this mesh generator.
    """
    if mesh_opts is None:
        mesh_opts = MeshOptions()
    if markers is None:
        markers = BoundaryMarkers()
    if mesh_opts.gdim != 2:
        raise ValueError(f"build_bfs_mesh currently supports gdim=2 only. Got {mesh_opts.gdim}.")

    L_total = float(geom.L_up + geom.L_down)
    H = float(geom.h2)

    _gmsh_reset()

    if comm.rank == model_rank:
        # Geometry (CAD) definition
        up = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, geom.L_up, geom.h1)
        down = gmsh.model.occ.addRectangle(geom.L_up, 0.0, 0.0, geom.L_down, geom.h2)

        # Fuse (boolean union) and synchronise CAD kernel.
        gmsh.model.occ.fuse([(2, up)], [(2, down)], removeObject=True, removeTool=True)
        gmsh.model.occ.synchronize()

        # Collect surfaces after fuse and mark the fluid domain.
        surfs = [s[1] for s in gmsh.model.occ.getEntities(dim=2)]
        gmsh.model.addPhysicalGroup(2, surfs, markers.fluid)
        gmsh.model.setPhysicalName(2, markers.fluid, "fluid")

        # Identify boundary curves and classify inlet/outlet/walls.
        bnds = gmsh.model.getBoundary([(2, s) for s in surfs], oriented=False)

        inlet: list[int] = []
        outlet: list[int] = []
        wall: list[int] = []

        for dim, tag in bnds:
            cx, _cy, _cz = gmsh.model.occ.getCenterOfMass(dim, tag)
            if np.isclose(cx, 0.0, atol=atol):
                inlet.append(tag)
            elif np.isclose(cx, L_total, atol=atol):
                outlet.append(tag)
            else:
                wall.append(tag)

        if not inlet:
            raise RuntimeError("Failed to identify inlet boundary (no curves with x≈0).")
        if not outlet:
            raise RuntimeError("Failed to identify outlet boundary (no curves with x≈L_total).")
        if not wall:
            raise RuntimeError("Failed to identify wall boundary (no remaining boundary curves).")

        gmsh.model.addPhysicalGroup(1, inlet, markers.inlet)
        gmsh.model.setPhysicalName(1, markers.inlet, "inlet")

        gmsh.model.addPhysicalGroup(1, outlet, markers.outlet)
        gmsh.model.setPhysicalName(1, markers.outlet, "outlet")

        gmsh.model.addPhysicalGroup(1, wall, markers.wall)
        gmsh.model.setPhysicalName(1, markers.wall, "wall")

        # Mesh options
        gmsh.option.setNumber("Mesh.Algorithm", int(mesh_opts.algorithm))
        gmsh.option.setNumber("Mesh.RecombineAll", 1 if mesh_opts.recombine else 0)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", float(mesh_opts.lc_min))
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", float(mesh_opts.lc_max))

        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.setOrder(int(mesh_opts.order))

        if mesh_opts.optimize is not None:
            gmsh.model.mesh.optimize(str(mesh_opts.optimize))

    # Distribute mesh and tags to dolfinx
    mesh, _cell_tags, facet_tags = gmshio.model_to_mesh(gmsh.model, comm, model_rank, 2)

    # Clean shutdown: only the model-building rank finalizes.
    if comm.rank == model_rank:
        gmsh.finalize()

    return BFSMesh(mesh=mesh, facet_tags=facet_tags, L=L_total, H=H)
