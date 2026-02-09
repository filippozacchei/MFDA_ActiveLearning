# src/your_pkg/cfd/bfs_mesh.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import gmsh
from mpi4py import MPI
from dolfinx.io import gmshio

from .types import BFSGeometry, MeshOptions, BoundaryMarkers


def build_bfs_mesh(
    *,
    geom: BFSGeometry,
    mesh_opts: MeshOptions = MeshOptions(),
    markers: BoundaryMarkers = BoundaryMarkers(),
    comm: MPI.Comm = MPI.COMM_WORLD,
    model_rank: int = 0,
):
    """
    Build a 2D backward-facing step mesh and facet tags via Gmsh.

    Returns
    -------
    mesh : dolfinx.mesh.Mesh
    facet_tags : dolfinx.mesh.MeshTags
    L : float
        Total length L_up + L_down
    H : float
        Downstream height h2
    """
    if gmsh.is_initialized():
        gmsh.finalize()

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("General.Verbosity", 0)

    gdim = mesh_opts.gdim
    L = geom.L_up + geom.L_down

    if comm.rank == model_rank:
        up = gmsh.model.occ.addRectangle(0.0, 0.0, 0, geom.L_up, geom.h1)
        down = gmsh.model.occ.addRectangle(geom.L_up, 0.0, 0, geom.L_down, geom.h2)
        gmsh.model.occ.fuse([(2, up)], [(2, down)])
        gmsh.model.occ.synchronize()

        surfs = [s[1] for s in gmsh.model.occ.getEntities(dim=2)]
        gmsh.model.addPhysicalGroup(2, surfs, markers.fluid)
        gmsh.model.setPhysicalName(2, markers.fluid, "Fluid")

        bnds = gmsh.model.getBoundary([(2, s) for s in surfs], oriented=False)
        inflow, outflow, walls = [], [], []
        for dim, tag in bnds:
            cx, cy, _ = gmsh.model.occ.getCenterOfMass(dim, tag)
            if np.isclose(cx, 0.0, atol=1e-10):
                inflow.append(tag)
            elif np.isclose(cx, L, atol=1e-10):
                outflow.append(tag)
            else:
                walls.append(tag)

        gmsh.model.addPhysicalGroup(1, inflow, markers.inlet)
        gmsh.model.addPhysicalGroup(1, outflow, markers.outlet)
        gmsh.model.addPhysicalGroup(1, walls, markers.wall)

        gmsh.option.setNumber("Mesh.Algorithm", mesh_opts.algorithm)
        gmsh.option.setNumber("Mesh.RecombineAll", 1 if mesh_opts.recombine else 0)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_opts.lc_min)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_opts.lc_max)
        gmsh.model.mesh.generate(gdim)
        gmsh.model.mesh.setOrder(mesh_opts.order)
        if mesh_opts.optimize is not None:
            gmsh.model.mesh.optimize(mesh_opts.optimize)

    mesh, cell_tags, facet_tags = gmshio.model_to_mesh(
        gmsh.model, comm, model_rank, gdim
    )

    if comm.rank == model_rank:
        gmsh.finalize()

    return mesh, facet_tags, float(L), float(geom.h2)
