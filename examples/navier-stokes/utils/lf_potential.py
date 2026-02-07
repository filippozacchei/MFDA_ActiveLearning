# src/your_pkg/cfd/lf_potential.py
from __future__ import annotations

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from basix.ufl import element
from dolfinx.fem import (
    Function,
    functionspace,
    dirichletbc,
    form,
    locate_dofs_topological,
    Constant,
)
from dolfinx.fem.petsc import (
    assemble_matrix,
    assemble_vector,
    apply_lifting,
    create_vector,
    set_bc,
)
from ufl import TrialFunction, TestFunction, inner, grad, dx

from .bfs_mesh import build_bfs_mesh
from .outlet import sample_outlet_u_x
from .types import BFSGeometry, MeshOptions, BoundaryMarkers, OutletProfile


def solve_potential_bfs(
    *,
    geom: BFSGeometry,
    U_in: float,
    mesh_opts: MeshOptions = MeshOptions(),
    markers: BoundaryMarkers = BoundaryMarkers(),
    comm: MPI.Comm = MPI.COMM_WORLD,
) -> OutletProfile:
    mesh, ft, L, H = build_bfs_mesh(
        geom=geom, mesh_opts=mesh_opts, markers=markers, comm=comm
    )
    Vphi = functionspace(mesh, ("CG", 2))
    v_cg1 = element("Lagrange", mesh.basix_cell(), 1, shape=(mesh.geometry.dim,))
    Vvec = functionspace(mesh, v_cg1)
    fdim = mesh.topology.dim - 1
    inlet_marker, outlet_marker, wall_marker = 2, 3, 4

    # BCs for potential
    Phi_in, Phi_out = 0.0, U_in * L
    phi_inlet = Function(Vphi)
    phi_inlet.x.array[:] = Phi_in
    phi_outlet = Function(Vphi)
    phi_outlet.x.array[:] = Phi_out

    inlet_dofs = locate_dofs_topological(Vphi, fdim, ft.find(inlet_marker))
    outlet_dofs = locate_dofs_topological(Vphi, fdim, ft.find(outlet_marker))
    bcs = [dirichletbc(phi_inlet, inlet_dofs), dirichletbc(phi_outlet, outlet_dofs)]

    # Laplace problem
    phi = TrialFunction(Vphi)
    psi = TestFunction(Vphi)
    a_form = inner(grad(phi), grad(psi)) * dx
    rhs_form = Constant(mesh, PETSc.ScalarType(0.0)) * psi * dx

    A = assemble_matrix(form(a_form), bcs=bcs)
    A.assemble()
    b = create_vector(form(rhs_form))
    with b.localForm() as blf:
        blf.set(0.0)
    assemble_vector(b, form(rhs_form))
    apply_lifting(b, [form(a_form)], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, bcs)

    Phi = Function(Vphi, name="Phi")
    solver = PETSc.KSP().create(mesh.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    solver.getPC().setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=1e-10, atol=1e-12)
    solver.solve(b, Phi.x.petsc_vec)
    Phi.x.scatter_forward()

    # Project velocity = ∇Φ
    u = Function(Vvec, name="u")
    u_t, v_t = TrialFunction(Vvec), TestFunction(Vvec)
    Aproj = assemble_matrix(form(inner(u_t, v_t) * dx))
    Aproj.assemble()
    bproj = create_vector(form(inner(grad(Phi), v_t) * dx))
    with bproj.localForm() as blf:
        blf.set(0.0)
    assemble_vector(bproj, form(inner(grad(Phi), v_t) * dx))

    kspP = PETSc.KSP().create(mesh.comm)
    kspP.setOperators(Aproj)
    kspP.setType(PETSc.KSP.Type.CG)
    kspP.getPC().setType(PETSc.PC.Type.JACOBI)
    kspP.setTolerances(rtol=1e-12, atol=1e-14)
    kspP.solve(bproj, u.x.petsc_vec)
    u.x.scatter_forward()

    return sample_outlet_u_x(mesh=mesh, u_function=u, L=L, H=H, ny=100)


def forward_model(
    h1: float,
    *,
    U_in: float = 1.5,
    h2: float = 0.20,
    L_up: float = 0.10,
    L_down: float = 0.40,
) -> tuple[np.ndarray, np.ndarray]:
    """
    UQ-friendly wrapper: vary downstream height h2, return (y, u_x).
    """
    prof = solve_potential_bfs(
        geom=BFSGeometry(h1=h1, h2=h2, L_up=L_up, L_down=L_down), U_in=U_in
    )
    return prof.y, prof.u_x
