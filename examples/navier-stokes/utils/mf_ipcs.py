# src/your_pkg/cfd/mf_ipcs.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tqdm.autonotebook
from mpi4py import MPI
from petsc4py import PETSc
from basix.ufl import element
from dolfinx.fem import (
    Constant,
    Function,
    dirichletbc,
    form,
    functionspace,
    locate_dofs_topological,
)
from dolfinx.fem.petsc import (
    assemble_matrix,
    assemble_vector,
    apply_lifting,
    create_vector,
    set_bc,
)
from ufl import (
    TrialFunction,
    TestFunction,
    dot,
    grad,
    div,
    nabla_grad,
    dx,
    lhs,
    rhs,
    inner,
)

from .bfs_mesh import build_bfs_mesh
from .outlet import sample_outlet_u_x
from .types import BFSGeometry, MeshOptions, BoundaryMarkers, OutletProfile


@dataclass(frozen=True)
class MFTimeConfig:
    dt: float = 1e-3
    t_end: float = 2.0
    progress: bool = True


@dataclass(frozen=True)
class MFFluidConfig:
    rho: float = 1.0
    nu: float = 1e-3  # kinematic viscosity


def solve_ipcs_bfs(
    *,
    geom: BFSGeometry,
    U_in: float,
    time: MFTimeConfig = MFTimeConfig(),
    fluid: MFFluidConfig = MFFluidConfig(),
    mesh_opts: MeshOptions = MeshOptions(),
    markers: BoundaryMarkers = BoundaryMarkers(),
    outlet_ny: int = 100,
    comm: MPI.Comm = MPI.COMM_WORLD,
) -> OutletProfile:
    """
    Medium-fidelity forward model: incompressible Navier–Stokes on a backward-facing-step
    using an IPCS-like splitting.

    Parameters
    ----------
    geom
        BFS geometry. If you want UQ in downstream height, vary geom.h2.
    U_in
        Inlet streamwise velocity (uniform profile).
    time
        Time stepping configuration.
    fluid
        Density and viscosity.
    mesh_opts, markers
        Mesh and boundary tagging configuration.
    outlet_ny
        Number of sampling points for the outlet velocity profile.

    Returns
    -------
    OutletProfile
        y coordinates and u_x(y) on the outlet.
    """
    mesh, ft, L, H = build_bfs_mesh(
        geom=geom, mesh_opts=mesh_opts, markers=markers, comm=comm
    )
    fdim = mesh.topology.dim - 1

    # --- Function spaces (Taylor–Hood P2/P1) ---------------------------------
    v_cg2 = element("Lagrange", mesh.basix_cell(), 2, shape=(mesh.geometry.dim,))
    s_cg1 = element("Lagrange", mesh.basix_cell(), 1)
    V = functionspace(mesh, v_cg2)  # velocity
    Q = functionspace(mesh, s_cg1)  # pressure

    # --- Constants -----------------------------------------------------------
    dt = float(time.dt)
    num_steps = int(np.round(time.t_end / dt))
    k = Constant(mesh, PETSc.ScalarType(dt))
    rho = Constant(mesh, PETSc.ScalarType(fluid.rho))
    nu = Constant(mesh, PETSc.ScalarType(fluid.nu))
    mu = rho * nu

    # --- Boundary conditions -------------------------------------------------
    class InletVelocity:
        def __call__(self, x):
            vals = np.zeros((2, x.shape[1]), dtype=PETSc.ScalarType)
            vals[0] = PETSc.ScalarType(U_in)
            return vals

    u_inlet = Function(V)
    u_inlet.interpolate(InletVelocity())

    inlet_dofs = locate_dofs_topological(V, fdim, ft.find(markers.inlet))
    wall_dofs = locate_dofs_topological(V, fdim, ft.find(markers.wall))
    outlet_p_dofs = locate_dofs_topological(Q, fdim, ft.find(markers.outlet))

    bcu_in = dirichletbc(u_inlet, inlet_dofs)
    bcu_w = dirichletbc(
        np.array((0.0, 0.0), dtype=PETSc.ScalarType),
        wall_dofs,
        V,
    )
    bcu = [bcu_in, bcu_w]

    bcp_out = dirichletbc(PETSc.ScalarType(0.0), outlet_p_dofs, Q)
    bcp = [bcp_out]

    # --- Unknowns / history --------------------------------------------------
    u = TrialFunction(V)
    v = TestFunction(V)
    p = TrialFunction(Q)
    q = TestFunction(Q)

    u_ = Function(V, name="u")  # corrected velocity
    u_s = Function(V, name="u_tentative")  # tentative
    u_n = Function(V, name="u_n")  # previous
    u_nm1 = Function(V, name="u_nm1")  # previous-previous

    p_ = Function(Q, name="p")  # pressure
    phi = Function(Q, name="phi")  # pressure increment

    # --- Step 1: tentative velocity (explicit convection, implicit diffusion) -
    # AB2 advecting velocity for explicit convection
    u_adv = 1.5 * u_n - 0.5 * u_nm1

    F1_lhs = rho / k * dot(u, v) * dx + inner(mu * grad(u), grad(v)) * dx
    a1 = form(lhs(F1_lhs))
    A1 = assemble_matrix(a1, bcs=bcu)
    A1.assemble()

    F1_rhs = (
        rho / k * dot(u_n, v) * dx
        - rho * inner(dot(u_adv, nabla_grad(u_n)), v) * dx
        + dot(p_, div(v)) * dx
    )
    L1 = form(F1_rhs)
    b1 = create_vector(L1)

    # --- Step 2: pressure Poisson -------------------------------------------
    a2 = form(dot(grad(p), grad(q)) * dx)
    L2 = form(-rho / k * dot(div(u_s), q) * dx)
    A2 = assemble_matrix(a2, bcs=bcp)
    A2.assemble()
    b2 = create_vector(L2)

    # --- Step 3: velocity correction ----------------------------------------
    a3 = form(rho * dot(u, v) * dx)
    L3 = form(rho * dot(u_s, v) * dx - k * dot(nabla_grad(phi), v) * dx)
    A3 = assemble_matrix(a3)
    A3.assemble()
    b3 = create_vector(L3)

    # --- Linear solvers ------------------------------------------------------
    solver1 = PETSc.KSP().create(mesh.comm)
    solver1.setOperators(A1)
    solver1.setType(PETSc.KSP.Type.BCGS)
    solver1.getPC().setType(PETSc.PC.Type.JACOBI)

    solver2 = PETSc.KSP().create(mesh.comm)
    solver2.setOperators(A2)
    solver2.setType(PETSc.KSP.Type.MINRES)
    pc2 = solver2.getPC()
    pc2.setType(PETSc.PC.Type.HYPRE)
    try:
        pc2.setHYPREType("boomeramg")
    except Exception:
        pc2.setType(PETSc.PC.Type.JACOBI)

    solver3 = PETSc.KSP().create(mesh.comm)
    solver3.setOperators(A3)
    solver3.setType(PETSc.KSP.Type.CG)
    solver3.getPC().setType(PETSc.PC.Type.SOR)

    # --- Time loop -----------------------------------------------------------
    progress = None
    if time.progress and mesh.comm.rank == 0:
        progress = tqdm.autonotebook.tqdm(total=num_steps, desc="MF IPCS BFS")

    for _ in range(num_steps):
        if progress is not None:
            progress.update(1)

        # Step 1: tentative velocity
        with b1.localForm() as loc:
            loc.set(0.0)
        assemble_vector(b1, L1)
        apply_lifting(b1, [a1], [bcu])
        b1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b1, bcu)
        solver1.solve(b1, u_s.x.petsc_vec)
        u_s.x.scatter_forward()

        # Step 2: pressure increment
        with b2.localForm() as loc:
            loc.set(0.0)
        assemble_vector(b2, L2)
        apply_lifting(b2, [a2], [bcp])
        b2.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b2, bcp)
        solver2.solve(b2, phi.x.petsc_vec)
        phi.x.scatter_forward()

        # Update pressure
        p_.x.petsc_vec.axpy(1.0, phi.x.petsc_vec)
        p_.x.scatter_forward()

        # Step 3: corrected velocity
        with b3.localForm() as loc:
            loc.set(0.0)
        assemble_vector(b3, L3)
        b3.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        solver3.solve(b3, u_.x.petsc_vec)
        u_.x.scatter_forward()

        # Rotate velocity history: u_nm1 <- u_n <- u_
        with (
            u_.x.petsc_vec.localForm() as lu,
            u_n.x.petsc_vec.localForm() as lun,
            u_nm1.x.petsc_vec.localForm() as lunm1,
        ):
            lun.copy(lunm1)
            lu.copy(lun)

    if progress is not None:
        progress.close()

    return sample_outlet_u_x(mesh=mesh, u_function=u_, L=L, H=H, ny=outlet_ny)


def forward_model(
    h1: float,
    *,
    U_in: float = 1.5,
    h2: float = 0.2,
    L_up: float = 0.10,
    L_down: float = 0.40,
) -> tuple[np.ndarray, np.ndarray]:
    """
    UQ-friendly wrapper: vary downstream height h2, return (y, u_x).
    """
    prof = solve_ipcs_bfs(
        geom=BFSGeometry(h1=h1, h2=h2, L_up=L_up, L_down=L_down), U_in=U_in
    )
    return prof.y, prof.u_x
