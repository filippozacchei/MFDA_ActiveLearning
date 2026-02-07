# src/your_pkg/cfd/hf_boussinesq.py
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
    # assemble_scalar,
    dirichletbc,
    form,
    functionspace,
    locate_dofs_topological,
)
from dolfinx.fem.petsc import (
    assemble_matrix,
    assemble_vector,
    apply_lifting,
    create_matrix,
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
    exp,
)

from .bfs_mesh import build_bfs_mesh
from .outlet import sample_outlet_u_x
from .types import BFSGeometry, MeshOptions, BoundaryMarkers, OutletProfile


@dataclass(frozen=True)
class HFTimeConfig:
    dt: float = 1e-3
    t_end: float = 2.0
    progress: bool = True


@dataclass(frozen=True)
class HFThermoConfig:
    T_hot: float = 10.0
    T_cold: float = 0.0
    cp: float = 1.0
    k_th: float = 0.01
    alpha: float | None = None  # if None, alpha = k_th/(rho*cp)


@dataclass(frozen=True)
class HFFluidConfig:
    rho: float = 1.0
    nu0: float = 1e-3  # base viscosity scale
    a_vis: float = 0.5  # viscosity temperature sensitivity
    beta: float = 3e-3  # buoyancy coefficient
    g: tuple[float, float] = (0.0, -9.81)


def solve_boussinesq_bfs(
    *,
    geom: BFSGeometry,
    U_in: float,
    time: HFTimeConfig = HFTimeConfig(),
    thermo: HFThermoConfig = HFThermoConfig(),
    fluid: HFFluidConfig = HFFluidConfig(),
    mesh_opts: MeshOptions = MeshOptions(),
    markers: BoundaryMarkers = BoundaryMarkers(),
    outlet_ny: int = 100,
    comm: MPI.Comm = MPI.COMM_WORLD,
) -> OutletProfile:
    """
    High-fidelity forward model:
    - incompressible momentum + pressure splitting
    - temperature transport
    - buoyancy term (Boussinesq)
    - temperature-dependent viscosity nu(T) = nu0 * exp(-a_vis*(T - T_ref))

    Returns outlet u_x(y) after time integration.
    """
    mesh, ft, L, H = build_bfs_mesh(
        geom=geom, mesh_opts=mesh_opts, markers=markers, comm=comm
    )
    fdim = mesh.topology.dim - 1

    # --- Time/physical constants --------------------------------------------
    dt = float(time.dt)
    num_steps = int(np.round(time.t_end / dt))
    k = Constant(mesh, PETSc.ScalarType(dt))

    rho_v = float(fluid.rho)
    rho = Constant(mesh, PETSc.ScalarType(rho_v))
    beta = Constant(mesh, PETSc.ScalarType(fluid.beta))
    g_vec = Constant(mesh, PETSc.ScalarType(fluid.g))

    alpha_val = thermo.alpha
    if alpha_val is None:
        alpha_val = float(thermo.k_th / (rho_v * thermo.cp))
    alpha = Constant(mesh, PETSc.ScalarType(alpha_val))

    nu0 = Constant(mesh, PETSc.ScalarType(fluid.nu0))
    a_vis = Constant(mesh, PETSc.ScalarType(fluid.a_vis))
    T_ref = Constant(mesh, PETSc.ScalarType(0.5 * thermo.T_hot))

    # --- Spaces --------------------------------------------------------------
    v_cg2 = element("Lagrange", mesh.basix_cell(), 2, shape=(mesh.geometry.dim,))
    s_cg1 = element("Lagrange", mesh.basix_cell(), 1)
    V = functionspace(mesh, v_cg2)  # velocity
    Q = functionspace(mesh, s_cg1)  # pressure
    QT = Q  # temperature in CG1

    # --- BCs -----------------------------------------------------------------
    class InletVelocity:
        def __call__(self, x):
            vals = np.zeros((2, x.shape[1]), dtype=PETSc.ScalarType)
            vals[0] = PETSc.ScalarType(U_in)
            return vals

    u_inlet = Function(V)
    u_inlet.interpolate(InletVelocity())

    inlet_dofs_V = locate_dofs_topological(V, fdim, ft.find(markers.inlet))
    wall_dofs_V = locate_dofs_topological(V, fdim, ft.find(markers.wall))
    outlet_dofs_Q = locate_dofs_topological(Q, fdim, ft.find(markers.outlet))

    bcu_in = dirichletbc(u_inlet, inlet_dofs_V)
    bcu_w = dirichletbc(np.array((0.0, 0.0), dtype=PETSc.ScalarType), wall_dofs_V, V)
    bcu = [bcu_in, bcu_w]

    bcp_out = dirichletbc(PETSc.ScalarType(0.0), outlet_dofs_Q, Q)
    bcp = [bcp_out]

    # Temperature BCs: hot at inlet, cold at walls, natural at outlet
    T_hot = Function(QT)
    T_hot.x.array[:] = PETSc.ScalarType(thermo.T_hot)
    T_cold = Function(QT)
    T_cold.x.array[:] = PETSc.ScalarType(thermo.T_cold)

    inlet_dofs_T = locate_dofs_topological(QT, fdim, ft.find(markers.inlet))
    wall_dofs_T = locate_dofs_topological(QT, fdim, ft.find(markers.wall))
    bcT_in = dirichletbc(T_hot, inlet_dofs_T)
    bcT_wall = dirichletbc(T_cold, wall_dofs_T)
    bcT = [bcT_in, bcT_wall]

    # --- Unknowns / history --------------------------------------------------
    u = TrialFunction(V)
    v = TestFunction(V)
    p = TrialFunction(Q)
    q = TestFunction(Q)
    Tt = TrialFunction(QT)
    w = TestFunction(QT)

    u_ = Function(V, name="u")
    u_s = Function(V, name="u_tentative")
    u_n = Function(V, name="u_n")
    u_nm1 = Function(V, name="u_nm1")

    p_ = Function(Q, name="p")
    phi = Function(Q, name="phi")

    T_ = Function(QT, name="T")
    T_n = Function(QT, name="T_n")
    T_n.interpolate(lambda x: np.full(x.shape[1], thermo.T_hot, dtype=PETSc.ScalarType))

    # --- Temperature-dependent viscosity ------------------------------------
    def nu_of_T(Tf):
        return nu0 * exp(-a_vis * (Tf - T_ref))

    mu_eff = rho * nu_of_T(T_n)

    # --- Momentum tentative (Picard in time: mu from T_n) --------------------
    # AB2 convection and Crank–Nicolson-ish diffusion (as in your original HF)
    F1 = rho / k * dot(u - u_n, v) * dx
    F1 += inner(dot(1.5 * u_n - 0.5 * u_nm1, 0.5 * nabla_grad(u + u_n)), v) * dx
    F1 += inner(mu_eff * grad(0.5 * (u + u_n)), grad(v)) * dx
    F1 -= dot(p_, div(v)) * dx
    F1 += -rho * beta * (T_n - T_ref) * dot(g_vec, v) * dx

    a1, L1 = form(lhs(F1)), form(rhs(F1))
    A1 = create_matrix(a1)
    b1 = create_vector(L1)

    # Pressure Poisson
    a2 = form(dot(grad(p), grad(q)) * dx)
    L2 = form(-rho / k * dot(div(u_s), q) * dx)
    A2 = assemble_matrix(a2, bcs=bcp)
    A2.assemble()
    b2 = create_vector(L2)

    # Velocity correction
    a3 = form(rho * dot(u, v) * dx)
    L3 = form(rho * dot(u_s, v) * dx - k * dot(nabla_grad(phi), v) * dx)
    A3 = assemble_matrix(a3)
    A3.assemble()
    b3 = create_vector(L3)

    # Temperature: implicit diffusion, advection by u_
    aT = form(
        (1.0 / k) * Tt * w * dx
        + dot(u_, grad(Tt)) * w * dx
        + alpha * dot(grad(Tt), grad(w)) * dx
    )
    LT = form((1.0 / k) * T_n * w * dx)
    AT = assemble_matrix(aT, bcs=bcT)
    AT.assemble()
    bT = create_vector(LT)

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

    solverT = PETSc.KSP().create(mesh.comm)
    solverT.setOperators(AT)
    solverT.setType(PETSc.KSP.Type.GMRES)
    solverT.getPC().setType(PETSc.PC.Type.ILU)

    # --- Time loop -----------------------------------------------------------
    progress = None
    if time.progress and mesh.comm.rank == 0:
        progress = tqdm.autonotebook.tqdm(total=num_steps, desc="HF Boussinesq BFS")

    for _ in range(num_steps):
        if progress is not None:
            progress.update(1)

        # Step 1: tentative velocity (depends on mu_eff(T_n), so reassemble A1)
        A1.zeroEntries()
        assemble_matrix(A1, a1, bcs=bcu)
        A1.assemble()

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

        p_.x.petsc_vec.axpy(1.0, phi.x.petsc_vec)
        p_.x.scatter_forward()

        # Step 3: corrected velocity
        with b3.localForm() as loc:
            loc.set(0.0)
        assemble_vector(b3, L3)
        b3.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        solver3.solve(b3, u_.x.petsc_vec)
        u_.x.scatter_forward()

        # Step 4: temperature (advection by u_)
        AT.zeroEntries()
        assemble_matrix(AT, aT, bcs=bcT)
        AT.assemble()

        with bT.localForm() as loc:
            loc.set(0.0)
        assemble_vector(bT, LT)
        apply_lifting(bT, [aT], [bcT])
        bT.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(bT, bcT)
        solverT.solve(bT, T_.x.petsc_vec)
        T_.x.scatter_forward()

        # rotate histories
        with (
            u_.x.petsc_vec.localForm() as lu,
            u_n.x.petsc_vec.localForm() as lun,
            u_nm1.x.petsc_vec.localForm() as lunm1,
        ):
            lun.copy(lunm1)
            lu.copy(lun)

        T_n.x.array[:] = T_.x.array[:]

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
    prof = solve_boussinesq_bfs(
        geom=BFSGeometry(h1=h1, h2=h2, L_up=L_up, L_down=L_down), U_in=U_in
    )
    return prof.y, prof.u_x
