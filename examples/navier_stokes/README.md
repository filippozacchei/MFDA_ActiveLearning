# Navier–Stokes (FEniCSx) example: backward-facing step (BFS)

This folder contains an **advanced** example based on a medium/high-fidelity incompressible
Navier–Stokes solver implemented with **FEniCSx/DOLFINx** (PETSc, MPI, Gmsh). It is intended
as a realistic demonstration of the workflow used by `gp_active_mcmc` on PDE-based problems.

**Status in the documentation**
- This example is **not executed** as part of the MkDocs build.
- It is kept in `examples/` to avoid imposing heavy dependencies (FEniCSx stack) on the
  core library and documentation environment.
- The tutorial-style notebooks/scripts can still be run locally when the PDE environment
  is available.

---

## Scientific summary

### Governing equations
We solve the **incompressible Navier–Stokes equations** in a 2D backward-facing-step domain:

\[
\rho(\partial_t u + (u\cdot\nabla)u) - \mu \Delta u + \nabla p = 0,
\qquad
\nabla \cdot u = 0.
\]

with density `rho` and kinematic viscosity `nu` (dynamic viscosity `mu = rho * nu`).

### Domain (backward-facing step)
The geometry is parameterised by:

- upstream channel height: `h1`
- downstream channel height: `h2`
- upstream length: `L_up`
- downstream length: `L_down`

See `utils/types.py` (`BFSGeometry`).

### Boundary conditions
- **Inlet**: uniform velocity profile \((u_x, u_y) = (U_in, 0)\)
- **Walls**: no-slip \((u_x, u_y) = (0, 0)\)
- **Outlet**: pressure Dirichlet \(p = 0\) (a pragmatic choice for this demo)

### Discretisation and solver
- Spatial discretisation: Taylor–Hood (P2 velocity, P1 pressure)
- Time discretisation: splitting scheme (IPCS-like)
  - tentative velocity: explicit convection (AB2 advecting velocity), implicit diffusion
  - pressure Poisson solve
  - velocity correction

Implementation: `utils/solver.py` (function `solve_ipcs_bfs`).

### Quantity of interest (QoI): outlet profile
The QoI is the **streamwise outlet velocity profile** \(u_x(y)\) sampled on the outlet line \(x=L\),
with `ny` sampling points. See `utils/outlet.py` (`sample_outlet_u_x`).

For surrogate modelling we often resample the profile to a fixed length `T`
(e.g. `T=100`) using linear interpolation (`resample_profile`).

---

## Repository layout

- `utils/types.py`  
  Dataclasses for geometry, mesh options, boundary markers, and QoI containers.

- `utils/bfs_mesh.py`  
  Gmsh-based mesh generator for the BFS domain + boundary facet tags.

- `utils/solver.py`  
  FEniCSx/DOLFINx IPCS-like solver producing the outlet velocity profile.

- `utils/outlet.py`  
  Outlet sampling and resampling utilities.

- `utils/animation.py`  
  Helper to generate a GIF/MP4 by evaluating a velocity field on a structured grid.

- `cfd/run_mf.py` (or `run_mf.py`)  
  Lightweight script to sweep parameters and plot outlet profiles.

- `run_forward_*.py`  
  Notebook-style script to build and validate a POD–GP surrogate for the outlet profile.

- `run_backward_*.py`  
  Notebook-style script to run Active/Adaptive MCMC for the inverse problem.

---

## Installation notes (FEniCSx environment)

This example requires a working FEniCSx/DOLFINx installation with:
- `dolfinx`
- `basix`, `ufl`
- `petsc4py`
- `mpi4py`
- `gmsh` (Python API)

Recommended practice:
1. Create a dedicated environment for the PDE stack.
2. Verify MPI and PETSc are functional (the solver uses PETSc KSP).
3. Ensure the Gmsh Python module matches your installed Gmsh.

This is intentionally not pinned here, because installation details vary substantially across:
- Linux vs macOS,
- conda-forge vs system MPI/PETSc,
- workstation vs HPC cluster modules.

If you maintain your project with pinned environments, consider adding:
- `examples/navier-stokes/environment.yml` (conda) **or**
- `examples/navier-stokes/requirements.txt` (pip, if appropriate)

---

## Quick run: outlet profiles

### Sweep inlet velocity
From the repository root:

```bash
python examples/navier-stokes/cfd/run_mf.py
