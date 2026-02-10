# GP-Active MCMC

`gp_active_mcmc` is a research-oriented Python package for **multi-fidelity Bayesian inference**
using **Gaussian-process surrogates** and **Active/Adaptive MCMC** strategies. It couples a fast,
uncertain *low-fidelity* (LF) surrogate with an accurate but expensive *high-fidelity* (HF) model,
switching to HF evaluations when needed and optionally adapting the HF subchain length during sampling.

The goal is to rely on the GP surrogate as much as possible during MCMC sampling, and leverage GP predictive uncertainty to decide when/where to switch to HF solver. This mantains the accuracy of teh MCMC sampling and reduces at minimum the costs of the data-fit model training phase.

---

## Installation

### Core package

The package is developed and tested with the following pinned versions:

- Python `3.10.19`
- NumPy `1.26.4`
- SciPy `1.12.0`
- scikit-learn `1.7.2`
- tinyDA `0.9.21`
- Ray `2.53.0`
- GPy `1.13.2`

A typical development install:

```bash
pip install -e .
```

Run tests:

```bash
pytest -q
```

---

## Optional dependencies

### Documentation (MkDocs)

Documentation is generated with:

- mkdocs `1.6.1`

Install documentation dependencies (if you enable the `docs` extra in `pyproject.toml`):

```bash
pip install -e ".[docs]"
mkdocs serve
```

### Examples: Navier–Stokes (FEniCS/DOLFINx stack)

The Navier–Stokes example uses the FEniCS/DOLFINx ecosystem and a standard MPI toolchain.
These are typically installed via **conda-forge** (not reliably via `pip`):

- `fenics-dolfinx=0.9.0` (conda-forge)
- `mpich` (conda-forge)
- `gmsh=4.15.0`
- `python-gmsh=4.15.0`
- `pyvista`

A representative conda installation (adjust to your environment):

```bash
conda install -c conda-forge fenics-dolfinx=0.9.0 mpich gmsh=4.15.0 python-gmsh=4.15.0 pyvista
```

---

## Package structure

Main subpackages:

- [`gp_active_mcmc.surrogates`][]: POD, GP, and POD–GP surrogate components.
- [`gp_active_mcmc.inference`][]: likelihoods, proposals, adaptive subchain policy, samplers, and result containers.
- [`gp_active_mcmc.utils`][]: pure numerical helpers and post-processing utilities (no plotting).
- [`gp_active_mcmc.diagnostics`][]: plotting utilities returning `(fig, ax)` (no `plt.show()` calls).

Examples are located in `examples/`:
- `examples/toy_problem/`
- `examples/navier-stokes/`

---

## Core concepts

### Multi-fidelity active sampling
At each evaluation, the sampler uses the LF surrogate unless the surrogate uncertainty exceeds a threshold.
When HF is used, the surrogate is updated online.

### Adaptive subchains (optional)
When enabled, the HF subsampling rate (HF correction frequency) is adjusted online based on the
surrogate–HF discrepancy, aiming to reduce HF calls while controlling error.

### Results
Sampling functions return a [`gp_active_mcmc.inference.SamplingResult`][] containing:
- a [`gp_active_mcmc.inference.MCMCChain`][] of samples and aligned per-step extras (e.g., HF usage flags, subchain length history),
- flexible `metadata` describing the run configuration.



---

## License

See `LICENSE`.
