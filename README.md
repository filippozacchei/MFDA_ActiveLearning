# GP-Active MCMC

`gp_active_mcmc` is a research-grade Python package for **multi-fidelity Bayesian inference** using **Gaussian-process surrogates** and **Active/Adaptive MCMC** strategies. It couples a fast, uncertain _low-fidelity_ (LF) surrogate with an accurate but expensive _high-fidelity_ (HF) model, switching to HF evaluations when needed and optionally adapting the HF subchain length during sampling.

Use it when you need to:

- prototype active-learning workflows on toy problems (`docs/tutorials/forward_toy_notebook.py`);
- couple POD–GP surrogates with delayed-acceptance / adaptive MCMC;
- run PDE-scale examples such as the Navier–Stokes backward-facing-step benchmark;
- collect diagnostics (HF usage, subchain history, predictive error) for publications.

Hosted documentation: https://filippozacchei.github.io/MFDA_ActiveLearning/

[![Tests](https://github.com/filippozacchei/MFDA_ActiveLearning/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/filippozacchei/MFDA_ActiveLearning/actions/workflows/ci.yml)
[![Docs](https://github.com/filippozacchei/MFDA_ActiveLearning/actions/workflows/docs.yml/badge.svg?branch=main)](https://github.com/filippozacchei/MFDA_ActiveLearning/actions/workflows/docs.yml)
[![Documentation](https://img.shields.io/badge/docs-mkdocs%20material-blue)](https://filippozacchei.github.io/MFDA_ActiveLearning/)

---

## Installation

The package is developed and tested with pinned versions to stay compatible with `tinyDA` and `GPy`:

- Python `3.10.19`
- NumPy `1.26.4`
- SciPy `1.12.0`
- scikit-learn `1.7.2`
- tinyDA `0.9.21`
- Ray `2.53.0` (transitively required by `tinyDA`)
- GPy `1.13.2`

### Core install

```bash
git clone https://github.com/filippozacchei/MFDA_ActiveLearning.git
cd MFDA_ActiveLearning
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

### macOS install (Apple Silicon and Intel)

Install system prerequisites first:

```bash
xcode-select --install
brew install python@3.10
```

Then create the project environment with the pinned interpreter:

```bash
git clone https://github.com/filippozacchei/MFDA_ActiveLearning.git
cd MFDA_ActiveLearning
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
```

If you use Conda/Mamba on macOS, you can also create a `python=3.10` environment
and run the same `pip install -e .` command inside it.

Smoke-test the install:

```bash
python examples/toy_problem/run_forward_toy.py
```

> **Why is Ray pinned?** `tinyDA` currently depends on `ray>=2.53.0`. Even if you do not call Ray
> explicitly, the pin ensures our stack stays synchronized with upstream releases.

---

## Optional dependencies

### Testing

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

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

- \[`gp_active_mcmc.surrogates`\]\[\]: POD, GP, and POD–GP surrogate components.
- \[`gp_active_mcmc.inference`\]\[\]: likelihoods, proposals, adaptive subchain policy, samplers, and result containers.
- \[`gp_active_mcmc.utils`\]\[\]: pure numerical helpers and post-processing utilities (no plotting).
- \[`gp_active_mcmc.diagnostics`\]\[\]: plotting utilities returning `(fig, ax)` (no `plt.show()` calls).

Examples are located in `examples/`:

- `examples/toy_problem/`
- `examples/navier-stokes/`

---

## Quickstart workflow

1. **Generate surrogate snapshots**
   ```bash
   python examples/toy_problem/run_forward_toy.py --n-snapshots 200 --pod-rank 20
   ```
2. **Run the inverse problem**
   ```bash
   python examples/toy_problem/run_backward_toy.py --mode adaptive --n-evals 1000
   ```
3. **Inspect diagnostics**
   Use `gp_active_mcmc.diagnostics` plotting helpers or run `mkdocs serve` and open the _Tutorials_
   section.

For the Navier–Stokes benchmark, provision a conda environment with the requirements listed in the
documentation block above, then run the scripts under `examples/navier_stokes/`.

## Development workflow

- Install dev extras with `pip install -e ".[dev]"`.
- Run the `nox` sessions locally (`tests`, `lint`, `typecheck`, `docs`) before opening a PR.
- Enable pre-commit hooks via `pre-commit install`.
- See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidance.

## Governance

- Community expectations: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
- Contribution process: [CONTRIBUTING.md](CONTRIBUTING.md)
- Citation info: [CITATION.cff](CITATION.cff)

## Background reading

- [MCMC-guided active learning with Gaussian-process surrogates](https://www.sciencedirect.com/science/article/pii/S099775382600001X) – core strategy this project implements and extends with POD regression.
- [Delayed-acceptance multi-level data assimilation (MLDA)](https://epubs.siam.org/doi/10.1137/22M1476770) – reference for the MLDA/tinyDA library that powers the delayed-acceptance workflow.

---

## Core concepts

### Multi-fidelity active sampling

At each evaluation, the sampler uses the LF surrogate unless the surrogate uncertainty exceeds a threshold.
When HF is used, the surrogate is updated online.

### Adaptive subchains (optional)

When enabled, the HF subsampling rate (HF correction frequency) is adjusted online based on the
surrogate–HF discrepancy, aiming to reduce HF calls while controlling error.

### Results

Sampling functions return a \[`gp_active_mcmc.inference.SamplingResult`\][] containing:

- a \[`gp_active_mcmc.inference.MCMCChain`\][] of samples and aligned per-step extras (e.g., HF usage flags, subchain length history),
- flexible `metadata` describing the run configuration.

---

## License

This project is distributed under the MIT License (see [`LICENSE`](LICENSE)).
