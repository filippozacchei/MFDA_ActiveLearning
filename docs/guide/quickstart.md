# Quickstart

This page is command-first. If you want a working run with minimal setup, follow
sections 1-3 in order.

---

## 1. Install

Prerequisites:

- Python 3.10
- C/C++ toolchain (`tinyDA`/`GPy` build dependencies)

```bash
git clone https://github.com/filippozacchei/MFDA_ActiveLearning.git
cd MFDA_ActiveLearning
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

macOS note:

```bash
xcode-select --install
brew install python@3.10
```

If your shell does not find `pytest`, run tests as:

```bash
python -m pytest -q
```

---

## 2. Run the toy forward workflow

```bash
python examples/toy_problem/run_forward_toy.py
```

This script:

- generates synthetic snapshots from the toy HF model,
- fits a POD basis and a multi-output GP,
- prints validation metrics (RMSE and coverage),
- produces diagnostic plots.

---

## 3. Run the toy inverse workflow

```bash
python examples/toy_problem/run_backward_toy.py
```

This script runs both:

- single-posterior active MCMC, and
- delayed-acceptance active MCMC with adaptive subchain updates.

It also plots posterior samples, HF usage, and subchain behavior.

---

## 4. Change run settings

Both toy scripts are notebook-style Python files with configuration constants near
the top. Edit those constants directly:

- `examples/toy_problem/run_forward_toy.py` for snapshot count, POD rank, GP kernel
- `examples/toy_problem/run_backward_toy.py` for chain length, burn-in, thresholds, and adaptive controls

There is currently no CLI argument parser for these scripts.

---

## 5. Use the docs notebooks

For step-by-step narrative versions of the same workflows:

- `docs/tutorials/forward_toy_notebook.py`
- `docs/tutorials/backward_toy_notebook.py`

Serve docs locally:

```bash
python -m pip install -e ".[docs]"
mkdocs serve
```

---

## 6. Next steps

- Read [Concepts](summary.md) for the model and algorithm details.
- Open the API reference pages for module-level docs.
- For the PDE benchmark, see `examples/navier_stokes/` and the Navier-Stokes tutorial page.
