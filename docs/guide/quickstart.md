# Quickstart

This guide walks you through the fastest path to a functioning
`gp_active_mcmc` environment and a reproducible toy experiment. You will:

1. create a dedicated Python environment,
2. build a POD–GP surrogate from synthetic data,
3. wrap the surrogate + HF model in `ActiveMCMCModel`,
4. run both fixed-rate and adaptive active-learning samplers,
5. inspect diagnostics to confirm everything worked.

---

## 0. Prerequisites

- Linux or macOS shell with Python 3.10 (matching CI).
- A working C/C++ toolchain (required by `tinyDA`, `GPy`, and scientific deps).
- Optional: conda-forge environment with `fenics-dolfinx`, `mpich`, `gmsh`,
  `python-gmsh`, and `pyvista` if you plan to run the Navier–Stokes benchmark.

---

## 1. Set up your environment

```bash
git clone https://github.com/filippozacchei/MFDA_ActiveLearning.git
cd MFDA_ActiveLearning
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pre-commit install
```

Run the test suite to ensure the stack is healthy:

```bash
pytest -q
```

> **Heads-up:** `tinyDA` currently depends on `ray==2.53.0`. Even if your workflow
> does not use Ray, keep the pin to remain compatible with upstream releases.

---

## 2. Build the toy surrogate

Use the ready-made script (mirrors the _Forward toy_ tutorial):

```bash
python examples/toy_problem/run_forward_toy.py \
  --n-snapshots 200 \
  --test-fraction 0.25 \
  --pod-rank 20 \
  --kernel matern52
```

This produces:

- a POD basis fitted on HF snapshots (`rank=20`),
- a multi-output GP mapping parameters → POD coefficients,
- quick metrics (RMSE, predictive std, coverage) printed to stdout,
- optional plots (POD energy, error vs uncertainty, prediction slices).

You can interactively reproduce the steps by opening
`docs/tutorials/forward_toy_notebook.py` in your IDE or via `mkdocs serve`.

---

## 3. Configure the active model

The minimal pattern:

```python
from gp_active_mcmc.inference import ActiveMCMCModel, ActiveGPLogLike
from gp_active_mcmc.inference import (
    AdaptiveSubchain,
    AdaptiveSubchainControl,
    AdaptiveSubchainState,
)
from gp_active_mcmc.surrogates import PODGPSurrogate

lf_surrogate = PODGPSurrogate(pod=pod, gp=gp)
model = ActiveMCMCModel(
    lf_model=lf_surrogate,
    hf_model=hf_forward,
    gamma_threshold=0.10,
    adaptive=AdaptiveSubchain(
        state=AdaptiveSubchainState(subchain_length=20),
        control=AdaptiveSubchainControl(
            update_every=5,
            target_error=0.05,
            min_subchain=1,
            max_subchain=500,
        ),
    ),
)

loglike = ActiveGPLogLike(y_obs, C_obs)
posterior = [
    tda.Posterior(prior, loglike, model.coarse),
    tda.Posterior(prior, loglike, model.fine),
]
```

Key decisions:

- **Single posterior** → LF-only active learning (simple but no formal DA correction).
- **Two posteriors** → delayed-acceptance Active MCMC; required for adaptive subchains.

---

## 4. Run sampling

Fixed subsampling rate (single `tinyDA` call):

```python
from gp_active_mcmc.inference import sample_active_chain

result = sample_active_chain(
    model=model,
    posterior=posterior,
    proposal=proposal,
    iterations=2000,
    initial_parameters=theta0,
    subsampling_rate=50,
    chain_key="chain_0",
)
```

Adaptive subchain workflow (chunked runs, subsampling changes online):

```python
from gp_active_mcmc.inference import sample_adaptive_active_chain, ChunkedMCMCConfig

result = sample_adaptive_active_chain(
    model=model,
    posterior=posterior,
    proposal=proposal,
    n_chunks=8,
    chunk_config=ChunkedMCMCConfig(chain_key="chain_coarse_0", chunk_size=250),
    initial_parameters=theta0,
    subsampling_rate=50,
)
```

Both functions return a `SamplingResult` containing an immutable `MCMCChain`
with HF usage flags, (optional) acceptance flags, and subchain history.

---

## 5. Inspect diagnostics

```python
from gp_active_mcmc.diagnostics import (
    plot_chain_2d,
    plot_cumulative_hf_fraction,
    plot_subchain_length_history,
)

chain = result.chain
plot_chain_2d(chain.samples, theta_true=theta_star)
plot_cumulative_hf_fraction(chain.extras.used_hf, burn_in=100)
plot_subchain_length_history(chain.extras.subchain_length)
```

For publication-ready figures, reuse the helper functions in `docs/tutorials`
or run `mkdocs build --strict` to regenerate the site (CI executes the same command).

---

## Where to go next

- **User Guide → Concepts** for deeper explanations of adaptive policies and likelihood design.
- **Tutorials → Navier–Stokes** for a PDE-scale benchmark (remember to provision the MPI/FEniCS stack).
- **API Reference** for exhaustive documentation of each module.
- **Contributing** if you plan to propose changes—run `nox -s tests lint typecheck docs`.

Happy sampling!
