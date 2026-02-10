# Examples

The repository includes two **notebook-style** toy tutorials intended for documentation and onboarding.
They prioritise clarity and API usage over performance, and can be read either as:

- a VSCode *Python Interactive* file (`# %%` cells), or
- a plain Python script executed top-to-bottom, or
- a rendered notebook page in the MkDocs site (recommended).

The toy tutorials cover the two main entrypoints of the library:

- **Forward workflow**: build and validate a POD–GP surrogate.
- **Backward workflow**: Bayesian inversion with Active / DA-MCMC (optionally adaptive).

---

## Where to find the tutorials

In the documentation navigation:

- **Tutorials → Forward toy (POD-GP)**: `tutorials/forward_toy_notebook.py`
- **Tutorials → Backward toy (Active-MCMC)**: `tutorials/backward_toy_notebook.py`

If you are browsing the repository:

- `tutorials/forward_toy_notebook.py`
- `tutorials/backward_toy_notebook.py`

> If you still have legacy scripts under `examples/`, keep using them if you want.
> For the documentation build, the tutorial files under `tutorials/` are the canonical versions.

---

## Running the toy tutorials locally

From the repository root:

```bash
python tutorials/forward_toy_notebook.py
python tutorials/backward_toy_notebook.py
```

### VSCode (cell-by-cell)
Open either file and run cells using the Python extension.

### MkDocs build
With `mkdocs-jupyter` enabled, the tutorial pages can be executed and rendered during the docs build.
Plots will be embedded in the page (the tutorial files call the diagnostics functions with `show=True`
when appropriate).

---

## Forward toy: POD–GP surrogate

**Goal:** build a reduced-order surrogate for a trajectory-valued forward model, and assess prediction quality.

### What you learn
- how to generate training data from a prior and a forward model,
- how to fit a POD basis and project snapshots into POD coefficients,
- how to fit a multi-output GP on coefficients,
- how to call `PODGPSurrogate.predict(theta)` to obtain a trajectory mean and uncertainty,
- how to sanity-check POD truncation and basic calibration.

### Key objects
- [`POD`][gp_active_mcmc.surrogates.pod.POD] (basis construction)
- [`MultiOutputGP`][gp_active_mcmc.surrogates.gp.MultiOutputGP] (GP regression on coefficients)
- [`PODGPSurrogate`][gp_active_mcmc.surrogates.podgp.PODGPSurrogate] (end-to-end surrogate)

### Typical outputs
- POD energy curves (helps choose a rank),
- summary metrics on a held-out test set (RMSE + coverage),
- trajectory plots at representative test parameters,
- calibration plot: error vs predicted uncertainty.

---

## Backward toy: Bayesian inversion with Active / DA-MCMC

**Goal:** solve a small Bayesian inverse problem while learning the surrogate online.

### What you learn
- how to generate a synthetic observation from the HF model,
- how to fit a POD–GP surrogate and use it as LF,
- how to couple LF and HF with the active model,
- how to run either single-level Active-MCMC or two-level DA-MCMC,
- how to enable adaptive HF correction frequency with adaptive subchains,
- how to inspect HF usage and (optional) subchain-length history.

### Key objects
- [`ActiveMCMCModel`][gp_active_mcmc.inference.model.ActiveMCMCModel] (core LF/HF coupling)
- [`ActiveGPLogLike`][gp_active_mcmc.inference.likelihood.ActiveGPLogLike] (surrogate-aware Gaussian likelihood)
- [`AdaptiveSubchain`][gp_active_mcmc.inference.adaptive_subchain.AdaptiveSubchain] (adaptive HF correction frequency)
- [`AdaptiveMetropolisShared`][gp_active_mcmc.inference.proposal.AdaptiveMetropolisShared] (proposal with controlled deepcopy semantics)
- [`sample_active_chain`][gp_active_mcmc.inference.sampling.sample_active_chain] (fixed subsampling)
- [`sample_adaptive_active_chain`][gp_active_mcmc.inference.sampling.sample_adaptive_active_chain] (chunked adaptive subsampling)

---

## Choosing the inference mode (most important design choice)

The *posterior argument* determines how sampling interacts with the active model.

### Single posterior → MCMC-guided active learning
Use only the coarse model:

```python
post = tda.Posterior(prior, loglike, model.coarse)
result = sample_active_chain(
    model=model,
    posterior=post,
    subsampling_rate=1,
    ...
)
```

HF calls occur only when triggered inside `model.coarse(...)`.

### Two posteriors → DA-MCMC guided active learning
Use both coarse and fine posteriors:

```python
post_coarse = tda.Posterior(prior, loglike_coarse, model.coarse)
post_fine   = tda.Posterior(prior, loglike_fine, model.fine)

result = sample_active_chain(
    model=model,
    posterior=[post_coarse, post_fine],
    subsampling_rate=K,
    ...
)
```

This corresponds to delayed-acceptance MCMC (DA-MCMC).

### Adaptive DA-MCMC (recommended)
Enable adaptive subchains **and** use two posteriors:

```python
model = ActiveMCMCModel(..., adaptive=AdaptiveSubchain(...))

result = sample_adaptive_active_chain(
    model=model,
    posterior=[post_coarse, post_fine],
    ...
)
```

In this mode, DA-MCMC is mandatory: adaptive subchains tune *how often* the fine correction is applied.

---

## Reading the outputs

Both samplers return a [`SamplingResult`][gp_active_mcmc.inference.chain.SamplingResult] with:

- `result.chain.samples`: array of shape `(n_steps, n_dim)`,
- `result.chain.extras.used_hf`: HF usage flag aligned with coarse-chain samples,
- `result.chain.extras.subchain_length`: subchain-length history (present for adaptive runs).

For plotting:
- diagnostics functions live in [`gp_active_mcmc.diagnostics`][gp_active_mcmc.diagnostics],
- they return `(fig, ax)` and do not call `plt.show()` unless you pass `show=True`.
