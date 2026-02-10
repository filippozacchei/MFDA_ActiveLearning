# Inference

This section documents the inference-facing API: the **active model** (the core abstraction),
the **adaptive subchain** policy, the sampling entrypoints, likelihoods, proposals, and
result containers.

---

## Conceptual overview

The library implements *active learning inside MCMC* by coupling two forward models:

- **LF model** (low-fidelity): typically a surrogate (e.g., POD–GP) that is cheap but imperfect.
- **HF model** (high-fidelity): the “true” forward model that is accurate but expensive.

The central idea is to evaluate the LF model most of the time, and selectively call the HF
model to correct bias and (optionally) improve the LF model online.

### The heart of the library: `ActiveMCMCModel`

`ActiveMCMCModel` is the core component. It is responsible for:

1. **Coupling LF and HF** evaluation pathways,
2. **Deciding when to trigger HF** (e.g., via an uncertainty threshold),
3. **Updating the LF surrogate** when HF evaluations become available (active learning),
4. Exposing callables compatible with `tinyDA` posteriors and samplers.

::: gp_active_mcmc.inference.model.ActiveMCMCModel

---

## Choosing the inference mode

The sampling behavior is mainly determined by two user choices:

1. **Which `posterior` objects you pass to the sampler**, and
2. **Whether you pass an `AdaptiveSubchain` policy to `ActiveMCMCModel`**.

These choices define three practical modes of use.

### Mode A — MCMC-guided active learning (single posterior)

If you pass **one posterior** (a single `tinyDA.Posterior`), the chain is driven by one model
evaluation function (typically `model.coarse`).

- Interpretation: *MCMC is used to explore the parameter space; active learning is guided by
  internal LF uncertainty/HF triggers inside the active model.*
- Use when: you want a simple workflow and a single-level chain.

**Rule of thumb**
- `posterior = Posterior(prior, loglike, model.coarse)`
- call `sample_active_chain(...)`

### Mode B — DA-MCMC guided active learning (two posteriors)

If you pass a **list of two posteriors** (`[coarse, fine]`), the sampler runs a delayed-acceptance
scheme:

- **coarse posterior**: cheap evaluation (LF-first via `model.coarse`)
- **fine posterior**: expensive evaluation (HF via `model.fine`)

This corresponds to **DA-MCMC guided active learning**.

**Rule of thumb**
- `posterior = [Posterior(..., model.coarse), Posterior(..., model.fine)]`
- call `sample_active_chain(...)`

The two-posterior choice is *essential*: it changes the Markov chain mechanism and ensures
the HF model participates in acceptance/rejection in a principled delayed-acceptance way.

### Mode C — DA-MCMC guided active learning with *adaptive* subchains (recommended)

If you want the best-supported “active DA” workflow in this library, use:

- a **two-posterior** setup (DA-MCMC is mandatory), and
- an `AdaptiveSubchain` policy passed into `ActiveMCMCModel`.

In this configuration:
- DA-MCMC controls *how* coarse and fine posteriors interact,
- the **adaptive subchain** controls *how often* the fine (HF) correction is applied,
  by monitoring an LF–HF discrepancy signal online and updating the subsampling rate.

This is the recommended approach when HF is expensive and you want to **adaptively trade**
accuracy and cost during sampling.

**Rule of thumb**
- Construct `ActiveMCMCModel(..., adaptive=AdaptiveSubchain(...))`
- `posterior = [Posterior(..., model.coarse), Posterior(..., model.fine)]`
- call `sample_adaptive_active_chain(...)` (chunked sampling so the subchain length can change)

---

## Adaptive subchain policy

`AdaptiveSubchain` adapts the HF correction frequency online based on an LF–HF discrepancy signal.
Conceptually, it manages a **subchain length**: the number of coarse steps taken between fine
corrections.

- Short subchains ⇒ more frequent HF corrections (more expensive, typically more accurate)
- Long subchains  ⇒ less frequent HF corrections (cheaper, higher risk if LF is poor)

::: gp_active_mcmc.inference.adaptive_subchain.AdaptiveSubchain

### AdaptiveSubchainState

Stores the evolving state (e.g., current subchain length, error history).

::: gp_active_mcmc.inference.adaptive_subchain.AdaptiveSubchainState

### AdaptiveSubchainControl

Stores user-facing control parameters (update frequency, targets, bounds).

::: gp_active_mcmc.inference.adaptive_subchain.AdaptiveSubchainControl

---

## Samplers

High-level entrypoints that run `tinyDA` and return a `SamplingResult`.

### `sample_active_chain`

Use this when:
- you run **single-posterior** active learning (Mode A), or
- you run **two-posterior DA-MCMC** with a *fixed* correction behavior (Mode B).

::: gp_active_mcmc.inference.sampling.sample_active_chain

### `sample_adaptive_active_chain` (chunked, adaptive)

Use this when:
- you run **DA-MCMC with adaptive subchains** (Mode C).

The adaptive subchain length may change over time, so sampling is performed in **chunks**
(`ChunkedMCMCConfig`) to re-enter `tinyDA` periodically and update the subsampling rate.

::: gp_active_mcmc.inference.sampling.sample_adaptive_active_chain

::: gp_active_mcmc.inference.sampling.ChunkedMCMCConfig

---

## Likelihood

`ActiveGPLogLike` is a Gaussian log-likelihood designed for active surrogates. When LF is used,
it can inflate the observation covariance using surrogate predictive variance; when HF is used,
it reduces to the standard Gaussian likelihood with the provided covariance.

::: gp_active_mcmc.inference.likelihood.ActiveGPLogLike

---

## Proposal

Adaptive Metropolis proposal intended for chunked/active workflows. It supports sharing state
across deepcopies (useful when chunking re-enters `tinyDA`).

::: gp_active_mcmc.inference.proposal.AdaptiveMetropolisShared

---

## Results

### ChainExtras

Per-step metadata aligned with samples (HF usage flags, acceptance flags, subchain length history).

::: gp_active_mcmc.inference.chain.ChainExtras

### MCMCChain

Immutable container for samples plus aligned per-step extras.

::: gp_active_mcmc.inference.chain.MCMCChain

### SamplingResult

Sampler output: `(chain, metadata)`.

::: gp_active_mcmc.inference.chain.SamplingResult
