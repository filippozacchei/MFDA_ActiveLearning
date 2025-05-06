# Research Agenda and Hints for DA-GP Algorithm

This document outlines the research questions and methodological guidance for a proposed **Delayed Acceptance (DA)** and **Gaussian Process (GP)** based algorithm for active learning and surrogate model construction.

---

## I. Theoretical / Methodological Design

### 1. What are the theoretical foundations for using Delayed Acceptance in the context of GP-based surrogate modeling?
- Use DA-MCMC literature (e.g., Christen & Fox, 2005) as the basis.
- Frame DA as a two-stage acceptance scheme: use GP to pre-filter proposals before evaluating the true model.

### 2. How can one formally define and monitor the “fidelity” or “convergence” of the GP surrogate?
- Metrics:
  - Predictive variance (mean or max).
  - Leave-one-out cross-validation error.
  - Stabilization of marginal likelihood.
- Thresholds on these metrics can trigger trust in the GP.

### 3. What criteria should be used to trigger the switch from evaluating the forward model to relying solely on the GP?
- Example criteria:
  - Predictive variance below a threshold across recent samples.
  - No significant change in GP posterior over several updates.
  - Stagnant improvement in log marginal likelihood.

---

## II. Algorithm Design

### 4. How is the Schur complement used to update the Cholesky decomposition in an online fashion?
- When a new point is added:
  \[
  K_{n+1} = \begin{bmatrix} K_n & k \\ k^T & k_{new} \end{bmatrix}
  \]
  Update \( L_{n+1} \) using Schur complement logic to avoid full recomputation.

### 5. What are the computational benefits of online learning for GP versus full retraining?
- Full retraining: \( \mathcal{O}(n^3) \)
- Online update: \( \mathcal{O}(n^2) \)
- Enables adaptive, low-latency learning.

### 6. Can the DA-GP framework maintain posterior accuracy compared to full MCMC or MCMC-GP hybrid approaches?
- Use metrics such as:
  - KL divergence.
  - Wasserstein distance.
  - Posterior moment comparison.
- Empirically validate using controlled test functions.

---

## III. Evaluation and Benchmarking

### 7. How does the proposed DA-GP method perform on standard test functions?
- Benchmark on Branin, Hartmann, Rosenbrock, etc.
- Track:
  - Posterior accuracy.
  - Runtime.
  - GP prediction error (RMSE).

### 8. How does sample efficiency, acceptance rate, and computational cost compare with traditional MCMC-guided GP approaches?
- Evaluate:
  - Number of model evaluations.
  - Acceptance rate pre/post DA.
  - Time per iteration and total convergence time.

### 9. How does performance scale with dimensionality and noise levels in the model?
- Increase input dimension from 2D to 10D+.
- Add Gaussian noise to the output.
- Measure effect on GP stability, DA rejection rate, and accuracy.

---

## IV. Broader Impact and Scalability

### 10. Under what conditions does the DA-GP approach break down or lose effectiveness?
- Scenarios:
  - High-dimensional or sparse data.
  - Poor GP kernel tuning.
  - Non-smooth/multimodal likelihoods.

### 11. Can this method be generalized to other surrogate models?
- Replace GP with:
  - Bayesian neural networks.
  - Ensemble trees with uncertainty quantification.
- Maintain DA logic: cheap filter before costly evaluation.

### 12. How can the DA-GP framework be extended for real-time or streaming data scenarios?
- Use online GP updates with:
  - Sliding window or forgetting factors.
  - Real-time adaptation to data drift.

---

## Next Steps

- [ ] Draft pseudocode of the DA-GP algorithm.
- [ ] Design experimental setup with benchmarks and metrics.
- [ ] Write a literature review comparing MCMC-GP, DA-GP, and pure GP methods.
