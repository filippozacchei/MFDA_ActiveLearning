# Solving Bayesian inverse problems with expensive likelihoods using constrained Gaussian processes and active learning


**Abstract:** model the uncertainties of the surrogate in order to incorporate tehe epistemic uncertainty due to limited data. Approximate the log-likelihood by a Constrained Gaussian Process based on prior knowledge about its boundedness. State-of-the art active learning startegy for selecting training points. 

##Methodology: 
(Computationally expensive) forward model $\mathcal{M} : R^{n_x} \rightarrow R^{n_obs}$: $$\boldsymbol{y}_{\mathrm{obs}}=\mathcal{M}(\boldsymbol{x})+\boldsymbol{\epsilon}.$$

#### Surrogate Modeling of the Log Likelihood
The exponent of the likelihood function $f(x) = − 1/2 \|y_{obs} − \mathcal{M}(\boldsymbol{x})\|^2_{\Sigma_n}$ is approximated by a Gaussian process.

