# Project: Gaussian Process regression for Parameter Identification 

The goal of the project is to provide a robust Gaussian Process regression methodology for parameter identification of MEMS accelerometer. 


Paper to read:
- [ ] GPR Stuart
- [ ] Dongwei Ye
- [ ] Kernel GP
- [ ] Neurips
- [ ] MultiFidelity GP


# Stuff to do:

- Implement paper like MCMC-guided active learning GP
- Switch to DA-MCMC guided active learning GP


How to compare the two algorithms:
1. Robustness wrt Number of Initial Training Points
2. Robustness wrt to GP Kernel (if possible, try NN kernel)
3. Overall Efficiency in terms of HF evaluations and total CPU time
4. Robustenss wrt to initial MCMC point
5. Effects of various uncertianty modelling


Include uncertainty of the GP in likelihhod calculation.
