"""Quantify model error: how well can 3 piecewise parameters reproduce the true obs?"""
import sys; sys.path.insert(0, '../../src')
import h5py, numpy as np
from scipy.optimize import minimize

f = h5py.File('model/ProblemDefinition.h5', 'r')
x = np.array(f['/ForwardModel/NodeLocations']).ravel()
loads = np.array(f['/ForwardModel/Loads'])
modulus_true = np.array(f['/ForwardModel/Modulus'])
u_true = np.array(f['/ForwardModel/TrueDisplacement'])
radius = float(f['/ForwardModel'].attrs['BeamRadius'])
length = float(f['/ForwardModel'].attrs['BeamLength'])
B_obs = np.array(f['/Observations/ObservationMatrix'])
f.close()

n = len(x); dx = length/(n-1); I = np.pi/4*radius**4
obs_idx = np.sort(np.where(B_obs == 1.0)[1])
B = np.zeros((len(obs_idx), n))
for j, i in enumerate(obs_idx): B[j,i] = 1.0
y_obs_true = B @ u_true

def build_K(E, n, dx):
    K = np.zeros((n,n))
    for i in range(2, n-2):
        K[i,i+2]=E[i]; K[i,i+1]=E[i+1]-6*E[i]+E[i-1]
        K[i,i]=-2*E[i+1]+10*E[i]-2*E[i-1]; K[i,i-1]=E[i+1]-6*E[i]+E[i-1]; K[i,i-2]=E[i]
    K[1,3]=E[1]; K[1,2]=E[2]-6*E[1]+E[0]; K[1,1]=-2*E[2]+11*E[1]-2*E[0]
    K[n-2,n-1]=E[n-1]-4*E[n-2]+E[n-3]; K[n-2,n-2]=-2*E[n-1]+9*E[n-2]-2*E[n-3]
    K[n-2,n-3]=E[n-1]-6*E[n-2]+E[n-3]; K[n-2,n-4]=E[n-2]
    K[n-1,n-1]=2*E[n-1]; K[n-1,n-2]=-4*E[n-1]; K[n-1,n-3]=2*E[n-1]
    K[0,:]=0; K[:,0]=0; K[0,0]=1
    return K/dx**4

def forward(theta):
    xi = x / length
    logE = np.empty_like(x)
    logE[(xi >= 0.0) & (xi <= 1./3)] = theta[0]
    logE[(xi > 1./3) & (xi <= 2./3)] = theta[1]
    logE[(xi > 2./3) & (xi <= 1.0)]  = theta[2]
    E = np.exp(logE)
    K = build_K(E, n, dx)
    rhs = loads/I; rhs[0] = 0.0
    return np.linalg.solve(K, rhs)

def misfit(theta):
    u = forward(theta)
    return np.sum((B @ u - y_obs_true)**2)

# Find best-fit theta
xi = x / length
theta0 = np.array([
    np.mean(np.log(modulus_true)[(xi >= 0.0) & (xi <= 1./3)]),
    np.mean(np.log(modulus_true)[(xi > 1./3) & (xi <= 2./3)]),
    np.mean(np.log(modulus_true)[(xi > 2./3) & (xi <= 1.0)]),
])

res = minimize(misfit, theta0, method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-6})
theta_best = res.x

y_best = B @ forward(theta_best)
y_ref = B @ forward(theta0)
residual_best = np.sqrt(np.mean((y_best - y_obs_true)**2))
residual_ref = np.sqrt(np.mean((y_ref - y_obs_true)**2))
max_obs = np.max(np.abs(y_obs_true))

print(f"theta0 (mean)  = {theta0}")
print(f"theta_best     = {theta_best}")
print(f"RMSE(theta0)   = {residual_ref:.4f}")
print(f"RMSE(best)     = {residual_best:.4f}")
print(f"max|y_obs|     = {max_obs:.4f}")
print(f"Relative model error (best) = {residual_best/max_obs*100:.1f}%")
print(f"Relative model error (mean) = {residual_ref/max_obs*100:.1f}%")
print(f"\nRecommended sigma_obs >= {residual_best:.4f} to account for model error")

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(9,4))
ax.plot(x, u_true, 'g--', lw=1.5, label='True displacement')
ax.plot(x, forward(theta_best), 'b-', lw=2, label=f'Best-fit piecewise (θ={np.round(theta_best,2)})')
ax.plot(x, forward(theta0), 'r:', lw=1.5, label=f'Mean-logE piecewise (θ_ref)')
ax.plot(x[obs_idx], y_obs_true, 'ko', ms=5, label='Observations (clean)')
ax.legend(); ax.set_xlabel('x'); ax.set_ylabel('u(x)')
ax.set_title('Best-fit piecewise vs true displacement')
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig('_model_error.png', dpi=150)
plt.show()
