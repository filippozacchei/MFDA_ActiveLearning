"""Quick check of displacement values."""
import sys; sys.path.insert(0, '../../src')
import h5py, numpy as np

f = h5py.File('model/ProblemDefinition.h5', 'r')
x = np.array(f['/ForwardModel/NodeLocations']).ravel()
loads = np.array(f['/ForwardModel/Loads'])
modulus_true = np.array(f['/ForwardModel/Modulus'])
u_true = np.array(f['/ForwardModel/TrueDisplacement'])
radius = float(f['/ForwardModel'].attrs['BeamRadius'])
length = float(f['/ForwardModel'].attrs['BeamLength'])
f.close()

n = len(x); dx = length/(n-1); I = np.pi/4*radius**4
xi = x / length

theta_ref = np.array([
    np.mean(np.log(modulus_true)[(xi >= 0.0) & (xi <= 1./3)]),
    np.mean(np.log(modulus_true)[(xi > 1./3) & (xi <= 2./3)]),
    np.mean(np.log(modulus_true)[(xi > 2./3) & (xi <= 1.0)]),
])

logE = np.empty_like(x)
logE[(xi >= 0.0) & (xi <= 1./3)] = theta_ref[0]
logE[(xi > 1./3) & (xi <= 2./3)] = theta_ref[1]
logE[(xi > 2./3) & (xi <= 1.0)]  = theta_ref[2]
E_pw = np.exp(logE)

print(f"theta_ref = {theta_ref}")
print(f"I = {I:.6e}")
print(f"u_true range = [{u_true.min():.4f}, {u_true.max():.4f}]")

def build_K(modulus, n, dx):
    E = modulus; K = np.zeros((n,n))
    for i in range(2, n-2):
        K[i,i+2]=E[i]; K[i,i+1]=E[i+1]-6*E[i]+E[i-1]
        K[i,i]=-2*E[i+1]+10*E[i]-2*E[i-1]
        K[i,i-1]=E[i+1]-6*E[i]+E[i-1]; K[i,i-2]=E[i]
    K[1,3]=E[1]; K[1,2]=E[2]-6*E[1]+E[0]; K[1,1]=-2*E[2]+11*E[1]-2*E[0]
    K[n-2,n-1]=E[n-1]-4*E[n-2]+E[n-3]; K[n-2,n-2]=-2*E[n-1]+9*E[n-2]-2*E[n-3]
    K[n-2,n-3]=E[n-1]-6*E[n-2]+E[n-3]; K[n-2,n-4]=E[n-2]
    K[n-1,n-1]=2*E[n-1]; K[n-1,n-2]=-4*E[n-1]; K[n-1,n-3]=2*E[n-1]
    K[0,:]=0; K[:,0]=0; K[0,0]=1
    return K/dx**4

K = build_K(E_pw, n, dx)
rhs = loads/I; rhs[0] = 0.0
u_pw = np.linalg.solve(K, rhs)
print(f"u_pw (theta_ref) range = [{u_pw.min():.4f}, {u_pw.max():.4f}]")

# Observations
obs_idx = np.sort(np.array([1,2,3,5,6,8,9,11,12,14,15,16,17,18,19,22,23,24,26,30]))
B = np.zeros((len(obs_idx), n))
for j, i in enumerate(obs_idx): B[j,i] = 1.0

y_obs_pw = B @ u_pw
y_obs_true = B @ u_true
print(f"y_obs (piecewise) range = [{y_obs_pw.min():.4f}, {y_obs_pw.max():.4f}]")
print(f"y_obs (true E)    range = [{y_obs_true.min():.4f}, {y_obs_true.max():.4f}]")

# What does the surrogate see? The issue is the GP predicts y_obs
# and plot_prediction_at_theta plots surrogate.predict(theta) vs y_obs
# Let me check if scale is OK
print(f"\nSurrogate will map theta(3,) -> y_obs({len(obs_idx)},)")
print(f"y_obs values are order O(1), which is fine for GP")
