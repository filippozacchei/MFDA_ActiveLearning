"""Verify that a pure-numpy reimplementation of BeamModel.py matches the HDF5 truth."""
import sys; sys.path.insert(0, '../../src')
import h5py, numpy as np

f = h5py.File('model/ProblemDefinition.h5', 'r')
x = np.array(f['/ForwardModel/NodeLocations']).ravel()
loads = np.array(f['/ForwardModel/Loads'])
modulus = np.array(f['/ForwardModel/Modulus'])
u_true = np.array(f['/ForwardModel/TrueDisplacement'])
length = float(f['/ForwardModel'].attrs['BeamLength'])
radius = float(f['/ForwardModel'].attrs['BeamRadius'])
f.close()

n = len(x)
dx = length / (n - 1)
I = np.pi / 4.0 * radius**4

def build_K_muq(modulus, n, dx):
    """Replicate BeamModel.py's BuildK stencil."""
    K = np.zeros((n, n))
    E = modulus
    for i in range(2, n-2):
        K[i, i+2] = E[i]
        K[i, i+1] = E[i+1] - 6.0*E[i] + E[i-1]
        K[i, i]   = -2.0*E[i+1] + 10.0*E[i] - 2.0*E[i-1]
        K[i, i-1] = E[i+1] - 6.0*E[i] + E[i-1]
        K[i, i-2] = E[i]

    # row 1
    K[1, 3] = E[1]
    K[1, 2] = E[2] - 6.0*E[1] + E[0]
    K[1, 1] = -2.0*E[2] + 11.0*E[1] - 2.0*E[0]

    # row n-2
    K[n-2, n-1] = E[n-1] - 4.0*E[n-2] + E[n-3]
    K[n-2, n-2] = -2.0*E[n-1] + 9.0*E[n-2] - 2.0*E[n-3]
    K[n-2, n-3] = E[n-1] - 6.0*E[n-2] + E[n-3]
    K[n-2, n-4] = E[n-2]

    # row n-1
    K[n-1, n-1] = 2.0*E[n-1]
    K[n-1, n-2] = -4.0*E[n-1]
    K[n-1, n-3] = 2.0*E[n-1]

    # dirichlet u(0)=0
    K[0, :] = 0.0; K[:, 0] = 0.0; K[0, 0] = 1.0

    return K / dx**4

K = build_K_muq(modulus, n, dx)
rhs = loads / I
rhs[0] = 0.0
u_recon = np.linalg.solve(K, rhs)

print(f"max |u_recon - u_true| = {np.max(np.abs(u_recon - u_true)):.2e}")
print(f"u_true[:5]  = {u_true[:5]}")
print(f"u_recon[:5] = {u_recon[:5]}")

# Now test with piecewise-constant modulus (3 parameters)
theta_test = np.array([9.336, 9.336, 9.336])  # ~mean of log(Modulus)
xi = x / length
logE = np.empty_like(x)
logE[(xi >= 0.0) & (xi <= 1./3)] = theta_test[0]
logE[(xi > 1./3) & (xi <= 2./3)] = theta_test[1]
logE[(xi > 2./3) & (xi <= 1.0)]  = theta_test[2]
E_pw = np.exp(logE)

K2 = build_K_muq(E_pw, n, dx)
rhs2 = loads / I
rhs2[0] = 0.0
u_pw = np.linalg.solve(K2, rhs2)
print(f"\nPiecewise-constant test:")
print(f"u_pw range = [{u_pw.min():.4f}, {u_pw.max():.4f}]")
print(f"u_true range = [{u_true.min():.4f}, {u_true.max():.4f}]")
