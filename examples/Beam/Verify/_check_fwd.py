"""Quick check: does the forward model produce meaningful displacement?"""
import sys; sys.path.insert(0, '../../src')
import h5py, numpy as np
from pathlib import Path

f = h5py.File('model/ProblemDefinition.h5', 'r')
x = np.array(f['/ForwardModel/NodeLocations']).ravel()
loads = np.array(f['/ForwardModel/Loads'])
modulus_true = np.array(f['/ForwardModel/Modulus'])
u_true = np.array(f['/ForwardModel/TrueDisplacement'])
radius = float(f['/ForwardModel'].attrs['BeamRadius'])
f.close()

print(f"u_true range: [{u_true.min():.4f}, {u_true.max():.4f}]")
print(f"loads range: [{loads.min():.2f}, {loads.max():.2f}]")
print(f"loads are POSITIVE")

# Test with piecewise theta
xi = x / 1.0
theta_ref = np.array([
    np.mean(np.log(modulus_true)[(xi >= 0.0) & (xi <= 1./3)]),
    np.mean(np.log(modulus_true)[(xi > 1./3) & (xi <= 2./3)]),
    np.mean(np.log(modulus_true)[(xi > 2./3) & (xi <= 1.0)]),
])
print(f"theta_ref = {theta_ref}")

# Import the forward model
exec(open('backward_muq.py').read().split('rng = set_seed')[0])
# Manually test
n = len(x); dx = 1.0/(n-1); I = np.pi/4*radius**4
logE = np.empty_like(x)
logE[(xi >= 0.0) & (xi <= 1./3)] = theta_ref[0]
logE[(xi > 1./3) & (xi <= 2./3)] = theta_ref[1]
logE[(xi > 2./3) & (xi <= 1.0)]  = theta_ref[2]
E = np.exp(logE)

print(f"\nI = {I:.6e}")
print(f"E range = [{E.min():.1f}, {E.max():.1f}]")
print(f"loads/I range = [{(loads/I).min():.1f}, {(loads/I).max():.1f}]")
print(f"loads/I[0] set to 0 for BC")

# Full check
from backward_muq import beam_forward_muq, _build_stiffness_matrix
u_pw = beam_forward_muq(theta_ref, x, loads, radius)
print(f"\nu_pw (piecewise theta_ref):")
print(f"  range = [{u_pw.min():.4f}, {u_pw.max():.4f}]")
print(f"  u_pw[:5] = {u_pw[:5]}")
print(f"  u_pw[-5:] = {u_pw[-5:]}")

# What does hf_forward return (B @ u)?
obs_idx = np.sort(np.array([1,2,3,5,6,8,9,11,12,14,15,16,17,18,19,22,23,24,26,30]))
B = np.zeros((len(obs_idx), n))
for j, i in enumerate(obs_idx):
    B[j,i] = 1.0

y_obs_pw = B @ u_pw
print(f"\ny_obs (B @ u_pw) range = [{y_obs_pw.min():.4f}, {y_obs_pw.max():.4f}]")
