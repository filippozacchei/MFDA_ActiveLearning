"""Diagnostic: measure model discrepancy vs sigma_obs."""
import sys; sys.path.insert(0, '../../src')
import h5py, numpy as np
from pathlib import Path

h5_path = Path(__file__).with_name("model") / "ProblemDefinition.h5"
with h5py.File(h5_path, "r") as f:
    x = np.array(f["/ForwardModel/NodeLocations"]).ravel()
    loads = np.array(f["/ForwardModel/Loads"]).ravel()
    modulus_true = np.array(f["/ForwardModel/Modulus"]).ravel()
    u_true_full = np.array(f["/ForwardModel/TrueDisplacement"]).ravel()
    radius = float(f["/ForwardModel"].attrs["BeamRadius"])
    B_obs = np.array(f["/Observations/ObservationMatrix"])

print("loads.shape:", loads.shape)
print("u_true_full.shape:", u_true_full.shape)
print("modulus_true.shape:", modulus_true.shape)

xi = x / 1.0
theta_ref = np.array([
    np.mean(np.log(modulus_true)[(xi >= 0.) & (xi <= 1./3)]),
    np.mean(np.log(modulus_true)[(xi > 1./3) & (xi <= 2./3)]),
    np.mean(np.log(modulus_true)[(xi > 2./3) & (xi <= 1.0)]),
])
print("theta_ref =", theta_ref)

n = len(x); length = 1.0; dx = length/(n-1)
I = np.pi/4.0*radius**4

n_intervals = 3
endPts = np.linspace(0, length, n_intervals + 1)
A_pw = np.zeros((n, n_intervals))
for i in range(n_intervals):
    A_pw[(x >= endPts[i]) & (x <= endPts[i + 1]), i] = 1.0
E_pw = A_pw @ np.exp(theta_ref)

def build_K(modulus, dx, n):
    K = np.zeros((n, n))
    for i in range(2, n - 2):
        K[i, i + 2] = modulus[i]
        K[i, i + 1] = modulus[i + 1] - 6.0 * modulus[i] + modulus[i - 1]
        K[i, i]     = -2.0 * modulus[i + 1] + 10.0 * modulus[i] - 2.0 * modulus[i - 1]
        K[i, i - 1] = modulus[i + 1] - 6.0 * modulus[i] + modulus[i - 1]
        K[i, i - 2] = modulus[i]
    K[1, 3] = modulus[1]
    K[1, 2] = modulus[2] - 6.0 * modulus[1] + modulus[0]
    K[1, 1] = -2.0 * modulus[2] + 11.0 * modulus[1] - 2.0 * modulus[0]
    K[n-2, n-1] = modulus[n-1] - 4.0*modulus[n-2] + modulus[n-3]
    K[n-2, n-2] = -2.0*modulus[n-1] + 9.0*modulus[n-2] - 2.0*modulus[n-3]
    K[n-2, n-3] = modulus[n-1] - 6.0*modulus[n-2] + modulus[n-3]
    K[n-2, n-4] = modulus[n-2]
    K[n-1, n-1] = 2.0*modulus[n-1]
    K[n-1, n-2] = -4.0*modulus[n-1]
    K[n-1, n-3] = 2.0*modulus[n-1]
    K[0, :] = 0.0; K[:, 0] = 0.0; K[0, 0] = 1.0
    return K / dx**4

K = build_K(E_pw, dx, n)
rhs = loads / I
rhs[0] = 0.0
u_pw = np.linalg.solve(K, rhs)

obs_idx = np.sort(np.where(B_obs == 1.0)[1])
B = np.zeros((len(obs_idx), n))
for j, i in enumerate(obs_idx):
    B[j, i] = 1.0

y_obs = B @ u_true_full
y_pw = B @ u_pw

residual = y_obs - y_pw
sigma_obs = 0.01

print()
print("y_obs range:", y_obs.min(), y_obs.max())
print("y_pw  range:", y_pw.min(), y_pw.max())
print()
print("residual (y_obs - y_pw):")
print("  max abs:", np.abs(residual).max())
print("  rms:    ", np.sqrt(np.mean(residual**2)))
print("  sigma_obs:", sigma_obs)
print()
print("RATIO rms_residual / sigma_obs:", np.sqrt(np.mean(residual**2)) / sigma_obs)
print("=> If ratio >> 1, the model discrepancy dominates the noise!")
print()
print("Suggested sigma_obs (5x RMS residual):", 5.0 * np.sqrt(np.mean(residual**2)))
