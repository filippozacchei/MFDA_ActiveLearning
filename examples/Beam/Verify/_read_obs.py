"""Extract obs indices from the observation matrix in ProblemDefinition.h5."""
import h5py, numpy as np

f = h5py.File('model/ProblemDefinition.h5', 'r')
B = np.array(f['/Observations/ObservationMatrix'])
obs_data = np.array(f['/Observations/ObservationData'])
true_disp = np.array(f['/ForwardModel/TrueDisplacement'])
modulus = np.array(f['/ForwardModel/Modulus'])
loads = np.array(f['/ForwardModel/Loads'])
x = np.array(f['/ForwardModel/NodeLocations']).ravel()

obs_idx = np.where(B == 1.0)[1]
print(f"obs_idx (sorted) = {np.sort(obs_idx)}")
print(f"obs_idx (original order) = {obs_idx}")
print(f"n_obs = {len(obs_idx)}")
print(f"n_nodes = {len(x)}")
print(f"log(Modulus) range = [{np.log(modulus).min():.3f}, {np.log(modulus).max():.3f}]")
print(f"log(Modulus) mean  = {np.log(modulus).mean():.3f}")
print(f"Modulus range = [{modulus.min():.1f}, {modulus.max():.1f}]")
print(f"true_disp range = [{true_disp.min():.6f}, {true_disp.max():.6f}]")
print(f"obs_data range = [{obs_data.min():.6f}, {obs_data.max():.6f}]")

# Check: B @ true_disp == obs_data?
reconstructed = B @ true_disp
print(f"\nmax |B @ u_true - obs_data| = {np.max(np.abs(reconstructed - obs_data)):.2e}")

f.close()
