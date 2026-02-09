# examples/navier-stokes/cfd/run_lf.py
import matplotlib.pyplot as plt
from utils.types import BFSGeometry
from utils.lf_potential import solve_potential_bfs

heights = [0.05, 0.08, 0.10, 0.12, 0.15]
U_in = 1.5

profiles = []
for h in heights:
    prof = solve_potential_bfs(geom=BFSGeometry(h1=h), U_in=U_in)
    profiles.append((h, prof))

plt.figure()
for h, prof in profiles:
    plt.plot(prof.u_x, prof.y, label=f"h2={h:.2f}")
plt.xlabel("uₓ (outlet)")
plt.ylabel("y")
plt.legend()
plt.tight_layout()
plt.show()
