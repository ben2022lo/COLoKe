import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from tqdm import tqdm

# Lorenz parameters
sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0

def lorenz(t, state):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return [dx, dy, dz]

# Simulation settings
t_span = (0, 5)
dt = 0.01
t_eval = np.arange(*t_span, dt)

# Number of trajectories and init range
n_trajectories = 2000
init_low, init_high = -10.0, 10.0

# Storage
trajectories = []

for _ in tqdm(range(n_trajectories), desc="Generating trajectories"):
    # Sample uniformly from cube [-20, 20]^3
    init_state = np.random.uniform(init_low, init_high, size=3)
    
    sol = solve_ivp(lorenz, t_span, init_state, t_eval=t_eval, method='RK45')
    traj = sol.y.T  # Shape: (T, 3)
    
    trajectories.append(traj)

trajs = np.stack(trajectories, axis = 0)

# Save to a file
np.save('../simu_data/lorenz.npy', trajs)

# plot a few trajectories
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(20):
    ax.plot(*trajectories[i].T, alpha=0.6)
ax.set_title("Sample Lorenz Trajectories")
plt.show()
