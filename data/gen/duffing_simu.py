import numpy as np
from scipy.integrate import odeint
from matplotlib import pyplot as plt
delta = 0.5
beta = -1
alpha = 1

T = 100
t_values = np.linspace(0, 2.5, T)

num_trajectories = 2000
x0s = np.random.uniform(low=-2, high=2, size=(num_trajectories, 2))


def system(y, t, delta, beta, alpha):
    x, x_dot = y  # Unpack state variables
    dxdt = x_dot
    dx_dotdt = -delta * x_dot - x * (beta + alpha * x**2)
    return [dxdt, dx_dotdt]

# List to store all trajectories
trajectories = []

# Simulate and store each trajectory
for x0 in x0s:
    sol_odeint = odeint(system, x0, t_values, args=(delta, beta, alpha))
    trajectories.append(sol_odeint)

# Convert list to numpy array for easy saving
trajectories = np.array(trajectories)

# Save to a file
np.save('../simu_data/duffing.npy', trajectories)

# Plot all trajectories
plt.figure(figsize=(8, 6))
for trajectory in trajectories:
    plt.plot(trajectory[:, 0], trajectory[:, 1], alpha=0.5)
plt.xlabel("x")
plt.ylabel("x'")
plt.title("Phase Portrait of Trajectories")
plt.grid()
plt.show()

