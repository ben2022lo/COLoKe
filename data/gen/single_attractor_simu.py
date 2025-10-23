import numpy as np
from scipy.integrate import odeint
from matplotlib import pyplot as plt
mu = -0.05
lambda_ = -1
t_values = np.linspace(0, 10, 101)

num_trajectories = 2000
x0s = np.random.uniform(low=-2, high=2, size=(num_trajectories, 2))


def system(y, t, mu, lambda_):
    x1, x2 = y  # Unpack state variables
    dx1dt = mu * x1
    dx2dt = lambda_ * (x2 - x1**2)
    return [dx1dt, dx2dt]

# List to store all trajectories
trajectories = []

# Simulate and store each trajectory
for x0 in x0s:
    sol_odeint = odeint(system, x0, t_values, args=(mu, lambda_))
    trajectories.append(sol_odeint)

# Convert list to numpy array for easy saving
trajectories = np.array(trajectories)

# Save to a file
np.save('../simu_data/single_attractor.npy', trajectories)

# Plot all trajectories
plt.figure(figsize=(8, 6))
for trajectory in trajectories:
    plt.plot(trajectory[:, 0], trajectory[:, 1], alpha=0.5)
plt.xlabel("x1")
plt.ylabel("x2'")
plt.title("Phase Portrait")
plt.grid()
plt.show()

