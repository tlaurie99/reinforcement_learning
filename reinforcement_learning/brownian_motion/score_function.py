import numpy as np
import matplotlib.pyplot as plt
import math

# Simulation parameters
T_final = 10      # Total time for the reverse process
dt = 0.1           # Time step
n_steps = int(T_final/dt) + 1
times = np.linspace(0, T_final, n_steps)
a0 = 1.0           # Original policy action (normalized)

# Forward process parameters at time T_final:
# a_T ~ N(e^{-T_final}*a0, 1-e^{-2T_final})
mean_T = math.exp(-T_final) * a0
var_T = 1 - math.exp(-2*T_final)

# Initialize the reverse process at t=0 with a sample from the final forward distribution.
a_reverse = np.zeros(n_steps)
a_reverse[0] = np.random.normal(mean_T, np.sqrt(var_T))

# Define the score function for a Gaussian density
def score(t, a):
    # In the reverse process, t runs from 0 to T_final, but the corresponding
    # forward process time is s = T_final - t.
    s = T_final - t  
    mean_s = math.exp(-s) * a0
    var_s = 1 - math.exp(-2*s)
    return -(a - mean_s) / var_s

# Simulate the reverse process with noise
for i in range(1, n_steps):
    t_current = times[i-1]
    drift = a_reverse[i-1] + 2 * score(t_current, a_reverse[i-1])
    dW = np.random.normal(0, np.sqrt(dt))
    a_reverse[i] = a_reverse[i-1] + drift * dt + np.sqrt(2) * dW

# For comparison, simulate a deterministic reverse trajectory (no noise)
a_det = np.zeros(n_steps)
a_det[0] = a_reverse[0]
for i in range(1, n_steps):
    t_current = times[i-1]
    drift_det = a_det[i-1] + 2 * score(t_current, a_det[i-1])
    a_det[i] = a_det[i-1] + drift_det * dt

# Plot the results
plt.figure(figsize=(10,6))
plt.plot(times, a_reverse, label="Reverse Process Sample Path", marker='o', linestyle='--', alpha=0.8)
plt.plot(times, a_det, label="Deterministic Reverse Trajectory", linewidth=2, color='black')
plt.xlabel("Time")
plt.ylabel("Action Value")
plt.title("Reverse Process: Transforming Noise to Policy Action")
plt.legend()
plt.grid(True)
plt.show()
