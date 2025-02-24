import numpy as np
import matplotlib.pyplot as plt
import math

# Set parameters for normalized actions
a_initial = 1.0     # normalized initial action
T_final = 10        # final time
sigma = np.sqrt(2)        # noise scale

# Define a function to simulate the OU process given a time step dt
def simulate_ou(dt, a0, T, sigma):
    n_steps = int(T / dt) + 1  # include t=0
    times = np.linspace(0, T, n_steps)
    trace = np.zeros(n_steps)
    trace[0] = a0
    for i in range(1, n_steps):
        # Wiener increment: Normal with mean 0 and variance dt
        dW = np.random.normal(0, np.sqrt(dt))
        # Euler discretization: a[t+1] = a[t] + (-a[t]*dt) + sigma*dW
        trace[i] = trace[i-1] + (-trace[i-1] * dt) + sigma * dW
    return times, trace

# Simulate for dt=0.001
dt1 = 0.001
times1, ou_trace_dt1 = simulate_ou(dt1, a_initial, T_final, sigma)

# Simulate for dt=0.1
dt2 = 0.1
times2, ou_trace_dt2 = simulate_ou(dt2, a_initial, T_final, sigma)

# Compute the deterministic exponential decay trace: a(t) = exp(-t) * a_initial
# We use a common set of times for plotting the deterministic curve.
deterministic_times = np.linspace(0, T_final, 300)
deterministic_trace = np.exp(-deterministic_times) * a_initial

# Plot the results for comparison
plt.figure(figsize=(12, 6))
plt.plot(deterministic_times, deterministic_trace, label='Deterministic Decay', linewidth=3, color='black')
plt.plot(times1, ou_trace_dt1, label=f'OU Process (dt={dt1})', linestyle='--', alpha=0.8)
plt.plot(times2, ou_trace_dt2, label=f'OU Process (dt={dt2})', linestyle='--', alpha=0.8)

plt.xlabel('Time')
plt.ylabel('Action Value')
plt.title('Comparison of OU Process Sample Paths with Different dt Values')
plt.legend()
plt.grid(True)

# Save the figure
plt.savefig('OU_process_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
