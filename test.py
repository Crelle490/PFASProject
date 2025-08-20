
from HPINN_predictor import HPINNPredictor
import pandas as pd
import numpy as np
from predictor.kinetic_model import f 
import matplotlib.pyplot as plt
import tensorflow as tf




# Prediction horeizon defined as number of steps to simulat response. Total prediction time would be N_sim*dt_sim
dt = 1
frequency = 0.001
N = int(1/frequency)
n = 1000

# Setup predictor
predictor = HPINNPredictor(dt=dt,sensor_frequency=frequency)

simulation_time = np.arange(N)
simulation_time = simulation_time.reshape(1, -1, 1)
simulation_time = tf.convert_to_tensor(simulation_time, dtype=tf.float32)

estimation_input = [predictor.simulation_time, predictor.initial_state]
y_pred = predictor.model.predict(estimation_input)

y_pred = y_pred[0]  # drop batch dim, now (n, 8)

# Labels for the 8 PINN states
labels = [
    'C7F15COO-', 'C6F13COO-', 'F-',
    'C5F11COO-', 'C4F9COO-', 'C3F7COO-',
    'C2F5COO-', 'CF3COO-'
]

# Plot in 4x2 subplots
fig, axes = plt.subplots(4, 2, figsize=(14, 12), sharex=True)
axes = axes.flatten()

for i in range(8):
    axes[i].plot(simulation_time.numpy().flatten(), y_pred[:, i], label=labels[i])
    axes[i].set_title(labels[i])
    axes[i].set_ylabel("Concentration")
    axes[i].grid(True)
    axes[i].legend()

axes[-1].set_xlabel("Time (s)")
plt.tight_layout()


x = predictor.x0.copy()
states = [x.copy()]
for i in range(1000):
    x = f(x, 0, predictor.k)
    states.append(x.copy())

states = np.array(states)  # shape: (10001, 9)
steps = np.arange(len(states))  # [0, 1, 2, ..., 10000]

labels = [
    "e_aq", "C7F15COO-", "C6F13COO-", "F-",
    "C5F11COO-", "C4F9COO-", "C3F7COO-",
    "C2F5COO-", "CF3COO-"
]

fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharex=True)
axes = axes.flatten()

for i in range(9):
    axes[i].plot(steps, states[:, i])
    axes[i].set_title(labels[i])
    axes[i].set_ylabel("Concentration")
    axes[i].grid(True)

axes[-1].set_xlabel("Step index")
fig.tight_layout()
plt.show()