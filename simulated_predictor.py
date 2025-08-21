import numpy as np
import matplotlib.pyplot as plt
from predictor.kinetic_model import f 
from predictor.HPINN_predictor import HPINNPredictor
import pandas as pd

# --- Setup predictor and data ---
dt = 1
frequency = 0.01
N = int(1/frequency)
predictor = HPINNPredictor(dt=dt, sensor_frequency=frequency)

simulated_sensor_data = pd.read_csv('./data/simulated_F_concentraction.csv')
data_array = simulated_sensor_data.values[:]

mean, std_dev = 0, 2.0e-8 # Standard deviation from data sheet
noise = np.random.normal(mean, std_dev, size=data_array.shape)
noisy_data = data_array + noise # + noise if you want
sensor_data_for_simulation = noisy_data[N+1::N+1]

y_preds, t_inputs = [], []
for sensor_data in sensor_data_for_simulation:
    predictor.get_sensor_measuerment(z=sensor_data)
    t_input = predictor.simulation_time
    PINN_prediction, concentractions, sensor_measuerment = predictor.step()
    y_pred_reshaped = PINN_prediction.reshape(N, 8)
    t_input_reshaped = t_input.numpy().reshape(N, 1)
    y_preds.append(y_pred_reshaped)
    t_inputs.append(t_input_reshaped)

y_pred = np.array(y_preds).reshape(len(sensor_data_for_simulation)*N, 8)
t_input = np.array(t_inputs).reshape(len(sensor_data_for_simulation)*N, 1)

# --- Raw forward integration with f ---
"""
x = predictor.x0.copy()
states = [x.copy()]
for i in range(len(y_pred)):
    x = f(x, 0, predictor.k)
    states.append(x.copy())
states = np.array(states)   # shape: (len(y_pred)+1, 9)
steps = np.arange(len(states))
"""
dt = 1
N = 3601
frequency = 1/N  # => period = 1/freq = 1000 s, so N = 1000
N = int(1/frequency)  # 1000

# Build predictor (this sets predictor.N_sim == N)
predictor = HPINNPredictor(dt=dt, sensor_frequency=frequency)

# Predict with the PINN on its internal time grid
estimation_input = [predictor.simulation_time, predictor.initial_state]
y_pred_pure_PINN = predictor.model.predict(estimation_input, verbose=0)[0]  # shape: (N, 8)
t_pure_PINN = predictor.simulation_time.numpy().reshape(-1)                 # shape: (N,)

# --- Plot combined results ---
labels = [
    'C7F15COO-','C6F13COO-',
    'C5F11COO-','C4F9COO-','C3F7COO-',
    'C2F5COO-','CF3COO-','F-'
]

fig, axes = plt.subplots(4, 2, figsize=(14, 12), sharex=True)
axes = axes.flatten()

for i, label in enumerate(labels):
    axes[i].plot(t_input, y_pred[:, i], 'r', label='PINN and EKF prediction')
    axes[i].plot(t_pure_PINN, y_pred_pure_PINN[:, i], 'k', label='PINN prediction')
    #axes[i].plot(steps, states[:, i+1], 'b--', label='Raw integration')  
    # note: states[:,0] is e_aq, so shift by +1
    
    # 👇 Add simulated F- concentration only for F- subplot
    if label == "F-":
        sim_F = simulated_sensor_data.values.flatten()[:len(t_input)]
        axes[i].plot(t_input, sim_F, 'g-', alpha=0.6, label='Simulated F-')

        noisy_F = noisy_data.flatten()[:len(t_input)]
        axes[i].scatter(t_input, noisy_F, color='b', s=10, alpha=0.5, label='Noisy sensor')

    axes[i].set_title(label)
    axes[i].set_ylabel("Concentration")
    axes[i].grid(True)
    axes[i].legend()

axes[-1].set_xlabel("Time (s)")
plt.tight_layout()
plt.show()
