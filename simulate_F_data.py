import numpy as np
import tensorflow as tf
import yaml
from model_testing.HPINN_Fused_model import RungeKuttaIntegratorCell, PINNModel, interpolate_predictions 
from tensorflow.keras.layers import RNN
import time
import matplotlib.pyplot as plt
from predictor.jacobian import Jacobian
from predictor.EKF import ExtendedKalmanFilter
from predictor.kinetic_model import f, h
import os
import pandas as pd

# Load parameters from config folder
with open("./config/physichal_paramters.yaml", "r") as file:
    params = yaml.safe_load(file)

with open("./config/initial_conditions.yaml", "r") as file:
    init_vals = yaml.safe_load(file)

with open("./config/trained_params.yaml", "r") as file:
    trained_reaction_rates = yaml.safe_load(file)

with open("./config/covariance_params.yaml", "r") as file:
    cov_params = yaml.safe_load(file)

# Assign initial conditions
pH = init_vals["pH"]
c_pfas_init = init_vals["c_pfas_init"]
c_cl = init_vals["c_cl"]
c_so3 = init_vals["c_so3"]

# Trained reaction rate constants (these could come from a file or be hard-coded)
k1, k2, k3, k4, k5, k6, k7 = [trained_reaction_rates[f'k{i}'] for i in range(1, 8)]
k = [k1, k2, k3, k4, k5, k6, k7]

# Set simulation resolution
dt_sim = 1.0

# Initial state for shape inference
# --- Data Loading and Preparation ---
df = pd.read_csv('./data/Batch_PFAS_data.csv')  # CSV columns: sequence_id, time (s), C7F15COO-, C5F11COO-, C3F7COO-, C2F5COO-, CF3COO-
groups = df.groupby("sequence_id")
columns = ["C7F15COO-", "C5F11COO-", "C3F7COO-", "C2F5COO-", "CF3COO-"]

initial_states_list = []  # each initial state: (8,)

for seq_id, group in groups:
    group_sorted = group.sort_values(by="time (s)")
    t_seq = group_sorted["time (s)"].values   # shape (T_exp,)
    y_seq = group_sorted[columns].values       # shape (T_exp, 5)
    init_state = np.zeros((8,), dtype=np.float32)
    init_state[0] = y_seq[0, 0]
    initial_states_list.append(init_state)

# Use only first component of initial state
initial_state = tf.convert_to_tensor(np.stack(initial_states_list[0], axis=0), dtype=tf.float32)
initial_state = tf.expand_dims(initial_state,axis=0)

# Prediction horeizon defined as number of steps to simulat response. Total prediction time would be N_sim*dt_sim
N_sim = 3601

# Build integrator cell and model
rk_cell = RungeKuttaIntegratorCell(k1, k2, k3, k4, k5, k6, k7, params, c_cl, c_so3, pH, dt_sim, initial_state,for_prediction=True)
model = PINNModel(rk_cell, num_steps=N_sim)  # Adjust num_steps based on desired simulation time

# Construct time input for model
simulation_time = np.arange(N_sim) * dt_sim
simulation_time = simulation_time.reshape(1, -1, 1)
dummy_input = tf.convert_to_tensor(simulation_time, dtype=tf.float32)

# Build model (required before loading weights)
_ = model([dummy_input, initial_state])

# Load trained weights
model.load_weights("./checkpoints/pinn_model.weights.h5")

# Predict
y_pred = model.predict([dummy_input, initial_state])

# Save only the F- results
print(np.shape(y_pred))
results = y_pred[0,:,-1]

base_dir = os.getcwd()  # current working directory
file_path = os.path.join(base_dir, "data", "simulated_F_concentraction.csv")

# Create the output directory if it doesn't exist
os.makedirs(os.path.dirname(file_path), exist_ok=True)

# Save the array
np.savetxt(file_path, results, delimiter=",", fmt="%.10f")