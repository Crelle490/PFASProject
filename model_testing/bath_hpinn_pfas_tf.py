#!/usr/bin/env python

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import RNN, Layer
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay

# ---------------------------
# Set simulation grid time step (fine resolution)
# ---------------------------
dt_sim = 1.0  # seconds (fine-grained simulation step)

# ----------------------------------------
# Custom Runge–Kutta Integrator RNN Cell
# ----------------------------------------
class RungeKuttaIntegratorCell(Layer):
    def __init__(self, k1, k2, k3, k4, k5, k6, k7, c_eaq, dt, dummy_initial_state, **kwargs):
        """
        dummy_initial_state: a 1x8 numpy array used only for building the cell.
        (The actual initial states—one per sequence—will be provided at run time.)
        """
        super().__init__(**kwargs)
        self.c_eaq = c_eaq
        self.dt = dt
        self.initial_state = dummy_initial_state  # shape (1,8) used for build only
        self.state_size = 8
        # Trainable parameters in log10-space.
        self.log_k1, self.log_k2, self.log_k3, self.log_k4, self.log_k5, self.log_k6, self.log_k7 = [
            np.log10(k1), np.log10(k2), np.log10(k3), np.log10(k4),
            np.log10(k5), np.log10(k6), np.log10(k7)
        ]

    def build(self, input_shape):
        self.log_k_values = {
            name: self.add_weight(
                name=f'log_{name}', shape=(),
                initializer=tf.constant_initializer(value), trainable=True
            )
            for name, value in zip(
                ['k1','k2','k3','k4','k5','k6','k7'],
                [self.log_k1, self.log_k2, self.log_k3, self.log_k4,
                 self.log_k5, self.log_k6, self.log_k7]
            )
        }
        self.built = True

    def call(self, inputs, states):
        # Convert log parameters to actual values.
        params = {name: 10 ** log_value for name, log_value in self.log_k_values.items()}
        y = states[0]  # shape: (batch,8)
        k1 = self._fun(y, params) * self.dt
        k2 = self._fun(y + 0.5 * k1, params) * self.dt
        k3 = self._fun(y + 0.5 * k2, params) * self.dt
        k4 = self._fun(y + k3, params) * self.dt
        y_next = y + (k1 + 2*k2 + 2*k3 + k4) / 6.0

        # For output, select columns 0,2,4,5,6 (5 outputs).
        y1 = y_next[:, 0:1]
        y3 = y_next[:, 2:3]
        y5 = y_next[:, 4:5]
        y6 = y_next[:, 5:6]
        y7 = y_next[:, 6:7]
        output = tf.concat([y1, y3, y5, y6, y7], axis=-1)
        return output, [y_next]

    def _fun(self, y, params):
        y_vars = [y[:, i:i+1] for i in range(7)]
        rates = [params[f'k{i+1}'] * self.c_eaq * y_vars[i] for i in range(7)]
        dy1 = -rates[0]
        dy2 = rates[0] - rates[1]
        dy3 = rates[1] - rates[2]
        dy4 = rates[2] - rates[3]
        dy5 = rates[3] - rates[4]
        dy6 = rates[4] - rates[5]
        dy7 = rates[5] - rates[6]
        dy8 = 2 * tf.reduce_sum(rates, axis=0)
        return tf.concat([dy1, dy2, dy3, dy4, dy5, dy6, dy7, dy8], axis=-1)

# ----------------------------------------
# Interpolation Function
# ----------------------------------------
def interpolate_predictions(t_pinn, t_true, y_pred):
    """
    Interpolate y_pred (shape: 1 x num_steps x output_dim) at experimental time points t_true.
    t_pinn: 1D numpy array (fine simulation grid for one sequence).
    t_true: 1D numpy array (experimental time points for that sequence).
    Returns: Tensor of shape (1, len(t_true), output_dim).
    """
    t_pinn_tensor = tf.constant(t_pinn, dtype=tf.float32)
    t_true_tensor = tf.constant(t_true, dtype=tf.float32)
    indices = tf.searchsorted(t_pinn_tensor, t_true_tensor, side='left') - 1
    indices = tf.clip_by_value(indices, 0, tf.shape(t_pinn_tensor)[0] - 2)
    t0 = tf.gather(t_pinn_tensor, indices)
    t1 = tf.gather(t_pinn_tensor, indices + 1)
    y0 = tf.gather(y_pred, indices, axis=1)
    y1 = tf.gather(y_pred, indices + 1, axis=1)
    w = (t_true_tensor - t0) / (t1 - t0)
    w = tf.reshape(w, [1, -1, 1])
    return y0 + w * (y1 - y0)

# ----------------------------------------
# Custom Loss Function with Dynamic Weighting
# ----------------------------------------
@tf.autograph.experimental.do_not_convert
def create_loss_fn_multi(t_pinn_list, t_true_list):
    def my_loss_fn(y_true, y_pred):
        # y_true, y_pred: shape (batch, T_exp_max, output_dim)
        losses = []
        epsilon = 1e-8  # Prevent division by zero
        batch = len(t_true_list)  # Python list length
        for i in range(batch):
            # Use the unpadded experimental grid for sequence i
            valid_len = t_true_list[i].shape[0]
            y_true_i = y_true[i:i+1, :valid_len, :]
            # For simulation, use the fine-grained grid for sequence i.
            # We assume that the model output is produced on a padded fine-grained grid.
            y_pred_i = y_pred[i:i+1, :len(t_pinn_list[i]), :]
            # Interpolate prediction onto the experimental grid.
            y_pred_interp = interpolate_predictions(t_pinn_list[i], t_true_list[i], y_pred_i)
            y_pred_interp.set_shape(y_true_i.shape)
            # Dynamic weighting: compute per-output max from true data.
            y_max = tf.reduce_max(tf.abs(y_true_i), axis=[0,1]) + epsilon
            max_y_max = tf.reduce_max(y_max)
            weights = max_y_max / y_max
            coefficient = 1.0 / (max_y_max + epsilon)
            state_losses = tf.reduce_mean(tf.square(y_true_i - y_pred_interp), axis=[0,1])
            weighted_loss = tf.reduce_sum(weights * state_losses)
            scaled_loss = 10 * (coefficient ** 2) * weighted_loss
            losses.append(scaled_loss)
        return tf.reduce_mean(tf.stack(losses))
    return my_loss_fn

# ----------------------------------------
# Custom PINN Model (Accepting a Batch of Initial States)
# ----------------------------------------
class PINNModel(tf.keras.Model):
    def __init__(self, integrator_cell, num_steps):
        super(PINNModel, self).__init__()
        self.num_steps = num_steps
        self.rnn = RNN(integrator_cell, return_sequences=True)
    def call(self, inputs):
        # Expect inputs as a tuple: (dummy_input, initial_state)
        dummy_input, initial_state = inputs
        return self.rnn(dummy_input, initial_state=[initial_state])

# ----------------------------------------
# Main Training Script
# ----------------------------------------
if __name__ == "__main__":
    # --- Data Loading and Preparation ---
    # CSV is assumed to have columns: sequence_id, time (s), C7F15COO-, C5F11COO-, C3F7COO-, C2F5COO-, CF3COO-
    df = pd.read_csv('./data/Batch_PFAS_data.csv')
    groups = df.groupby("sequence_id")
    t_true_list = []       # experimental time grids (unpadded), one per sequence
    y_true_list = []       # experimental outputs (each: (T_exp, 5))
    initial_states_list = []  # each initial state: (8,)
    columns = ["C7F15COO-", "C5F11COO-", "C3F7COO-", "C2F5COO-", "CF3COO-"]

    for seq_id, group in groups:
        group_sorted = group.sort_values(by="time (s)")
        t_seq = group_sorted["time (s)"].values   # shape (T_exp,)
        y_seq = group_sorted[columns].values       # shape (T_exp, 5)
        t_true_list.append(t_seq)
        y_true_list.append(y_seq)
        init_state = np.zeros((8,), dtype=np.float32)
        init_state[0] = y_seq[0, 0]
        initial_states_list.append(init_state)

    batch_size = len(t_true_list)

    # --- Build Fine-Grained Simulation Grids ---
    # For each sequence, create a fine-grained simulation grid from its start to its end time using dt_sim.
    t_pinn_list = []
    T_sim_list = []

    for t_seq in t_true_list:
        # Create a fine grid from the first to the last experimental time
        t_sim = np.arange(t_seq[0], t_seq[-1] + dt_sim, dt_sim)
        t_pinn_list.append(t_sim)
        T_sim_list.append(len(t_sim))
    T_sim_max = max(T_sim_list)

    # --- Prepare Dummy Input for Simulation ---
    # Dummy input must have shape (batch, T_sim_max, 1)
    dummy_input = np.zeros((batch_size, T_sim_max, 1), dtype=np.float32)
    dummy_input = tf.convert_to_tensor(dummy_input, dtype=tf.float32)

    # --- Pad Experimental Data for Loss ---
    # We need to pad the experimental outputs to a common length T_exp_max.
    T_exp_list = [t.shape[0] for t in t_true_list]
    T_exp_max = max(T_exp_list)
    y_true_padded = []
    t_true_padded = []
    for t_seq, y_seq in zip(t_true_list, y_true_list):
        pad_len = T_exp_max - t_seq.shape[0]
        t_seq_pad = np.pad(t_seq, (0, pad_len), mode='constant', constant_values=0)
        y_seq_pad = np.pad(y_seq, ((0, pad_len), (0, 0)), mode='constant', constant_values=0)
        t_true_padded.append(t_seq_pad)
        y_true_padded.append(y_seq_pad)
    y_train = tf.convert_to_tensor(np.stack(y_true_padded, axis=0), dtype=tf.float32)

    # --- Prepare Initial States ---
    initial_states = tf.convert_to_tensor(np.stack(initial_states_list, axis=0), dtype=tf.float32)

    # --- Create tf.data.Dataset ---
    # Each element is ((dummy_input_i, initial_state_i), y_train_i)
    dataset = tf.data.Dataset.from_tensor_slices(((dummy_input, initial_states), y_train))
    # Set batch size to full dataset or a chosen batch size
    dataset = dataset.batch(batch_size)

    # --- Create Model ---
    k_values = [5.259223e+06, 5.175076e+08, 5.232472e+08,
                5.551598e+08, 5.748630e+08, 5.647780e+08, 5.005279e+08]
    k1, k2, k3, k4, k5, k6, k7 = [np.float32(v) for v in k_values]
    c_eaq = np.float32(3.69e-12)
    # Use the first sequence's initial state as dummy for building the cell.
    dummy_initial_state = initial_states[0:1].numpy()
    rk_cell = RungeKuttaIntegratorCell(k1, k2, k3, k4, k5, k6, k7, c_eaq, dt=1.0,
                                       dummy_initial_state=dummy_initial_state)
    # The RNN will unroll for T_sim_max steps (fine-grained simulation steps).
    model = PINNModel(rk_cell, num_steps=T_sim_max)

    # --- Loss and Optimizer ---
    loss_fn = create_loss_fn_multi(t_pinn_list, t_true_list)
    lr_schedule = PiecewiseConstantDecay(boundaries=[70, 150, 250],
                                         values=[5e-2, 1e-2, 1e-3, 1e-4])
    optimizer = RMSprop(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer, loss=loss_fn)

    # --- Training ---
    print("Before training, predictions:")
    y_pred_before = model.predict([dummy_input, initial_states])
    start_time = time.time()
    model.fit(dataset, epochs=200, verbose=1)
    end_time = time.time()
    print(f"Training Time: {end_time - start_time:.2f} seconds")
    y_pred = model.predict([dummy_input, initial_states])

    # --- Plotting for First Sequence ---
    # For sequence 0, extract its prediction (shape: (1, T_sim_max, 5)) and interpolate onto its experimental grid.
    # y_pred_first = y_pred[0:1, :, :]
    # y_pred_interp_first = interpolate_predictions(t_pinn_list[0], t_true_list[0], y_pred_first)
    # y_pred_interp_first = y_pred_interp_first.numpy().squeeze(0)
    #
    # plt.figure(figsize=(9, 6))
    # outputs_to_plot = [0, 1, 2, 3, 4]
    # labels = ['C7F15COO-', 'C5F11COO-', 'C3F7COO-', 'C2F5COO-', 'CF3COO-']
    # for i, label in enumerate(labels):
    #     plt.subplot(3, 2, i + 1)
    #     plt.scatter(t_true_list[0], y_true_list[0][:, i], color='gray', label='Raw Data', marker='o', alpha=0.7)
    #     plt.plot(t_true_list[0], y_pred_interp_first[:, i], 'b', label='After Training')
    #     plt.xlabel('Time (s)')
    #     plt.ylabel(label)
    #     plt.grid(True)
    #     plt.legend()
    # plt.tight_layout()
    # os.makedirs("Results", exist_ok=True)
    # plt.savefig("Results/HPINN_minimal_model_exp.png", dpi=600, bbox_inches='tight')
    # plt.show()

    t_sim_seq0 = t_pinn_list[1]  # This is your fine simulation grid for sequence 0

    # Extract raw fine-grained prediction for the first sequence.
    y_pred_first = y_pred[0:1, :, :]  # shape: (1, T_sim_max, 5)
    y_pred_first = y_pred_first.squeeze(0)  # shape: (T_sim_max, 5)

    plt.figure(figsize=(9, 6))
    outputs_to_plot = [0, 1, 2, 3, 4]
    labels = ['C7F15COO-', 'C5F11COO-', 'C3F7COO-', 'C2F5COO-', 'CF3COO-']

    for i, label in enumerate(labels):
        plt.subplot(3, 2, i + 1)
        plt.scatter(t_true_list[0], y_true_list[0][:, i], color='gray', label='Raw Data', marker='o', alpha=0.7)
        plt.plot(t_sim_seq0, y_pred_first[:, i], 'b', label='After Training')
        plt.xlabel('Time (s)')
        plt.ylabel(label)
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/HPINN_minimal_model_exp_batch1.png", dpi=600, bbox_inches='tight')
    plt.show()

    # --- Print Trained Parameters ---
    rk_cell_trained = model.rnn.cell  # Access the custom cell inside the RNN layer.
    trained_params = {name: 10 ** rk_cell_trained.log_k_values[name].numpy() for name in rk_cell_trained.log_k_values}
    print("Trained parameters:")
    for name, value in trained_params.items():
        print(f"{name}: {value:.6e}")
