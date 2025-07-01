import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import RNN, Layer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.constraints import NonNeg
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay

class RungeKuttaIntegratorCell(Layer):
    def __init__(self, k1, k2, k3, k4, k5, k6, k7, constants, c_cl, c_so3, pH, dt, initial_state, **kwargs):
        super().__init__(**kwargs)
        self.constants = constants
        self.c_cl = c_cl
        self.c_so3 = c_so3
        self.pH = pH
        self.dt = dt
        self.initial_state = initial_state
        self.state_size = 8

        # Trainable physical parameters
        # self._k1, self._k2, self._k3, self._k4, self._k5, self._k6, self._k7 = k1, k2, k3, k4, k5, k6, k7
        self.log_k1, self.log_k2, self.log_k3, self.log_k4, self.log_k5, self.log_k6, self.log_k7 = [
            np.log10(k1), np.log10(k2), np.log10(k3), np.log10(k4), np.log10(k5), np.log10(k6), np.log10(k7)
        ]

    def build(self, input_shape):
        self.log_k_values = {
            name: self.add_weight(
                name=f'log_{name}', shape=(),
                initializer=tf.constant_initializer(value), trainable=True
            )
            for name, value in zip(
                ['k1', 'k2', 'k3', 'k4', 'k5', 'k6', 'k7'],
                [self.log_k1, self.log_k2, self.log_k3, self.log_k4, self.log_k5, self.log_k6, self.log_k7]
            )
        }
        self.built = True

    def call(self, inputs, states):
        params = {name: 10 ** log_value for name, log_value in self.log_k_values.items()}

        y = states[0]
        k1 = self._fun(y, params) * self.dt
        k2 = self._fun(y + 0.5 * k1, params) * self.dt
        k3 = self._fun(y + 0.5 * k2, params) * self.dt
        k4 = self._fun(y + k3, params) * self.dt
        y_next = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

        # Compute additional output
        y_1 = y_next[:, 0:1]
        y_3 = y_next[:, 2:3]
        y_5 = y_next[:, 4:5]
        y_6 = y_next[:, 5:6]
        y_7 = y_next[:, 6:7]
        y_8 = y_next[:, 7:8]

        # Concatenate the outputs
        output = tf.concat([y_1, y_3, y_5, y_6, y_7], axis=-1)
        return output, [y_next]

    def _fun(self, y, params):
        y_vars = [y[:, i:i + 1] for i in range(7)]
        k_vars = [params[f'k{i + 1}'] for i in range(7)]

        numerator = self.generation_of_eaq()

        k_so3_eaq = 1.5e6
        k_cl_eaq = 1e6
        beta_j = 2.57e4

        denominator = params['k1'] * c_pfas_init + beta_j + k_so3_eaq * c_so3 + k_cl_eaq * c_cl

        c_eaq = numerator / denominator

        rates = [k * c_eaq * y_var for k, y_var in zip(k_vars, y_vars)]

        dy1 = -rates[0]
        dy2 = rates[0] - rates[1]
        dy3 = rates[1] - rates[2]
        dy4 = rates[2] - rates[3]
        dy5 = rates[3] - rates[4]
        dy6 = rates[4] - rates[5]
        dy7 = rates[5] - rates[6]
        dy8 = 2 * tf.reduce_sum(rates, axis=0, keepdims=False)  # 保持形状为 [1, 1]

        return tf.concat([dy1, dy2, dy3, dy4, dy5, dy6, dy7, dy8], axis=-1)

    def generation_of_eaq(self):
        params = self.constants
        c_pfas_init = self.initial_state[0,0]
        # Compute hydroxide concentration (c_oh_m) from the given pH (usually a scalar)
        c_oh_m = np.power(10, -14.0 + self.pH)

        # Compute total absorption (Sigma_f) for 185 nm
        Sigma_f_185 = (
                params["epsilon_h2o_185"] * params["c_h2o"]
                + params["epsilon_oh_m_185"] * c_oh_m
                + params["epsilon_cl_185"] * self.c_cl
                + params["epsilon_so3_185"] * self.c_so3
                + params["epsilon_pfas_185"] * c_pfas_init
        )

        # Compute total absorption (Sigma_f) for 254 nm
        Sigma_f_254 = (
                params["epsilon_h2o_254"] * params["c_h2o"]
                + params["epsilon_so3_254"] * self.c_so3
                + params["epsilon_pfas_254"] * c_pfas_init
        )

        # Calculate the fraction of absorption for each species at 185 nm
        f_h2o_185 = (params["epsilon_h2o_185"] * params["c_h2o"]) / Sigma_f_185
        f_oh_m_185 = (params["epsilon_oh_m_185"] * c_oh_m) / Sigma_f_185
        f_cl_185 = (params["epsilon_cl_185"] * self.c_cl) / Sigma_f_185
        f_so3_185 = (params["epsilon_so3_185"] * self.c_so3) / Sigma_f_185

        # For 254 nm, calculate the fraction for sulfite
        f_so3_254 = (params["epsilon_so3_254"] * self.c_so3) / Sigma_f_254

        # Calculate contributions to eaq⁻ generation at 185 nm
        term_h2o_185 = f_h2o_185 * params["phi_h2o_185"] * (
                1 - np.power(10, -params["epsilon_h2o_185"] * params["l"] * params["c_h2o"]))
        term_oh_m_185 = f_oh_m_185 * params["phi_oh_m_185"] * (
                1 - np.power(10, -params["epsilon_oh_m_185"] * params["l"] * c_oh_m))
        term_cl_185 = f_cl_185 * params["phi_cl_185"] * (
                1 - np.power(10, -params["epsilon_cl_185"] * params["l"] * self.c_cl))
        term_so3_185 = f_so3_185 * params["phi_so3_185"] * (
                1 - np.power(10, -params["epsilon_so3_185"] * params["l"] * self.c_so3))

        numerator_185 = params["I0_185"] * (term_h2o_185 + term_oh_m_185 + term_cl_185 + term_so3_185)

        # Calculate contribution to eaq⁻ generation at 254 nm
        numerator_254 = params["I0_254"] * f_so3_254 * params["phi_so3_254"] * (
                1 - np.power(10, -params["epsilon_so3_254"] * params["l"] * self.c_so3))

        # Total eaq⁻ generation
        generation_eaq = numerator_185 + numerator_254
        return generation_eaq

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return self.initial_state

def interpolate_predictions(t_pinn, t_true, y_pred):
    """Interpolate y_pred at t_true points using t_pinn."""
    t_pinn_tensor = tf.constant(t_pinn, dtype=tf.float32)
    t_true_tensor = tf.constant(t_true, dtype=tf.float32)

    # find t_true at t_pinn
    indices = tf.searchsorted(t_pinn_tensor, t_true_tensor, side='left') - 1
    indices = tf.clip_by_value(indices, 0, len(t_pinn_tensor) - 2)  # prevent beyond the index

    # find the end point of section for interpolation
    t0 = tf.gather(t_pinn_tensor, indices)
    t1 = tf.gather(t_pinn_tensor, indices + 1)
    y0 = tf.gather(y_pred, indices, axis=1)      # value at left hand side
    y1 = tf.gather(y_pred, indices + 1, axis=1)  # value at right hand side

    # compute weights for linear interpolation
    w = (t_true_tensor - t0) / (t1 - t0)
    w = tf.reshape(w, [1, -1, 1])  # reshape for matching y0 and y1

    # interpolation
    y_pred_interp = y0 + w * (y1 - y0)
    return y_pred_interp

def create_loss_fn(t_pinn, t_true):
    def my_loss_fn(y_true, y_pred):
        """Custom loss function with dynamic scaling and weighting."""
        # epsilon = 1e-8  # Small constant to prevent division by zero

        # Interpolate y_pred to t_true time points
        y_pred_interp = interpolate_predictions(t_pinn, t_true, y_pred)
        y_pred_interp.set_shape(y_true.shape)

        # Step 1: Compute maximum absolute values for each state over batch and time
        y_max = tf.reduce_max(tf.abs(y_true), axis=[0, 1]) # + epsilon  # Shape: (num_outputs,)

        # Step 2: Compute weights based on the maximum concentration state
        max_y_max = tf.reduce_max(y_max)  # Find the maximum concentration across all states
        weights = max_y_max / y_max       # Larger states get smaller weights

        # Step 3: Compute scaling coefficient to normalize loss to around 1
        coefficient = 1.0 / max_y_max     # Scaling coefficient

        # Step 4: Compute per-state losses
        state_losses = tf.reduce_mean(tf.square(y_true - y_pred_interp), axis=[0, 1])  # Shape: (num_outputs,)

        # Step 5: Compute weighted loss
        weighted_loss = tf.reduce_sum(weights * state_losses)  # Sum weighted losses across states

        # Step 6: Apply scaling coefficient
        scaled_loss = 10 * coefficient ** 2 * weighted_loss

        return scaled_loss

    return my_loss_fn

def create_model(k1, k2, k3, k4, k5, k6, k7,  constants, c_cl, c_so3, pH, dt, initial_state, batch_input_shape, t_pinn, t_true):
    rk_cell = RungeKuttaIntegratorCell(k1, k2, k3, k4, k5, k6, k7, constants, c_cl, c_so3, pH, dt, initial_state)
    rnn_layer = RNN(cell=rk_cell, batch_input_shape=batch_input_shape, return_sequences=True)
    model = Sequential([rnn_layer])
    loss_fn = create_loss_fn(t_pinn, t_true)
    lr_schedule = PiecewiseConstantDecay(boundaries=[70, 150, 250], values=[5e-2, 1e-2, 1e-3, 1e-4])
    model.compile(optimizer=RMSprop(learning_rate=lr_schedule), loss=loss_fn)
    return model

if __name__ == "__main__":
# def generate_eaq(params, c_pfas_init, c_cl, c_so3, pH):
#     """
#     Calculate the generation of hydrated electrons (eₐq⁻) based on input parameters provided in a dictionary.
#
#     The dictionary 'params' should include all the parameters below:
#         - 'l': Path length (cm)
#         - 'I0_185': Incident intensity at 185 nm
#         - 'I0_254': Incident intensity at 254 nm
#         - 'c_h2o': Water concentration (mol/L)
#         - 'epsilon_h2o_185', 'phi_h2o_185': Molar absorptivity and quantum yield for water at 185 nm.
#         - 'epsilon_h2o_254', 'phi_h2o_254': Molar absorptivity and quantum yield for water at 254 nm.
#         - 'epsilon_oh_m_185', 'phi_oh_m_185': For the hydroxide ion at 185 nm.
#         - 'epsilon_cl_185', 'phi_cl_185': For chloride at 185 nm.
#         - 'epsilon_so3_185', 'phi_so3_185': For sulfite at 185 nm.
#         - 'epsilon_so3_254', 'phi_so3_254': For sulfite at 254 nm.
#         - 'epsilon_pfas_185': Molar absorptivity for PFAS at 185 nm.
#         - 'epsilon_pfas_254': Molar absorptivity for PFAS at 254 nm.
#         - 'beta_j': A constant (available for future use)
#         - 'pH': pH of the solution (used to compute hydroxide concentration)
#         - 'c_pfas_init': Initial concentration for PFAS
#         - 'c_cl': Chloride concentration
#         - 'c_so3': Sulfite concentration
#
#     All parameters that represent concentrations (keys starting with 'c') are ensured
#     to be NumPy arrays (vectors) to enable element‐wise operations.
#
#     Returns:
#       A NumPy array representing the generation of eaq⁻ for each element of the concentration vectors.
#     """
#
#     # Compute hydroxide concentration (c_oh_m) from the given pH (usually a scalar)
#     c_oh_m = np.power(10, -14.0 + pH)
#
#     # Compute total absorption (Sigma_f) for 185 nm
#     Sigma_f_185 = (
#             params["epsilon_h2o_185"] * params["c_h2o"]
#             + params["epsilon_oh_m_185"] * c_oh_m
#             + params["epsilon_cl_185"] * c_cl
#             + params["epsilon_so3_185"] * c_so3
#             + params["epsilon_pfas_185"] * c_pfas_init
#     )
#
#     # Compute total absorption (Sigma_f) for 254 nm
#     Sigma_f_254 = (
#             params["epsilon_h2o_254"] * params["c_h2o"]
#             + params["epsilon_so3_254"] * c_so3
#             + params["epsilon_pfas_254"] * c_pfas_init
#     )
#
#     # Calculate the fraction of absorption for each species at 185 nm
#     f_h2o_185 = (params["epsilon_h2o_185"] * params["c_h2o"]) / Sigma_f_185
#     f_oh_m_185 = (params["epsilon_oh_m_185"] * c_oh_m) / Sigma_f_185
#     f_cl_185 = (params["epsilon_cl_185"] * c_cl) / Sigma_f_185
#     f_so3_185 = (params["epsilon_so3_185"] * c_so3) / Sigma_f_185
#
#     # For 254 nm, calculate the fraction for sulfite
#     f_so3_254 = (params["epsilon_so3_254"] * c_so3) / Sigma_f_254
#
#     # Calculate contributions to eaq⁻ generation at 185 nm
#     term_h2o_185 = f_h2o_185 * params["phi_h2o_185"] * (
#                 1 - np.power(10, -params["epsilon_h2o_185"] * params["l"] * params["c_h2o"]))
#     term_oh_m_185 = f_oh_m_185 * params["phi_oh_m_185"] * (
#                 1 - np.power(10, -params["epsilon_oh_m_185"] * params["l"] * c_oh_m))
#     term_cl_185 = f_cl_185 * params["phi_cl_185"] * (
#                 1 - np.power(10, -params["epsilon_cl_185"] * params["l"] * c_cl))
#     term_so3_185 = f_so3_185 * params["phi_so3_185"] * (
#                 1 - np.power(10, -params["epsilon_so3_185"] * params["l"] * c_so3))
#
#     numerator_185 = params["I0_185"] * (term_h2o_185 + term_oh_m_185 + term_cl_185 + term_so3_185)
#
#     # Calculate contribution to eaq⁻ generation at 254 nm
#     numerator_254 = params["I0_254"] * f_so3_254 * params["phi_so3_254"] * (
#                 1 - np.power(10, -params["epsilon_so3_254"] * params["l"] * c_so3))
#
#     # Total eaq⁻ generation
#     generation_eaq = numerator_185 + numerator_254
#     return generation_eaq
    k1, k2, k3, k4, k5, k6, k7 = [np.float32(v) for v in
                                  [5.259223e+06, 5.175076e+08, 5.232472e+08, 5.551598e+08, 5.748630e+08, 5.647780e+08,
                                   5.005279e+08]]
    params = {
            "l": 0.24,           # path length in cm
            "I0_185": 2.07e-6,     # intensity at 185 nm
            "I0_254": 5.19e-4,     # intensity at 254 nm
            "c_h2o": 55.6,         # water concentration (mol/L)
            "epsilon_h2o_185": 0.0324,
            "phi_h2o_185": 0.045,
            "epsilon_h2o_254": 0.032,
            "phi_h2o_254": 0,
            "epsilon_oh_m_185": 3200.0,
            "phi_oh_m_185": 0.11,
            "epsilon_cl_185": 3540.0,
            "phi_cl_185": 0.43,
            "epsilon_so3_185": 3729.5,
            "phi_so3_185": 0.85,
            "epsilon_so3_254": 21.22,
            "phi_so3_254": 0.11,
            "epsilon_pfas_185": 2689.5,
            "epsilon_pfas_254": 28.8
        }

    # numerator = generate_eaq(params, c_pfas_init, c_cl, c_so3, pH)

    # k1 = 1e9
    # k_so3_eaq = 1.5e6
    # k_cl_eaq = 1e6
    # beta_j = 2.57e4
    #
    # denominator = k1 * c_pfas_init + beta_j + k_so3_eaq * c_so3 + k_cl_eaq * c_cl
    #
    # eaq_ss = numerator/denominator

    pH = 5.7
    c_pfas_init = 9.6e-7  # PFAS initial concentration (mol/L)
    c_cl = 0.0  # chloride concentration (mol/L)
    c_so3 = 0.0  # sulfite concentration (mol/L)

    df = pd.read_csv('./data/PFAS_data.csv')

    t_true = df['time (s)'].values
    t_pinn = np.arange(t_true[0], np.round(t_true[-1]), 1.0)
    y_train = df[['C7F15COO-', 'C5F11COO-', 'C3F7COO-', 'C2F5COO-', 'CF3COO-']].values[np.newaxis, :, :]
    batch_input = t_pinn[np.newaxis, :, np.newaxis]

    initial_state = np.array([[1.96e-07, 0, 0, 0, 0, 0, 0, 0]], dtype='float32')
    model = create_model(k1, k2, k3, k4, k5, k6, k7, params, c_cl, c_so3, pH, 1.0, initial_state, batch_input.shape, t_pinn, t_true)

    y_pred_before = model.predict(batch_input)

    start_time = time.time()
    model.fit(batch_input, y_train, epochs=200, steps_per_epoch=1, verbose=1)
    end_time = time.time()
    print(f"Training Time: {end_time - start_time:.2f} seconds")

    y_pred = model.predict(batch_input)

    # Plotting results
    # Plotting results
    outputs_to_plot = [0, 1, 2, 3, 4]
    labels = ['Output y0', 'Output y1', 'Output y2', 'Output y3', 'Output y4']
    plt.figure(figsize=(9, 6))

    # Ensure subplot layout matches the number of outputs
    num_plots = len(outputs_to_plot)
    rows = (num_plots + 1) // 2  # Calculate required rows for a 2-column layout

    for idx, (i, label) in enumerate(zip(outputs_to_plot, labels), start=1):
        plt.subplot(rows, 2, idx)  # Adjust for the number of rows and columns

        # Scatter plot for y_train
        plt.scatter(t_true, y_train[0, :, i], color='gray', label='Raw Data', marker='o', alpha=0.7)
        plt.plot(t_pinn, y_pred_before[0, :, i], color='r', label='Before Training')

        # Line plot for y_pred
        plt.plot(t_pinn, y_pred[0, :, i], 'b', label='After Training')

        plt.xlabel('Time (t)')
        plt.ylabel(label)
        plt.grid(True)
        plt.legend()

    # Display the final plot after the loop
    plt.tight_layout()
    output_file = "Results/HPINN_minimal_model_exp.png"
    plt.savefig(output_file, dpi=600, bbox_inches='tight')
    plt.show()

    # Access the RNN layer in the model
    rnn_layer = model.layers[0]  # Assuming the RNN is the first layer

    # Access the RungeKuttaIntegratorCell
    rk_cell = rnn_layer.cell

    # Extract and print trained parameter values
    trained_params = {name: 10 ** rk_cell.log_k_values[name].numpy() for name in rk_cell.log_k_values}

    print("Trained parameters:")
    for name, value in trained_params.items():
        print(f"{name}: {value:.6e}")