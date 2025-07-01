import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import RNN, Layer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay


class RungeKuttaIntegratorCell(Layer):
    def __init__(self, k1, k2, k3, k4, k5, k6, k7,
                 constants, c_cl, c_so3, pH, dt, initial_state, **kwargs):
        """
        A custom RNN cell that uses a 4th-order Runge-Kutta method for time integration.

        Parameters:
            k1, ..., k7: Reaction rate constants.
            constants: Dictionary of physical parameters.
            c_cl: Chloride concentration.
            c_so3: Sulfite concentration.
            pH: pH of the solution.
            dt: Time step for integration.
            initial_state: Initial state vector.
            kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.constants = constants
        self.c_cl = c_cl
        self.c_so3 = c_so3
        self.pH = pH
        self.dt = dt
        self.initial_state = initial_state  # shape: (1, 8)
        self.state_size = 8

        # Compute the base-10 logarithm of the reaction rate constants.
        self.log_k_values_init = np.log10([k1, k2, k3, k4, k5, k6, k7])

    def build(self, input_shape):
        """Build the trainable weights for reaction rate constants."""
        k_names = ['k1', 'k2', 'k3', 'k4', 'k5', 'k6', 'k7']
        self.log_k_values = {
            name: self.add_weight(
                name=f'log_{name}',
                shape=(),
                initializer=tf.constant_initializer(value),
                trainable=True
            )
            for name, value in zip(k_names, self.log_k_values_init)
        }
        self.built = True

    def call(self, inputs, states):
        """
        Perform one time-step update using a 4th-order Runge-Kutta method.

        Returns:
            A tuple (output, [new_state]), where output is a subset of the state.
        """
        # Compute the actual rate constants from their logarithms.
        params = {name: 10 ** log_val for name, log_val in self.log_k_values.items()}

        y = states[0]  # Current state vector.
        # Compute Runge-Kutta increments:
        k1 = self._fun(y, params) * self.dt
        k2 = self._fun(y + 0.5 * k1, params) * self.dt
        k3 = self._fun(y + 0.5 * k2, params) * self.dt
        k4 = self._fun(y + k3, params) * self.dt
        y_next = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

        # Select specific components for the output.
        output_components = [y_next[:, i:i + 1] for i in [0, 2, 4, 5, 6]]
        output = tf.concat(output_components, axis=-1)
        return output, [y_next]

    def _fun(self, y, params):
        """
        Compute state derivatives based on the reaction kinetics.

        Parameters:
            y: Current state tensor.
            params: Dictionary of current rate constant values.

        Returns:
            A tensor of concatenated derivatives.
        """
        # Extract the first seven state variables.
        y_vars = [y[:, i:i + 1] for i in range(7)]
        # Get reaction rate constants.
        k_vars = [params[f'k{i + 1}'] for i in range(7)]

        # Calculate hydrated electron generation.
        numerator = self.generation_of_eaq()

        # Additional kinetic parameters.
        k_so3_eaq = 1.5e6
        k_cl_eaq = 1e6
        beta_j = 2.57e4

        # Use the initial PFAS concentration from the state (assumed to be the first value).
        c_pfas_init = self.initial_state[0, 0]
        denominator = params['k1'] * c_pfas_init + beta_j + k_so3_eaq * self.c_so3 + k_cl_eaq * self.c_cl

        # Hydrated electron concentration.
        c_eaq = numerator / denominator

        # Compute the reaction rates for each state variable.
        rates = [k * c_eaq * y_var for k, y_var in zip(k_vars, y_vars)]

        # Define the state derivatives.
        dy1 = -rates[0]
        dy2 = rates[0] - rates[1]
        dy3 = rates[1] - rates[2]
        dy4 = rates[2] - rates[3]
        dy5 = rates[3] - rates[4]
        dy6 = rates[4] - rates[5]
        dy7 = rates[5] - rates[6]
        dy8 = 2 * tf.reduce_sum(rates, axis=0, keepdims=False)

        return tf.concat([dy1, dy2, dy3, dy4, dy5, dy6, dy7, dy8], axis=-1)

    def generation_of_eaq(self):
        """
        Compute the generation rate of hydrated electrons (eₐq⁻) based on absorption parameters.

        Returns:
            Hydrated electron generation as a scalar or tensor.
        """
        params = self.constants
        c_pfas_init = self.initial_state[0, 0]
        # Calculate hydroxide ion concentration from pH.
        c_oh_m = np.power(10, -14.0 + self.pH)

        # Total absorption at 185 nm.
        Sigma_f_185 = (params["epsilon_h2o_185"] * params["c_h2o"] +
                       params["epsilon_oh_m_185"] * c_oh_m +
                       params["epsilon_cl_185"] * self.c_cl +
                       params["epsilon_so3_185"] * self.c_so3 +
                       params["epsilon_pfas_185"] * c_pfas_init)

        # Total absorption at 254 nm.
        Sigma_f_254 = (params["epsilon_h2o_254"] * params["c_h2o"] +
                       params["epsilon_so3_254"] * self.c_so3 +
                       params["epsilon_pfas_254"] * c_pfas_init)

        # Absorption fractions at 185 nm.
        f_h2o_185 = (params["epsilon_h2o_185"] * params["c_h2o"]) / Sigma_f_185
        f_oh_m_185 = (params["epsilon_oh_m_185"] * c_oh_m) / Sigma_f_185
        f_cl_185 = (params["epsilon_cl_185"] * self.c_cl) / Sigma_f_185
        f_so3_185 = (params["epsilon_so3_185"] * self.c_so3) / Sigma_f_185

        # Fraction for 254 nm.
        f_so3_254 = (params["epsilon_so3_254"] * self.c_so3) / Sigma_f_254

        # Contributions to generation at 185 nm.
        term_h2o_185 = f_h2o_185 * params["phi_h2o_185"] * (
                1 - np.power(10, -params["epsilon_h2o_185"] * params["l"] * params["c_h2o"]))
        term_oh_m_185 = f_oh_m_185 * params["phi_oh_m_185"] * (
                1 - np.power(10, -params["epsilon_oh_m_185"] * params["l"] * c_oh_m))
        term_cl_185 = f_cl_185 * params["phi_cl_185"] * (
                1 - np.power(10, -params["epsilon_cl_185"] * params["l"] * self.c_cl))
        term_so3_185 = f_so3_185 * params["phi_so3_185"] * (
                1 - np.power(10, -params["epsilon_so3_185"] * params["l"] * self.c_so3))
        numerator_185 = params["I0_185"] * (term_h2o_185 + term_oh_m_185 + term_cl_185 + term_so3_185)

        # Contribution at 254 nm.
        numerator_254 = params["I0_254"] * f_so3_254 * params["phi_so3_254"] * (
                1 - np.power(10, -params["epsilon_so3_254"] * params["l"] * self.c_so3))

        return numerator_185 + numerator_254

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        """Return the initial state for the RNN cell."""
        return self.initial_state


def interpolate_predictions(t_pinn, t_true, y_pred):
    """
    Linearly interpolate predictions at the true time points.

    Args:
        t_pinn (array-like): Time points used in the predictions.
        t_true (array-like): True time points for evaluation.
        y_pred (tf.Tensor): Model predictions of shape [batch, timesteps, outputs].

    Returns:
        tf.Tensor: Interpolated predictions.
    """
    t_pinn_tensor = tf.constant(t_pinn, dtype=tf.float32)
    t_true_tensor = tf.constant(t_true, dtype=tf.float32)

    # Find indices corresponding to t_true in t_pinn.
    indices = tf.searchsorted(t_pinn_tensor, t_true_tensor, side='left') - 1
    indices = tf.clip_by_value(indices, 0, len(t_pinn_tensor) - 2)

    t0 = tf.gather(t_pinn_tensor, indices)
    t1 = tf.gather(t_pinn_tensor, indices + 1)
    y0 = tf.gather(y_pred, indices, axis=1)
    y1 = tf.gather(y_pred, indices + 1, axis=1)

    # Linear interpolation weight.
    w = (t_true_tensor - t0) / (t1 - t0)
    w = tf.reshape(w, [1, -1, 1])

    return y0 + w * (y1 - y0)


def create_loss_fn(t_pinn, t_true):
    """
    Create a custom loss function with dynamic scaling and weighting.

    Args:
        t_pinn: Time grid used in model predictions.
        t_true: Ground truth time points.

    Returns:
        A loss function that computes a scaled mean-squared error.
    """

    def my_loss_fn(y_true, y_pred):
        # Interpolate predictions to the true time grid.
        y_pred_interp = interpolate_predictions(t_pinn, t_true, y_pred)
        y_pred_interp.set_shape(y_true.shape)

        # Compute the maximum absolute values per state.
        y_max = tf.reduce_max(tf.abs(y_true), axis=[0, 1])
        max_y_max = tf.reduce_max(y_max)
        weights = max_y_max / y_max

        # Scaling coefficient.
        coefficient = 1.0 / max_y_max

        # Per-state mean squared errors.
        state_losses = tf.reduce_mean(tf.square(y_true - y_pred_interp), axis=[0, 1])
        weighted_loss = tf.reduce_sum(weights * state_losses)
        scaled_loss = 10 * coefficient ** 2 * weighted_loss

        return scaled_loss

    return my_loss_fn


def create_model(k1, k2, k3, k4, k5, k6, k7, constants,
                 c_cl, c_so3, pH, dt, initial_state, batch_input_shape,
                 t_pinn, t_true):
    """
    Create and compile the Keras model using the custom Runge-Kutta integrator cell.

    Args:
        k1, ..., k7: Reaction rate constants.
        constants: Dictionary of physical parameters.
        c_cl, c_so3: Chloride and sulfite concentrations.
        pH: pH value.
        dt: Time step size.
        initial_state: Initial state vector.
        batch_input_shape: Input shape for the RNN.
        t_pinn: Time grid used in the model.
        t_true: True time points.

    Returns:
        A compiled tf.keras.Model.
    """
    rk_cell = RungeKuttaIntegratorCell(k1, k2, k3, k4, k5, k6, k7,
                                       constants, c_cl, c_so3, pH, dt,
                                       initial_state)
    
    rnn_layer = RNN(cell=rk_cell, batch_input_shape=batch_input_shape, return_sequences=True)
    model = Sequential([rnn_layer])

    loss_fn = create_loss_fn(t_pinn, t_true)
    lr_schedule = PiecewiseConstantDecay(boundaries=[70, 150, 250],
                                         values=[5e-2, 1e-2, 1e-3, 1e-4])
    model.compile(optimizer=RMSprop(learning_rate=lr_schedule), loss=loss_fn)
    return model


def plot_results(t_true, t_pinn, y_train, y_pred_before, y_pred_after):
    """Plot raw data alongside model predictions before and after training."""
    outputs_to_plot = [0, 1, 2, 3, 4]
    labels = ['Output y0', 'Output y1', 'Output y2', 'Output y3', 'Output y4']

    plt.figure(figsize=(9, 6))
    num_plots = len(outputs_to_plot)
    rows = (num_plots + 1) // 2  # Two-column layout

    for idx, (i, label) in enumerate(zip(outputs_to_plot, labels), start=1):
        plt.subplot(rows, 2, idx)
        plt.scatter(t_true, y_train[0, :, i], color='gray', label='Raw Data',
                    marker='o', alpha=0.7)
        plt.plot(t_pinn, y_pred_before[0, :, i], color='r', label='Before Training')
        plt.plot(t_pinn, y_pred_after[0, :, i], 'b', label='After Training')
        plt.xlabel('Time (s)')
        plt.ylabel(label)
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    output_file = "Results/HPINN_minimal_model_exp.png"
    plt.savefig(output_file, dpi=600, bbox_inches='tight')
    plt.show()


def display_trained_parameters(model):
    """Extract and print the trained reaction rate parameters from the model."""
    # Assuming the first layer is the RNN.
    rnn_layer = model.layers[0]
    rk_cell = rnn_layer.cell
    trained_params = {name: 10 ** rk_cell.log_k_values[name].numpy()
                      for name in rk_cell.log_k_values}
    print("Trained parameters:")
    for name, value in trained_params.items():
        print(f"{name}: {value:.6e}")


def main():
    # Define reaction rate constants.
    k_values = [5.259223e+06, 5.175076e+08, 5.232472e+08,
                5.551598e+08, 5.748630e+08, 5.647780e+08, 5.005279e+08]
    k1, k2, k3, k4, k5, k6, k7 = [np.float32(v) for v in k_values]

    # Physical and experimental parameters.
    params = {
        "l": 0.24,  # Path length in cm.
        "I0_185": 2.07e-6,  # Incident intensity at 185 nm.
        "I0_254": 5.19e-4,  # Incident intensity at 254 nm.
        "c_h2o": 55.6,  # Water concentration (mol/L).
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

    # Operating conditions.
    pH = 5.7

    c_pfas_init = 9.6e-7  # PFAS initial concentration (mol/L).
    c_cl = 0.0      # Chloride concentration (mol/L).
    c_so3 = 0.0  # Sulfite concentration (mol/L).

    # Load experimental data.
    df = pd.read_csv('./data/PFAS_data.csv')
    t_true = df['time (s)'].values
    t_pinn = np.arange(t_true[0], np.round(t_true[-1]), 1.0)
    y_train = df[['C7F15COO-', 'C5F11COO-', 'C3F7COO-', 'C2F5COO-', 'CF3COO-']].values
    y_train = y_train[np.newaxis, :, :]  # Add batch dimension.
    batch_input = t_pinn[np.newaxis, :, np.newaxis]  # Time input with channel dimension.

    # Define the initial state for the RNN cell.
    initial_state = np.array([[df['C7F15COO-'][0], 0, 0, 0, 0, 0, 0, 0]], dtype='float32')

    # Create and compile the model.
    model = create_model(k1, k2, k3, k4, k5, k6, k7,
                         params, c_cl, c_so3, pH, dt=1.0,
                         initial_state=initial_state,
                         batch_input_shape=batch_input.shape,
                         t_pinn=t_pinn, t_true=t_true)

    # Predict before training.
    y_pred_before = model.predict(batch_input)

    # Train the model.
    start_time = time.time()
    model.fit(batch_input, y_train, epochs=200, steps_per_epoch=1, verbose=1)
    training_time = time.time() - start_time
    print(f"Training Time: {training_time:.2f} seconds")

    # Predict after training.
    y_pred_after = model.predict(batch_input)

    # Plot results.
    plot_results(t_true, t_pinn, y_train, y_pred_before, y_pred_after)

    # Display trained parameters.
    display_trained_parameters(model)


if __name__ == "__main__":
    main()
