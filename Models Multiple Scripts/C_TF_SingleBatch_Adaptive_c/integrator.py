# cinn_pfas/integrator.py
# RK4 Keras RNN cell with *adaptive* hydrated electron generation (c_eaq) from absorption physics

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer

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
            initial_state: Initial state vector (shape [1, 8]).
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
        self._log_k_init = np.log10([k1, k2, k3, k4, k5, k6, k7])

    def build(self, input_shape):
        """Build the trainable weights for reaction rate constants (log-space)."""
        k_names = ['k1', 'k2', 'k3', 'k4', 'k5', 'k6', 'k7']
        self.log_k_values = {
            name: self.add_weight(
                name=f'log_{name}',
                shape=(),
                initializer=tf.constant_initializer(val),
                trainable=True
            )
            for name, val in zip(k_names, self._log_k_init)
        }
        self.built = True

    def call(self, inputs, states):
        """
        Perform one time-step update using RK4.

        Returns:
            output: selected species concentrations
            new_state: full state (all 8 species) at next step
        """
        # Convert log-params to linear space
        params = {name: 10.0 ** logv for name, logv in self.log_k_values.items()}

        y = states[0]  # Current state vector: shape (batch, 8)

        # RK4 increments:
        k1 = self._fun(y, params) * self.dt
        k2 = self._fun(y + 0.5 * k1, params) * self.dt
        k3 = self._fun(y + 0.5 * k2, params) * self.dt
        k4 = self._fun(y + k3, params) * self.dt
        y_next = y + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

        # Outputs (match your working script): indices [0,2,4,5,6]
        output_components = [y_next[:, i:i+1] for i in [0, 2, 4, 5, 6]]
        output = tf.concat(output_components, axis=-1)
        return output, [y_next]

    def _fun(self, y, params):
        """
        Compute derivatives dy/dt from chain kinetics with adaptive c_eaq.
        y: (batch, 8)
        params: dict with k1..k7 (linear space)
        """
        # Extract PFAS chain species (first 7 entries)
        y_vars = [y[:, i:i+1] for i in range(7)]
        k_vars = [params[f'k{i+1}'] for i in range(7)]

        # Compute hydrated electron concentration, c_eaq, using current settings
        numerator = self.generation_of_eaq()

        # Additional kinetic params (constants)
        k_so3_eaq = 1.5e6
        k_cl_eaq = 1e6
        beta_j = 2.57e4

        # Use initial PFAS concentration from initial state (first species)
        c_pfas_init = float(self.initial_state[0, 0])
        denominator = params['k1'] * c_pfas_init + beta_j + k_so3_eaq * self.c_so3 + k_cl_eaq * self.c_cl
        c_eaq = numerator / denominator  # scalar

        # Reaction rates per species (first-order in PFAS and c_eaq)
        rates = [k * c_eaq * y_var for k, y_var in zip(k_vars, y_vars)]

        # Chain of ODEs
        dy1 = -rates[0]
        dy2 = rates[0] - rates[1]
        dy3 = rates[1] - rates[2]
        dy4 = rates[2] - rates[3]
        dy5 = rates[3] - rates[4]
        dy6 = rates[4] - rates[5]
        dy7 = rates[5] - rates[6]
        dy8 = 2.0 * tf.reduce_sum(rates, axis=0, keepdims=False)  # F-

        return tf.concat([dy1, dy2, dy3, dy4, dy5, dy6, dy7, dy8], axis=-1)

    def generation_of_eaq(self):
        """
        Generation of hydrated electrons from 185/254 nm absorption.
        Returns scalar generation rate.
        """
        p = self.constants
        c_pfas_init = float(self.initial_state[0, 0])
        # [OH-] from pH
        c_oh_m = np.power(10.0, -14.0 + self.pH)

        # Total absorption @185
        Sigma_f_185 = (p["epsilon_h2o_185"] * p["c_h2o"] +
                       p["epsilon_oh_m_185"] * c_oh_m +
                       p["epsilon_cl_185"] * self.c_cl +
                       p["epsilon_so3_185"] * self.c_so3 +
                       p["epsilon_pfas_185"] * c_pfas_init)

        # Total absorption @254
        Sigma_f_254 = (p["epsilon_h2o_254"] * p["c_h2o"] +
                       p["epsilon_so3_254"] * self.c_so3 +
                       p["epsilon_pfas_254"] * c_pfas_init)

        # Fractions @185
        f_h2o_185 = (p["epsilon_h2o_185"] * p["c_h2o"]) / Sigma_f_185
        f_oh_m_185 = (p["epsilon_oh_m_185"] * c_oh_m) / Sigma_f_185
        f_cl_185 = (p["epsilon_cl_185"] * self.c_cl) / Sigma_f_185
        f_so3_185 = (p["epsilon_so3_185"] * self.c_so3) / Sigma_f_185

        # Fraction @254
        f_so3_254 = (p["epsilon_so3_254"] * self.c_so3) / Sigma_f_254

        # Contributions @185
        term_h2o_185 = f_h2o_185 * p["phi_h2o_185"] * (1.0 - np.power(10.0, -p["epsilon_h2o_185"] * p["l"] * p["c_h2o"]))
        term_oh_m_185 = f_oh_m_185 * p["phi_oh_m_185"] * (1.0 - np.power(10.0, -p["epsilon_oh_m_185"] * p["l"] * c_oh_m))
        term_cl_185 = f_cl_185 * p["phi_cl_185"] * (1.0 - np.power(10.0, -p["epsilon_cl_185"] * p["l"] * self.c_cl))
        term_so3_185 = f_so3_185 * p["phi_so3_185"] * (1.0 - np.power(10.0, -p["epsilon_so3_185"] * p["l"] * self.c_so3))
        numerator_185 = p["I0_185"] * (term_h2o_185 + term_oh_m_185 + term_cl_185 + term_so3_185)

        # Contribution @254
        numerator_254 = p["I0_254"] * f_so3_254 * p["phi_so3_254"] * (1.0 - np.power(10.0, -p["epsilon_so3_254"] * p["l"] * self.c_so3))

        return numerator_185 + numerator_254

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        """Return initial state to Keras RNN wrapper."""
        return self.initial_state


def interpolate_predictions(t_pinn, t_true, y_pred):
    """
    Linearly interpolate predictions y_pred (on t_pinn) at the true time points t_true.

    Args:
        t_pinn (array-like): Time grid used in model predictions.
        t_true (array-like): Ground truth time points.
        y_pred (tf.Tensor): Predictions [batch, timesteps, outputs].

    Returns:
        tf.Tensor with shape like y_true.
    """
    t_pinn_tensor = tf.constant(t_pinn, dtype=tf.float32)
    t_true_tensor = tf.constant(t_true, dtype=tf.float32)

    # indices into t_pinn for each t_true
    indices = tf.searchsorted(t_pinn_tensor, t_true_tensor, side='left') - 1
    indices = tf.clip_by_value(indices, 0, tf.shape(t_pinn_tensor)[0] - 2)

    t0 = tf.gather(t_pinn_tensor, indices)
    t1 = tf.gather(t_pinn_tensor, indices + 1)
    y0 = tf.gather(y_pred, indices, axis=1)
    y1 = tf.gather(y_pred, indices + 1, axis=1)

    w = (t_true_tensor - t0) / (t1 - t0)
    w = tf.reshape(w, [1, -1, 1])

    return y0 + w * (y1 - y0)
