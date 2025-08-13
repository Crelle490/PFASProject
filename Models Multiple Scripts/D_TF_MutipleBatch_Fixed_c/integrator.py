# integrator.py
# RK4 Keras RNN cell with adaptive hydrated electron generation (c_eaq),
# with an option to output either selected PFAS species (training) or full state (prediction).

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer

class RungeKuttaIntegratorCell(Layer):
    def __init__(self, k1, k2, k3, k4, k5, k6, k7,
                 constants, c_cl, c_so3, pH, dt, dummy_initial_state,
                 for_prediction=False, **kwargs):
        """
        dummy_initial_state: a (1,8) numpy array used only for building the cell.
        The actual *batch* initial states are provided at call-time via the RNN's initial_state= argument.
        """
        super().__init__(**kwargs)
        self.dt = float(dt)
        self.initial_state = np.asarray(dummy_initial_state, dtype=np.float32)
        self.state_size = 8

        self.constants = constants
        self.c_cl = float(c_cl)
        self.c_so3 = float(c_so3)
        self.pH = float(pH)
        self.for_prediction = bool(for_prediction)

        # Trainable parameters in log10-space.
        self._log_k_init = np.log10([k1, k2, k3, k4, k5, k6, k7])

    def build(self, input_shape):
        k_names = ['k1','k2','k3','k4','k5','k6','k7']
        self.log_k_values = {
            name: self.add_weight(
                name=f'log_{name}', shape=(), initializer=tf.constant_initializer(val), trainable=True
            )
            for name, val in zip(k_names, self._log_k_init)
        }
        self.built = True

    def call(self, inputs, states):
        # Convert log parameters to actual values.
        params = {name: 10.0 ** log_v for name, log_v in self.log_k_values.items()}

        y = states[0]  # shape: (batch, 8)

        # RK4 increments
        k1 = self._fun(y, params) * self.dt
        k2 = self._fun(y + 0.5 * k1, params) * self.dt
        k3 = self._fun(y + 0.5 * k2, params) * self.dt
        k4 = self._fun(y + k3, params) * self.dt
        y_next = y + (k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0

        if self.for_prediction:
            output = y_next  # full state (8)
        else:
            # Select PFAS outputs [0,2,4,5,6] (5 outputs) for training
            output = tf.concat([y_next[:, 0:1], y_next[:, 2:3],
                                y_next[:, 4:5], y_next[:, 5:6], y_next[:, 6:7]], axis=-1)
        return output, [y_next]

    def _fun(self, y, params):
        # y: (batch, 8)
        y_vars = [y[:, i:i+1] for i in range(7)]  # first 7 PFAS species

        # Hydrated electron generation (scalar)
        numerator = self.generation_of_eaq()

        # Additional kinetic constants
        k_so3_eaq = 1.5e6
        k_cl_eaq  = 1.0e6
        beta_j    = 2.57e4

        # Use initial PFAS concentration from *dummy* initial state (first species)
        c_pfas_init = float(self.initial_state[0, 0])
        denominator = params['k1'] * c_pfas_init + beta_j + k_so3_eaq * self.c_so3 + k_cl_eaq * self.c_cl
        c_eaq = numerator / denominator  # scalar

        # Reaction rates (first-order in PFAS and c_eaq)
        rates = [params[f'k{i+1}'] * c_eaq * y_vars[i] for i in range(7)]

        dy1 = -rates[0]
        dy2 = rates[0] - rates[1]
        dy3 = rates[1] - rates[2]
        dy4 = rates[2] - rates[3]
        dy5 = rates[3] - rates[4]
        dy6 = rates[4] - rates[5]
        dy7 = rates[5] - rates[6]
        dy8 = 2.0 * tf.reduce_sum(rates, axis=0)  # fluoride

        return tf.concat([dy1, dy2, dy3, dy4, dy5, dy6, dy7, dy8], axis=-1)

    def generation_of_eaq(self):
        """
        Compute the generation rate of hydrated electrons (e_aq⁻) from 185/254 nm absorption.
        Returns a scalar (float).
        """
        p = self.constants
        c_pfas_init = float(self.initial_state[0, 0])
        # [OH-] from pH
        c_oh_m = np.power(10.0, -14.0 + self.pH)

        # Total absorption @185 nm
        Sigma_f_185 = (p["epsilon_h2o_185"] * p["c_h2o"] +
                       p["epsilon_oh_m_185"] * c_oh_m +
                       p["epsilon_cl_185"]   * self.c_cl +
                       p["epsilon_so3_185"]  * self.c_so3 +
                       p["epsilon_pfas_185"] * c_pfas_init)

        # Total absorption @254 nm
        Sigma_f_254 = (p["epsilon_h2o_254"] * p["c_h2o"] +
                       p["epsilon_so3_254"]  * self.c_so3 +
                       p["epsilon_pfas_254"] * c_pfas_init)

        # Fractions @185
        f_h2o_185 = (p["epsilon_h2o_185"] * p["c_h2o"]) / Sigma_f_185
        f_oh_m_185 = (p["epsilon_oh_m_185"] * c_oh_m) / Sigma_f_185
        f_cl_185   = (p["epsilon_cl_185"]   * self.c_cl) / Sigma_f_185
        f_so3_185  = (p["epsilon_so3_185"]  * self.c_so3) / Sigma_f_185

        # Fraction @254
        f_so3_254 = (p["epsilon_so3_254"] * self.c_so3) / Sigma_f_254

        # Contributions @185
        term_h2o_185 = f_h2o_185 * p["phi_h2o_185"] * (1.0 - np.power(10.0, -p["epsilon_h2o_185"] * p["l"] * p["c_h2o"]))
        term_oh_m_185 = f_oh_m_185 * p["phi_oh_m_185"] * (1.0 - np.power(10.0, -p["epsilon_oh_m_185"] * p["l"] * c_oh_m))
        term_cl_185   = f_cl_185   * p["phi_cl_185"]   * (1.0 - np.power(10.0, -p["epsilon_cl_185"]   * p["l"] * self.c_cl))
        term_so3_185  = f_so3_185  * p["phi_so3_185"]  * (1.0 - np.power(10.0, -p["epsilon_so3_185"]  * p["l"] * self.c_so3))
        numerator_185 = p["I0_185"] * (term_h2o_185 + term_oh_m_185 + term_cl_185 + term_so3_185)

        # Contribution @254
        numerator_254 = p["I0_254"] * f_so3_254 * p["phi_so3_254"] * (1.0 - np.power(10.0, -p["epsilon_so3_254"] * p["l"] * self.c_so3))

        return float(numerator_185 + numerator_254)

def interpolate_predictions(t_pinn, t_true, y_pred):
    """
    Interpolate y_pred (batch=1 assumed for a single sequence slice) at experimental time points t_true.
    t_pinn: (T_sim,) numpy array. y_pred: (1, T_sim, D).
    Returns: (1, len(t_true), D) tensor.
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
