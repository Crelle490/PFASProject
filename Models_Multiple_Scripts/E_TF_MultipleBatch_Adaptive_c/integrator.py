# integrator.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer

class RungeKuttaIntegratorCell(Layer):
    def __init__(self, k1, k2, k3, k4, k5, k6, k7,
                 constants, c_cl, c_so3, pH, dt, dummy_initial_state,
                 for_prediction=False, **kwargs):
        """
        RK4 integrator cell with adaptive hydrated electron generation.
        dummy_initial_state: (1, 8) array used only to build the cell.
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

        # update catalyst based on input
        self.c_cl = float(inputs[0])
        self.c_so3 = float(inputs[1])

        # RK4 increments
        k1 = self._fun(y, params) * self.dt
        k2 = self._fun(y + 0.5 * k1, params) * self.dt
        k3 = self._fun(y + 0.5 * k2, params) * self.dt
        k4 = self._fun(y + k3, params) * self.dt
        y_next = y + (k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0

        if self.for_prediction:
            output = y_next  # full 8-state
        else:
            # Select outputs [0,2,4,5,6] for training
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

        # Use initial PFAS concentration from dummy initial state (first species)
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
        Compute the generation rate of hydrated electrons (e_aq−) from 185/254 nm absorption.
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
