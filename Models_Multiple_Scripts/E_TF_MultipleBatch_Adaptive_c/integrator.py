# integrator.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer

class RungeKuttaIntegratorCell(Layer):
    def __init__(self, k1, k2, k3, k4, k5, k6, k7,
                 constants, c_cl, c_so3, pH, dt, dummy_initial_state,
                 for_prediction=False, **kwargs):
        super().__init__(**kwargs)

        self.dt = tf.constant(float(dt), dtype=tf.float32)
        self.initial_state = np.asarray(dummy_initial_state, dtype=np.float32)
        self.state_size = 8

        # Keep constants dict; assume floats
        self.constants = constants

        # Only-So3 dynamic: set Cl constant (either from config or force 0)
        self.c_cl = tf.constant(0.0, dtype=tf.float32)  # <-- fixed 0 as you did later
        self.pH = tf.constant(float(pH), dtype=tf.float32)

        self.for_prediction = bool(for_prediction)

        self._log_k_init = np.log10([k1, k2, k3, k4, k5, k6, k7])
        
        self.k_so3_eaq = tf.constant(1.5e6, tf.float32)
        self.k_cl_eaq  = tf.constant(1.0e6, tf.float32)
        self.beta_j    = tf.constant(2.57e4, tf.float32)
        self.c_pfas_init = tf.constant(float(self.initial_state[0, 0]), tf.float32)
        p = self.constants
        tf32 = tf.float32
        self.eps_h2o_185  = tf.constant(float(p["epsilon_h2o_185"]), dtype=tf32)
        self.eps_oh_185   = tf.constant(float(p["epsilon_oh_m_185"]), dtype=tf32)
        self.eps_cl_185   = tf.constant(float(p["epsilon_cl_185"]), dtype=tf32)
        self.eps_so3_185  = tf.constant(float(p["epsilon_so3_185"]), dtype=tf32)
        self.eps_pfas_185 = tf.constant(float(p["epsilon_pfas_185"]), dtype=tf32)

        self.eps_h2o_254  = tf.constant(float(p["epsilon_h2o_254"]), dtype=tf32)
        self.eps_so3_254  = tf.constant(float(p["epsilon_so3_254"]), dtype=tf32)
        self.eps_pfas_254 = tf.constant(float(p["epsilon_pfas_254"]), dtype=tf32)

        self.phi_h2o_185  = tf.constant(float(p["phi_h2o_185"]), dtype=tf32)
        self.phi_oh_185   = tf.constant(float(p["phi_oh_m_185"]), dtype=tf32)
        self.phi_cl_185   = tf.constant(float(p["phi_cl_185"]), dtype=tf32)
        self.phi_so3_185  = tf.constant(float(p["phi_so3_185"]), dtype=tf32)
        self.phi_so3_254  = tf.constant(float(p["phi_so3_254"]), dtype=tf32)

        self.I0_185 = tf.constant(float(p["I0_185"]), dtype=tf32)
        self.I0_254 = tf.constant(float(p["I0_254"]), dtype=tf32)

        self.c_h2o = tf.constant(float(p["c_h2o"]), dtype=tf32)
        self.l = tf.constant(float(p["l"]), dtype=tf32)

        # [OH-] from pH (scalar)
        self.c_oh_m = tf.pow(10.0, -14.0 + self.pH)

    def build(self, input_shape):
        k_names = ['k1','k2','k3','k4','k5','k6','k7']
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
        params = {name: tf.pow(10.0, log_v) for name, log_v in self.log_k_values.items()}
        y = states[0]  # (batch, 8)

        # inputs: (batch, 1) containing c_so3(t)
        c_so3_t = tf.cast(inputs[:, 0:1], tf.float32)

        k1 = self._fun(y, params, c_so3_t) * self.dt
        k2 = self._fun(y + 0.5 * k1, params, c_so3_t) * self.dt
        k3 = self._fun(y + 0.5 * k2, params, c_so3_t) * self.dt
        k4 = self._fun(y + k3, params, c_so3_t) * self.dt
        y_next = y + (k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0

        if self.for_prediction:
            output = y_next
        else:
            output = tf.concat([y_next[:, 0:1], y_next[:, 2:3],
                                y_next[:, 4:5], y_next[:, 5:6], y_next[:, 6:7]], axis=-1)
        return output, [y_next]

    def _fun(self, y, params, c_so3_t):
        y_vars = [y[:, i:i+1] for i in range(7)]  # first 7 PFAS species

        numerator = self.generation_of_eaq(c_so3_t)  # (batch,1)

        # IMPORTANT: use c_so3_t and self.c_cl (not undefined c_so3/c_cl)
        denominator = (params['k1'] * self.c_pfas_init
                       + self.beta_j
                       + self.k_so3_eaq * c_so3_t
                       + self.k_cl_eaq  * self.c_cl)

        c_eaq = numerator / denominator  # (batch,1)

        rates = [params[f'k{i+1}'] * c_eaq * y_vars[i] for i in range(7)]

        dy1 = -rates[0]
        dy2 = rates[0] - rates[1]
        dy3 = rates[1] - rates[2]
        dy4 = rates[2] - rates[3]
        dy5 = rates[3] - rates[4]
        dy6 = rates[4] - rates[5]
        dy7 = rates[5] - rates[6]
        dy8 = 2.0 * tf.add_n(rates)  # (batch,1)

        return tf.concat([dy1, dy2, dy3, dy4, dy5, dy6, dy7, dy8], axis=-1)

    def generation_of_eaq(self, c_so3_t):
        """
        Returns (batch,1) tensor. TF-only math, depends on c_so3_t.
        """
        # Total absorption (batch,1) because c_so3_t is (batch,1)
        Sigma_f_185 = (self.eps_h2o_185 * self.c_h2o +
                       self.eps_oh_185  * self.c_oh_m +
                       self.eps_cl_185  * self.c_cl +
                       self.eps_so3_185 * c_so3_t +
                       self.eps_pfas_185 * self.c_pfas_init)

        Sigma_f_254 = (self.eps_h2o_254 * self.c_h2o +
                       self.eps_so3_254 * c_so3_t +
                       self.eps_pfas_254 * self.c_pfas_init)

        # Fractions
        f_h2o_185  = (self.eps_h2o_185 * self.c_h2o) / Sigma_f_185
        f_oh_185   = (self.eps_oh_185  * self.c_oh_m) / Sigma_f_185
        f_cl_185   = (self.eps_cl_185  * self.c_cl) / Sigma_f_185
        f_so3_185  = (self.eps_so3_185 * c_so3_t) / Sigma_f_185
        f_so3_254  = (self.eps_so3_254 * c_so3_t) / Sigma_f_254

        def one_minus_10pow(eps, c):
            return 1.0 - tf.pow(10.0, -eps * self.l * c)

        term_h2o_185 = f_h2o_185 * self.phi_h2o_185 * one_minus_10pow(self.eps_h2o_185, self.c_h2o)
        term_oh_185  = f_oh_185  * self.phi_oh_185  * one_minus_10pow(self.eps_oh_185,  self.c_oh_m)
        term_cl_185  = f_cl_185  * self.phi_cl_185  * one_minus_10pow(self.eps_cl_185,  self.c_cl)
        term_so3_185 = f_so3_185 * self.phi_so3_185 * one_minus_10pow(self.eps_so3_185, c_so3_t)

        numerator_185 = self.I0_185 * (term_h2o_185 + term_oh_185 + term_cl_185 + term_so3_185)
        numerator_254 = self.I0_254 * f_so3_254 * self.phi_so3_254 * one_minus_10pow(self.eps_so3_254, c_so3_t)

        return numerator_185 + numerator_254  # (batch,1)