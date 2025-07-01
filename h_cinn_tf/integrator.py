# cinn_pfas/integrator.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer

class RungeKuttaIntegratorCell(Layer):
    def __init__(self, k1, k2, k3, k4, k5, k6, k7, c_eaq, dt, initial_state, **kwargs):
        super().__init__(**kwargs)
        self.c_eaq = c_eaq
        self.dt = dt
        self.initial_state = initial_state
        self.state_size = 8
        # Use logarithmic representation for parameters
        self.log_k1, self.log_k2, self.log_k3, self.log_k4, self.log_k5, self.log_k6, self.log_k7 = [
            np.log10(k1), np.log10(k2), np.log10(k3), np.log10(k4), np.log10(k5), np.log10(k6), np.log10(k7)
        ]

    def build(self, input_shape):
        self.log_k_values = {
            name: self.add_weight(
                name=f'log_{name}', shape=(),
                initializer=tf.constant_initializer(value),
                trainable=True
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

        # Select outputs (e.g., specific states)
        y_1 = y_next[:, 0:1]
        y_3 = y_next[:, 2:3]
        y_5 = y_next[:, 4:5]
        y_6 = y_next[:, 5:6]
        y_7 = y_next[:, 6:7]
        output = tf.concat([y_1, y_3, y_5, y_6, y_7], axis=-1)
        return output, [y_next]

    def _fun(self, y, params):
        y_vars = [y[:, i:i + 1] for i in range(7)]
        k_vars = [params[f'k{i + 1}'] for i in range(7)]
        rates = [k * self.c_eaq * y_var for k, y_var in zip(k_vars, y_vars)]

        dy1 = -rates[0]
        dy2 = rates[0] - rates[1]
        dy3 = rates[1] - rates[2]
        dy4 = rates[2] - rates[3]
        dy5 = rates[3] - rates[4]
        dy6 = rates[4] - rates[5]
        dy7 = rates[5] - rates[6]
        dy8 = 2 * tf.reduce_sum(rates, axis=0)
        return tf.concat([dy1, dy2, dy3, dy4, dy5, dy6, dy7, dy8], axis=-1)

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return self.initial_state

def interpolate_predictions(t_pinn, t_true, y_pred):
    """Interpolate y_pred at t_true time points."""
    t_pinn_tensor = tf.constant(t_pinn, dtype=tf.float32)
    t_true_tensor = tf.constant(t_true, dtype=tf.float32)
    indices = tf.searchsorted(t_pinn_tensor, t_true_tensor, side='left') - 1
    indices = tf.clip_by_value(indices, 0, len(t_pinn_tensor) - 2)
    t0 = tf.gather(t_pinn_tensor, indices)
    t1 = tf.gather(t_pinn_tensor, indices + 1)
    y0 = tf.gather(y_pred, indices, axis=1)
    y1 = tf.gather(y_pred, indices + 1, axis=1)
    w = (t_true_tensor - t0) / (t1 - t0)
    w = tf.reshape(w, [1, -1, 1])
    y_pred_interp = y0 + w * (y1 - y0)
    return y_pred_interp
