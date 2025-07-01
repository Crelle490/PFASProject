# cinn_pfas/loss.py

import tensorflow as tf
from h_cinn_tf.integrator import interpolate_predictions

def create_loss_fn(t_pinn, t_true):
    def my_loss_fn(y_true, y_pred):
        # Interpolate predictions to match experimental time points
        y_pred_interp = interpolate_predictions(t_pinn, t_true, y_pred)
        y_pred_interp.set_shape(y_true.shape)

        # Compute normalization and weighting factors
        y_max = tf.reduce_max(tf.abs(y_true), axis=[0, 1])
        max_y_max = tf.reduce_max(y_max)
        weights = max_y_max / y_max
        coefficient = 1.0 / max_y_max

        # Compute state-wise losses and apply weights
        state_losses = tf.reduce_mean(tf.square(y_true - y_pred_interp), axis=[0, 1])
        weighted_loss = tf.reduce_sum(weights * state_losses)
        scaled_loss = 10 * coefficient ** 2 * weighted_loss
        return scaled_loss
    return my_loss_fn
