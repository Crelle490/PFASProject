# cinn_pfas/loss.py

# Computes error between experimental data and predicted concentration:
# - Normalizes per-species errors by their maximum observed magnitudes to balance contributions between species of varying scales
# - Scales the overall loss to keep it numerically stable for optimization.
# - Returns a weighted, scaled mean squared error loss for training.

import tensorflow as tf
from h_cinn_tf.integrator import interpolate_predictions

def create_loss_fn(t_pinn, t_true):
    def my_loss_fn(y_true, y_pred):
        # Interpolate predictions to match experimental time points
        y_pred_interp = interpolate_predictions(t_pinn, t_true, y_pred)
        y_pred_interp.set_shape(y_true.shape)

        # Compute normalization and weighting factors
        y_max = tf.reduce_max(tf.abs(y_true), axis=[0, 1]) # max concentration for all species
        max_y_max = tf.reduce_max(y_max) # max concentration of all species at all times
        weights = max_y_max / y_max # scales each state inversely proportional to its max value
        coefficient = 1.0 / max_y_max # global scaling to keep loss within reasonable range

        # Compute state-wise losses and apply weights
        state_losses = tf.reduce_mean(tf.square(y_true - y_pred_interp), axis=[0, 1]) # MSE
        weighted_loss = tf.reduce_sum(weights * state_losses) # single loss 
        scaled_loss = 10 * coefficient ** 2 * weighted_loss
        return scaled_loss
    return my_loss_fn
