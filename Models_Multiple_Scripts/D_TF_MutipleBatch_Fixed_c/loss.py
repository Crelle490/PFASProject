# loss.py
# Loss for multi-sequence training: interpolate each sequence to its own experimental grid,
# apply dynamic per-output weighting, then average across sequences.

import tensorflow as tf
from .integrator import interpolate_predictions

@tf.autograph.experimental.do_not_convert
def create_loss_fn_multi(t_pinn_list, t_true_list):
    def my_loss_fn(y_true, y_pred):
        # y_true, y_pred: (batch, T_max, D)
        losses = []
        eps = 1e-8
        batch = len(t_true_list)
        for i in range(batch):
            T_exp = t_true_list[i].shape[0]
            # Take unpadded experimental data for seq i
            y_true_i = y_true[i:i+1, :T_exp, :]
            # Take model predictions for seq i on its sim grid length
            T_sim_i = len(t_pinn_list[i])
            y_pred_i = y_pred[i:i+1, :T_sim_i, :]
            # Interpolate
            y_pred_interp = interpolate_predictions(t_pinn_list[i], t_true_list[i], y_pred_i)
            y_pred_interp.set_shape(y_true_i.shape)

            # Per-output dynamic weights
            y_max = tf.reduce_max(tf.abs(y_true_i), axis=[0,1]) + eps
            max_y_max = tf.reduce_max(y_max)
            weights = max_y_max / y_max
            coeff = 1.0 / (max_y_max + eps)

            state_losses = tf.reduce_mean(tf.square(y_true_i - y_pred_interp), axis=[0,1])
            weighted_loss = tf.reduce_sum(weights * state_losses)
            scaled_loss = 10.0 * (coeff ** 2) * weighted_loss
            losses.append(scaled_loss)

        return tf.reduce_mean(tf.stack(losses))
    return my_loss_fn
