# loss.py
import tensorflow as tf

def interpolate_predictions(t_pinn, t_true, y_pred):
    """
    Interpolate y_pred (batch=1) at experimental time points t_true.
    t_pinn: (T_sim,) array. y_pred: (1, T_sim, D).
    Returns: (1, len(t_true), D).
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

@tf.autograph.experimental.do_not_convert
def create_loss_fn_multi(t_pinn_list, t_true_list):
    def my_loss_fn(y_true, y_pred):
        # y_true, y_pred: (batch, T_exp_max, D)
        losses = []
        eps = 1e-8
        batch = len(t_true_list)
        for i in range(batch):
            T_exp = t_true_list[i].shape[0]
            y_true_i = y_true[i:i+1, :T_exp, :]
            T_sim_i = len(t_pinn_list[i])
            y_pred_i = y_pred[i:i+1, :T_sim_i, :]
            y_pred_interp = interpolate_predictions(t_pinn_list[i], t_true_list[i], y_pred_i)
            y_pred_interp.set_shape(y_true_i.shape)

            y_max = tf.reduce_max(tf.abs(y_true_i), axis=[0,1]) + eps
            max_y_max = tf.reduce_max(y_max)
            weights = max_y_max / y_max
            coeff = 1.0 / (max_y_max + eps)

            state_losses = tf.reduce_mean(tf.square(y_true_i - y_pred_interp), axis=[0,1])
            weighted_loss = tf.reduce_sum(weights * state_losses)
            scaled = 10.0 * (coeff ** 2) * weighted_loss
            losses.append(scaled)
        return tf.reduce_mean(tf.stack(losses))
    return my_loss_fn
