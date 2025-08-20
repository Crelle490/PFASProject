# h_cinn_tf/model.py  (make sure the path/module name matches your imports)

import tensorflow as tf
from tensorflow.keras.layers import RNN, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from A_TF_SingleBatch_Fixed_c.integrator import RungeKuttaIntegratorCell
from A_TF_SingleBatch_Fixed_c.loss import create_loss_fn

def create_model(k1, k2, k3, k4, k5, k6, k7, c_eaq, dt, initial_state, batch_input_shape, t_pinn, t_true):
    # 1) RK4 cell
    rk_cell = RungeKuttaIntegratorCell(k1, k2, k3, k4, k5, k6, k7, c_eaq, dt, initial_state)

    # 2) Define inputs via Keras Input (sequence length flexible)
    # batch_input_shape is like (batch, timesteps, features); we only need the feature dim
    n_features = int(batch_input_shape[-1])
    inputs = Input(shape=(None, n_features))  # (timesteps, features)

    # 3) RNN over inputs
    outputs = RNN(rk_cell, return_sequences=True)(inputs)

    # 4) Build & compile model
    model = Model(inputs, outputs)

    loss_fn = create_loss_fn(t_pinn, t_true)
    lr_schedule = PiecewiseConstantDecay(boundaries=[70, 150, 250],
                                         values=[5e-2, 1e-2, 1e-3, 1e-4])
    model.compile(optimizer=RMSprop(learning_rate=lr_schedule), loss=loss_fn)
    return model
