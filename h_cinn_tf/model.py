# cinn_pfas/model.py

# Creates a RNN model based on the RK4 cell and custom loss function

import tensorflow as tf
from tensorflow.keras.layers import RNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from h_cinn_tf.integrator import RungeKuttaIntegratorCell
from h_cinn_tf.loss import create_loss_fn

def create_model(k1, k2, k3, k4, k5, k6, k7, c_eaq, dt, initial_state, batch_input_shape, t_pinn, t_true):

    # 1. RK4 cell based on integrator.py
    rk_cell = RungeKuttaIntegratorCell(k1, k2, k3, k4, k5, k6, k7, c_eaq, dt, initial_state) 

    # 2. Build RNN from the RK4 cell
    rnn_layer = RNN(cell=rk_cell, batch_input_shape=batch_input_shape, return_sequences=True)
    model = Sequential([rnn_layer])

    # 3. Loss function from loss.py
    loss_fn = create_loss_fn(t_pinn, t_true)

    # 4. Decaying learning rate
    lr_schedule = PiecewiseConstantDecay(boundaries=[70, 150, 250], values=[5e-2, 1e-2, 1e-3, 1e-4])

    # 5. Set optimizer and loss func for model
    model.compile(optimizer=RMSprop(learning_rate=lr_schedule), loss=loss_fn)
    
    return model
