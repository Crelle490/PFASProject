# cinn_pfas/model.py
# Build/compile the model from RK4 cell and custom loss

from tensorflow.keras.layers import RNN, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from .integrator import RungeKuttaIntegratorCell
from .loss import create_loss_fn

def create_model(k1, k2, k3, k4, k5, k6, k7, constants,
                 c_cl, c_so3, pH, dt, initial_state, batch_input_shape,
                 t_pinn, t_true):
    # RK cell
    rk_cell = RungeKuttaIntegratorCell(k1, k2, k3, k4, k5, k6, k7,
                                       constants, c_cl, c_so3, pH, dt, initial_state)

    # Inputs: (timesteps, features) — time values flow through but are unused by the cell
    n_features = int(batch_input_shape[-1])
    inputs = Input(shape=(None, n_features))

    # Recurrent integration across the time grid
    outputs = RNN(rk_cell, return_sequences=True)(inputs)

    # Build + compile
    model = Model(inputs, outputs)
    loss_fn = create_loss_fn(t_pinn, t_true)
    lr_schedule = PiecewiseConstantDecay(boundaries=[70, 150, 250],
                                         values=[5e-2, 1e-2, 1e-3, 1e-4])
    model.compile(optimizer=RMSprop(learning_rate=lr_schedule), loss=loss_fn)
    return model
