# model.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import RNN, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay

from Models_Multiple_Scripts.E_TF_MultipleBatch_Adaptive_c.integrator import RungeKuttaIntegratorCell
from Models_Multiple_Scripts.E_TF_MultipleBatch_Adaptive_c.loss import create_loss_fn_multi 

class PINNModel(tf.keras.Model):
    def __init__(self, integrator_cell):
        super().__init__()
        self.rnn = RNN(integrator_cell, return_sequences=True)

    def call(self, inputs):
        dummy_input, initial_states = inputs
        return self.rnn(dummy_input, initial_state=[initial_states])

def create_model(k1, k2, k3, k4, k5, k6, k7,
                 constants, c_cl, c_so3, pH, dt_sim,
                 initial_states, t_pinn_list, t_true_list,
                 for_prediction=False):
        # Build cell using first initial state as dummy
        if hasattr(initial_states, "numpy"):
            dummy_initial_state = np.asarray(initial_states[0:1].numpy(), dtype=np.float32)
        else:
            dummy_initial_state = np.asarray(initial_states[:1], dtype=np.float32)

        intensity = 1.0  # default intensity for prediction

        rk_cell = RungeKuttaIntegratorCell(
            k1, k2, k3, k4, k5, k6, k7,
            constants, c_cl, c_so3, pH, intensity, dt_sim,
            dummy_initial_state, for_prediction=for_prediction
        )

        T_sim_max = max(len(t) for t in t_pinn_list)

        dummy_input = Input(shape=(T_sim_max, 1), name="dummy_input")
        init_in = Input(shape=(8,), name="initial_states")

        outputs = RNN(rk_cell, return_sequences=True)(dummy_input, initial_state=[init_in])
        model = Model(inputs=[dummy_input, init_in], outputs=outputs)

        loss_fn = create_loss_fn_multi(t_pinn_list, t_true_list)
        lr = PiecewiseConstantDecay(boundaries=[70, 150, 250],
                                    values=[5e-2, 1e-2, 1e-3, 1e-4])
        model.compile(optimizer=RMSprop(learning_rate=lr), loss=loss_fn)
        return model
