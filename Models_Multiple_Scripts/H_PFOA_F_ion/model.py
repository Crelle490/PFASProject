# model.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import RNN, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay

try:
    from .integrator import RungeKuttaIntegratorCell
    from .loss import create_loss_fn_multi
except Exception:
    from integrator import RungeKuttaIntegratorCell
    from loss import create_loss_fn_multi

def create_model(k1, k2, k3, k4, k5, k6, k7, betaj, k_cl, k_so3,
                 constants,
                 c_cl_default, c_so3_default, pH_default,
                 dt_sim,
                 initial_states, t_pinn_list, t_true_list,
                 for_prediction=False,
                 exo_dim=4,
                 nF=15,
                 output_mode="defluorination_pct",
                 exo_map=None):

    # Dummy initial state used only for build-time fallback
    if hasattr(initial_states, "numpy"):
        dummy_initial_state = np.asarray(initial_states[0:1].numpy(), dtype=np.float32)
    else:
        dummy_initial_state = np.asarray(initial_states[:1], dtype=np.float32)

    rk_cell = RungeKuttaIntegratorCell(
        k1, k2, k3, k4, k5, k6, k7, betaj, k_cl, k_so3,
        constants,
        c_cl_default, c_so3_default, pH_default,
        dt_sim,
        dummy_initial_state,
        for_prediction=for_prediction,
        nF=nF,
        exo_map=exo_map,
        output_mode=output_mode,
    )

    T_sim_max = max(len(t) for t in t_pinn_list)

    exo_in = Input(shape=(T_sim_max, exo_dim), name="exo_input")   # u(t)
    init_in = Input(shape=(8,), name="initial_states")

    outputs = RNN(rk_cell, return_sequences=True)(exo_in, initial_state=[init_in])
    model = Model(inputs=[exo_in, init_in], outputs=outputs)

    loss_fn = create_loss_fn_multi(t_pinn_list, t_true_list)
    lr = PiecewiseConstantDecay(boundaries=[70, 150, 250, 700],
                                values=[1e-2, 5e-3, 2e-3, 2e-3, 1e-3])
    if not(for_prediction):
        model.compile(optimizer=RMSprop(learning_rate=lr), loss=loss_fn)
        
    return model
