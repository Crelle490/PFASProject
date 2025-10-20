import numpy as np
import yaml
import tensorflow as tf
import sys
from pathlib import Path
from helper_functions import find_project_root, load_yaml_params, load_yaml_constants

# import system 
try:
    from E_TF_MultipleBatch_Adaptive_c.integrator import RungeKuttaIntegratorCell
except Exception:
    here = Path(__file__).resolve().parent
    model_dir = (here / ".." / "Models_Multiple_Scripts" / "E_TF_MultipleBatch_Adaptive_c").resolve()
    if str(model_dir) not in sys.path:
        sys.path.insert(0, str(model_dir))
    from integrator import RungeKuttaIntegratorCell 


# Find parameters
here = Path(__file__).resolve().parent
root = find_project_root(here)
cfg_dir = root / "config"

# load parameters
params, init_vals = load_yaml_params(cfg_dir)
pH = float(init_vals["pH"])
c_cl_0 = float(init_vals["c_cl"])
c_so3_0 = float(init_vals["c_so3"])
dt_sim = 1.0
k_values = load_yaml_constants(cfg_dir)
k1, k2, k3, k4, k5, k6, k7 = [k_values[f'k{i}'] for i in range(1, 8)]
initial_state = np.array([init_vals["c_pfas_init"], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) 
initial_state = initial_state.reshape((1,1,8)).astype(np.float32) 


# make u function



# integrate
rk_cell = RungeKuttaIntegratorCell(
            k1, k2, k3, k4, k5, k6, k7,
            params, c_cl_0, c_so3_0, pH, dt_sim,
            initial_state.reshape(1,8), for_prediction=False
        )
rk_cell.build(input_shape=initial_state.shape)

states = initial_state

for step in range(10):
     output, states = rk_cell.call(inputs=tf.zeros((1,1,1)), states=states)
     print(f"Step {step+1}, Output: {output.numpy()}, States: {states[0].numpy()}")
     states = np.array(states).astype(np.float32)

print(output)

# plot 