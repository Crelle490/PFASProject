# ode_runtime.py (minimal)

# Load the model from 'E_TF_MultipleBatch_Adaptive_c' 
# with trained parameters and IC from 'config' folder.

import numpy as np
import yaml
import tensorflow as tf
import sys
from pathlib import Path

# --- Minimal robust import of create_model ---
try:
    from E_TF_MultipleBatch_Adaptive_c.model import create_model
except Exception:
    here = Path(__file__).resolve().parent
    model_dir = (here / ".." / "Models_Multiple_Scripts" / "E_TF_MultipleBatch_Adaptive_c").resolve()
    if str(model_dir) not in sys.path:
        sys.path.insert(0, str(model_dir))
    from model import create_model  # noqa: E402

def load_trained_k(path):
    d = yaml.safe_load(open(path, "r"))
    keys = [f"k{i}" for i in range(1, 8)]
    return np.array([d[k] for k in keys], dtype=np.float32)

def load_constants(cfg_dir):
    return yaml.safe_load(open(Path(cfg_dir) / "physichal_paramters.yaml", "r"))

def load_initials(cfg_dir):
    d = yaml.safe_load(open(Path(cfg_dir) / "initial_conditions.yaml", "r"))
    c_cl = d.get("c_cl")
    if c_cl is None:
        c_cl = d.get("c_cl_0")
    c_so3 = d.get("c_so3")
    if c_so3 is None:
        c_so3 = d.get("c_so3_0")
    if c_cl is None or c_so3 is None:
        raise KeyError("initial_conditions.yaml must define c_cl/c_so3 or c_cl_0/c_so3_0")
    return float(d["pH"]), float(c_cl), float(c_so3), float(d["c_pfas_init"])

def build_model_from_config(cfg_dir, trained_k_yaml, t_sim, dt=1.0,initial_states=None):
    """
    cfg_dir: folder with physichal_paramters.yaml and initial_conditions.yaml
    trained_k_yaml: path to trained_params.yaml (with k1..k7)
    t_sim: 1D array of times (e.g., np.arange(0, 601, 1, dtype=np.float32))
    """
    constants = load_constants(cfg_dir)
    pH, c_cl, c_so3, c_pfas_init = load_initials(cfg_dir)
    k = load_trained_k(trained_k_yaml)

    t_sim = np.asarray(t_sim, dtype=np.float32)
    t_pinn_list = [t_sim]
    t_true_list = [t_sim[:1]]
    if initial_states is None:              
        initial_states = np.zeros((1, 8), np.float32)
        initial_states[0, 0] = np.float32(c_pfas_init)
    else:
        initial_states = np.asarray(initial_states, dtype=np.float32)
    dummy = np.zeros((1, t_sim.size, 1), np.float32)

    model = create_model(*k, constants, c_cl, c_so3, pH, dt,
                         tf.convert_to_tensor(initial_states),
                         t_pinn_list, t_true_list,
                         for_prediction=True)
    return model, dummy, initial_states
