
import sys
import numpy as np
from pathlib import Path
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from predictor.ode_runtime import build_model_from_config

dt = 1.0
horizon_steps = int(2000/dt)

project_root = Path(__file__).resolve().parents[1]
cfg_dir = project_root / "config"
trained_k_yaml = cfg_dir / "trained_params.yaml"

t_sim = np.arange(horizon_steps, dtype=np.float32) * dt
model, dummy, initial_states = build_model_from_config(
    cfg_dir=cfg_dir,
    trained_k_yaml=trained_k_yaml,
    t_sim=t_sim,
    dt=dt,
)
T_sim_max = len(t_sim)
dummy = np.zeros((1, T_sim_max, 1), dtype=np.float32)
model.predict([dummy, initial_states])  # test call to ensure model works