import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml


# --- Hardcoded simulation/optimization settings ---
DT = 1.0
T_FINAL = 600.0
PFAS_THRESHOLD = 1e-9
THRESHOLD_SMOOTHING = 1e-9
W_TIME = 1.0
W_SO3 = 1.0
SO3_MIN = 0.0
SO3_MAX = 0.05
OPT_STEPS = 10
LEARNING_RATE = 1e-2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Models_Multiple_Scripts.D_TF_MutipleBatch_Fixed_c.model import create_model



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


def build_model_from_config(cfg_dir, trained_k_yaml, t_sim, dt, c_so3_value):
    constants = load_constants(cfg_dir)
    pH, c_cl, _c_so3_init, c_pfas_init = load_initials(cfg_dir)
    k = load_trained_k(trained_k_yaml)

    t_sim = np.asarray(t_sim, dtype=np.float32)
    t_pinn_list = [t_sim]
    t_true_list = [t_sim[:1]]

    initial_states = np.zeros((1, 8), np.float32)
    initial_states[0, 0] = np.float32(c_pfas_init)
    initial_states = tf.convert_to_tensor(initial_states)

    model = create_model(*k, constants, c_cl, c_so3_value, pH, dt,
                         initial_states, t_pinn_list, t_true_list,
                         for_prediction=True)
    model.trainable = False
    return model, initial_states


def main():
    project_root = Path(__file__).resolve().parents[1]
    cfg_dir = project_root / "config"
    trained_k_yaml = cfg_dir / "trained_params.yaml"

    t_sim = np.arange(0.0, T_FINAL, DT, dtype=np.float32)
    dummy = tf.zeros((1, t_sim.size, 1), dtype=tf.float32)

    c_so3_unconstrained = tf.Variable(0.0, dtype=tf.float32)
    c_so3 = SO3_MIN + (SO3_MAX - SO3_MIN) * tf.nn.sigmoid(c_so3_unconstrained)

    model, initial_states = build_model_from_config(cfg_dir, trained_k_yaml, t_sim, DT, c_so3)

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    best_cost = None
    best_c_so3 = None

    print(f"Optimizing c_so3 in [{SO3_MIN}, {SO3_MAX}] for {OPT_STEPS} steps...")
    for step in range(OPT_STEPS):
        with tf.GradientTape() as tape:
            c_so3 = SO3_MIN + (SO3_MAX - SO3_MIN) * tf.nn.sigmoid(c_so3_unconstrained)
            y_pred = model([dummy, initial_states], training=False)
            total_pfas = tf.reduce_sum(y_pred[:, :, :7], axis=-1)
            above = tf.nn.sigmoid((total_pfas - PFAS_THRESHOLD) / THRESHOLD_SMOOTHING)
            time_above = tf.reduce_sum(above) * DT
            cost = W_TIME * time_above + W_SO3 * c_so3

        grads = tape.gradient(cost, [c_so3_unconstrained])
        optimizer.apply_gradients(zip(grads, [c_so3_unconstrained]))

        cost_val = float(cost.numpy())
        c_so3_val = float(c_so3.numpy())
        if best_cost is None or cost_val < best_cost:
            best_cost = cost_val
            best_c_so3 = c_so3_val

        if step % 1 == 0 or step == OPT_STEPS - 1:
            time_above_val = float(time_above.numpy())
            print(
                f"step={step:04d} cost={cost_val:.6e} "
                f"time_above={time_above_val:.6e} c_so3={c_so3_val:.6e}"
            )

    print(f"best_c_so3={best_c_so3:.6e} best_cost={best_cost:.6e}")


if __name__ == "__main__":
    main()
