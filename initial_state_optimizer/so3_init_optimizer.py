import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml
import matplotlib.pyplot as plt   # <-- add


# --- Hardcoded simulation/optimization settings ---
DT = 5.0
T_FINAL = 1200.0
PFAS_THRESHOLD = 1e-10
THRESHOLD_SMOOTHING = 1e-9
W_TIME = 0.01
W_SO3 = 69.3 
SO3_MIN = 0
SO3_MAX = 0.01
GRID_POINTS = 30

from helper_functions import load_constants, load_initials, load_trained_k

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Models_Multiple_Scripts.E_TF_MultipleBatch_Adaptive_c.model import create_model



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

    c_so3_grid = np.linspace(SO3_MIN, SO3_MAX, GRID_POINTS, dtype=np.float32)

    best_cost = np.inf
    best_c_so3 = None

    cost_hist = []
    time_above_hist = []

    print(f"Grid searching c_so3 in [{SO3_MIN}, {SO3_MAX}] with {GRID_POINTS} points...")
    for idx, c_so3_value in enumerate(c_so3_grid):
        # Build the model for this specific c_so3 value (no batching support)
        model, initial_states = build_model_from_config(cfg_dir, trained_k_yaml, t_sim, DT, float(c_so3_value))

        y_pred = model([dummy, initial_states], training=False)
        
        total_pfas = tf.reduce_sum(y_pred[:, :, :-1], axis=-1)[0]
        
        # Find first index where PFAS is below threshold
        hits = tf.where(total_pfas <= PFAS_THRESHOLD)

        if tf.size(hits) == 0:
            # Threshold never reached
            time_above = np.inf
        else:
            k_hit = int(hits[0, 0].numpy())
            
            time_above = float(k_hit * DT)
        cost = float(W_TIME * time_above + W_SO3 * c_so3_value)

        cost_hist.append(cost)
        time_above_hist.append(time_above)

        if cost < best_cost:
            best_cost = cost
            best_c_so3 = float(c_so3_value)

        print(
            f"idx={idx:03d} c_so3={c_so3_value:.6e} "
            f"time_above={time_above:.6e} cost={cost:.6e}"
        )

    print(f"best_c_so3={best_c_so3:.6e} best_cost={best_cost:.6e}")

    # ---- plot sweep curves ----
    plt.figure()
    plt.plot(c_so3_grid, cost_hist, marker="o")
    plt.xlabel("c_so3 [M]")
    plt.ylabel("Cost")
    plt.grid(True)
    plt.tight_layout()

    plt.figure()
    plt.plot(c_so3_grid, time_above_hist, marker="o")
    plt.xlabel("c_so3 [M]")
    plt.ylabel("Time above PFAS threshold [s]")
    plt.grid(True)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
