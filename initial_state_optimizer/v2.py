import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
import yaml

# -----------------------------
# Settings
# -----------------------------
DT = 5.0
T_FINAL = 1200.0
PFAS_THRESHOLD = 1e-10
PFAS_THRESHOLD_FRACTION = 0.95

# Efficiency constraint: must reach threshold within this time
T_MAX = 800.0   # <-- choose your constraint

# Cost weights
W_TIME = 0.01
W_SO3  = 69.3

# Search bounds
SO3_MIN = 0.0
SO3_MAX = 0.01

# Grid/refinement controls
COARSE_POINTS = 30
REFINE_LEVELS = 4
REFINE_POINTS = 40

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Models_Multiple_Scripts.E_TF_MultipleBatch_Adaptive_c.model import create_model


def load_trained_k(path):
    d = yaml.safe_load(open(path, "r"))
    keys = [f"k{i}" for i in range(1, 8)]
    return np.array([d[k] for k in keys], dtype=np.float32)

def load_constants(cfg_dir):
    return yaml.safe_load(open(Path(cfg_dir) / "physichal_paramters.yaml", "r"))

def load_initials(cfg_dir):
    d = yaml.safe_load(open(Path(cfg_dir) / "initial_conditions.yaml", "r"))
    c_cl = d.get("c_cl", d.get("c_cl_0"))
    c_so3 = d.get("c_so3", d.get("c_so3_0"))
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

    model = create_model(*k, constants, c_cl, float(c_so3_value), pH, dt,
                         initial_states, t_pinn_list, t_true_list,
                         for_prediction=True)
    model.trainable = False
    return model, initial_states

def eval_candidate(c_so3_value, cfg_dir, trained_k_yaml, t_sim, dummy):
    model, initial_states = build_model_from_config(cfg_dir, trained_k_yaml, t_sim, DT, c_so3_value)

    y_pred = model([dummy, initial_states], training=False)   # (1, N, 8)
    total_pfas = tf.reduce_sum(y_pred[:, :, :7], axis=-1)[0]  

    total_pfas_0 = total_pfas[0]
    remaining_fraction = 1.0 - PFAS_THRESHOLD_FRACTION  # e.g. 0.05 if fraction=0.95 degraded
    pfas_threshold = remaining_fraction * total_pfas_0

    # Find first index where total_pfas <= threshold
    hit = tf.where(total_pfas <= pfas_threshold)
    if tf.size(hit) == 0:
        t_reach = np.inf
    else:
        k_hit = int(hit[0, 0].numpy())
        t_reach = float(t_sim[k_hit])

    feasible = (t_reach <= T_MAX)
    # electricity cost ∝ time (you can decide if it’s t_reach or T_MAX or time-on)
    # Here: assume lamps run until threshold is reached (or infeasible -> big penalty)
    if feasible:
        cost = W_TIME * t_reach + W_SO3 * float(c_so3_value)
    else:
        cost = np.inf

    return cost, t_reach, feasible

def grid_search_interval(a, b, num_points, cfg_dir, trained_k_yaml, t_sim, dummy):
    grid = np.linspace(a, b, num_points, dtype=np.float32)

    best = (np.inf, None, np.inf, False)  # (cost, c, t_reach, feasible)
    history = []

    for c in grid:
        cost, t_reach, feas = eval_candidate(float(c), cfg_dir, trained_k_yaml, t_sim, dummy)
        history.append((float(c), cost, t_reach, feas))
        if cost < best[0]:
            best = (cost, float(c), t_reach, feas)

    return best, np.array(history, dtype=object)

def refine_around_best(best_c, a, b, level, cfg_dir, trained_k_yaml, t_sim, dummy):
    # Window shrinks each level; you can tune the shrink factor
    width = (b - a) / (2 ** (level + 1))
    ra = max(SO3_MIN, best_c - width)
    rb = min(SO3_MAX, best_c + width)
    return ra, rb

def main():
    project_root = Path(__file__).resolve().parents[1]
    cfg_dir = project_root / "config"
    trained_k_yaml = cfg_dir / "trained_params.yaml"

    t_sim = np.arange(0.0, T_FINAL + DT, DT, dtype=np.float32)
    dummy = tf.zeros((1, t_sim.size, 1), dtype=tf.float32)

    # ---- coarse ----
    (best_cost, best_c, best_t, best_feas), hist0 = grid_search_interval(
        SO3_MIN, SO3_MAX, COARSE_POINTS, cfg_dir, trained_k_yaml, t_sim, dummy
    )

    print(f"Coarse best: c_so3={best_c:.8f}, cost={best_cost:.3e}, t_reach={best_t:.2f}, feasible={best_feas}")

    # ---- refinements ----
    a, b = SO3_MIN, SO3_MAX
    for level in range(REFINE_LEVELS):
        a, b = refine_around_best(best_c, a, b, level, cfg_dir, trained_k_yaml, t_sim, dummy)

        (best_cost, best_c, best_t, best_feas), hist = grid_search_interval(
            a, b, REFINE_POINTS, cfg_dir, trained_k_yaml, t_sim, dummy
        )
        step = (b - a) / (REFINE_POINTS - 1)
        print(f"Refine {level+1}: interval=[{a:.8f},{b:.8f}] step≈{step:.2e} "
              f"best c_so3={best_c:.8f}, cost={best_cost:.3e}, t_reach={best_t:.2f}")

    print("\nFINAL:")
    print(f"c_so3* = {best_c:.10f}")
    print(f"cost*  = {best_cost:.6e}")
    print(f"t_reach= {best_t:.2f} s (constraint T_MAX={T_MAX:.2f} s)")

if __name__ == "__main__":
    main()
