"""
Refine the optimal c_so3 using 1D golden-section search on [0.0024, 0.0028].
Relies on the same cost function as so3_init_optimizer.py but evaluates it
without batching (one model build per point).
"""

import sys
import math
from pathlib import Path

import numpy as np
import tensorflow as tf
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# Reuse utilities and constants from the coarse sweep script
from predictor.so3_init_optimizer import (
    DT,
    T_FINAL,
    PFAS_THRESHOLD,
    THRESHOLD_SMOOTHING,
    W_TIME,
    W_SO3,
    build_model_from_config,
)

# Search interval discovered from coarse sweep
SO3_LO = 0.0024
SO3_HI = 0.0028
TOL = 1e-6
MAX_ITERS = 100


def make_sim_context():
    project_root = Path(__file__).resolve().parents[1]
    cfg_dir = project_root / "config"
    trained_k_yaml = cfg_dir / "trained_params.yaml"

    t_sim = np.arange(0.0, T_FINAL, DT, dtype=np.float32)
    dummy = tf.zeros((1, t_sim.size, 1), dtype=tf.float32)
    return cfg_dir, trained_k_yaml, t_sim, dummy


def eval_cost(c_so3_value, cfg_dir, trained_k_yaml, t_sim, dummy):
    # Build the model for this specific c_so3 value (cannot batch)
    model, initial_states = build_model_from_config(
        cfg_dir, trained_k_yaml, t_sim, DT, float(c_so3_value)
    )
    y_pred = model([dummy, initial_states], training=False)
    total_pfas = tf.reduce_sum(y_pred[:, :, :7], axis=-1)
    above = tf.nn.sigmoid((total_pfas - PFAS_THRESHOLD) / THRESHOLD_SMOOTHING)
    time_above = float(tf.reduce_sum(above) * DT)
    cost = float(W_TIME * time_above + W_SO3 * c_so3_value)
    return cost, time_above


def golden_section_search(f, a, b, tol=TOL, max_iters=MAX_ITERS):
    phi = (1 + math.sqrt(5)) / 2
    resphi = 2 - phi  # 1/phi^2

    c = b - resphi * (b - a)
    d = a + resphi * (b - a)
    fc = f(c)
    fd = f(d)
    history = [(c, fc), (d, fd)]

    for _ in range(max_iters):
        if abs(b - a) < tol:
            break
        if fc[0] < fd[0]:
            b, fd = d, fc
            d = c
            c = b - resphi * (b - a)
            fc = f(c)
            history.append((c, fc))
        else:
            a, fc = c, fd
            c = d
            d = a + resphi * (b - a)
            fd = f(d)
            history.append((d, fd))

    mid = (a + b) / 2
    fmid = f(mid)
    history.append((mid, fmid))
    return mid, fmid, history


def main():
    cfg_dir, trained_k_yaml, t_sim, dummy = make_sim_context()

    def wrapped_cost(c):
        cost, time_above = eval_cost(c, cfg_dir, trained_k_yaml, t_sim, dummy)
        return cost, time_above

    best_c, (best_cost, best_time_above), history = golden_section_search(
        wrapped_cost, SO3_LO, SO3_HI
    )

    print(f"Search interval: [{SO3_LO}, {SO3_HI}]")
    print(f"Converged best_c_so3 = {best_c:.8f}")
    print(f"best_cost = {best_cost:.8e}")
    print(f"time_above = {best_time_above:.8e} s")
    print("\nTrace (c_so3, cost, time_above):")
    for c_val, (c_cost, c_time) in history:
        print(f"{c_val:.8f}, {c_cost:.8e}, {c_time:.8e}")


if __name__ == "__main__":
    main()
