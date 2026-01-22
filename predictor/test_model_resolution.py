import sys
import numpy as np
from pathlib import Path
import tensorflow as tf
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from predictor.ode_runtime import build_model_from_config

# --------------------------------------------------
# Settings
# --------------------------------------------------
DT_LIST = [1.0, 10.0, 50.0, 100.0]
DT_REF = 1.0
T_FINAL = 2000.0   # physical time horizon [s]
EPS = 1e-12        # numerical safety for relative error

cfg_dir = PROJECT_ROOT / "config"
trained_k_yaml = cfg_dir / "trained_params.yaml"

CHANNEL_NAMES = [
    r"$C_{7F15}$",
    r"$C_{6F13}$",
    r"$C_{5F11}$",
    r"$C_{4F9}$",
    r"$C_{3F7}$",
    r"$C_{2F5}$",
    r"$CF_3$",
    r"$F^-$",
]

# --------------------------------------------------
# Storage for all rollouts
# Each entry: dt -> (t_plot, y_plot)
# where y_plot includes the true initial condition at t=0
# --------------------------------------------------
results = {}

for dt in DT_LIST:
    horizon_steps = int(T_FINAL / dt)

    # RNN length: outputs correspond to x(dt), x(2dt), ..., x(T*dt)
    t_sim = np.arange(horizon_steps, dtype=np.float32) * dt
    T_sim_max = len(t_sim)
    batch_size = 1

    print(f"\n--- Running dt = {dt:.1f} s | steps = {horizon_steps} ---")

    model, dummy_from_builder, initial_states = build_model_from_config(
        cfg_dir=cfg_dir,
        trained_k_yaml=trained_k_yaml,
        t_sim=t_sim,
        dt=dt,
    )

    # Dummy input (time-axis driver). If your cell expects catalysts, change last dim to 2.
    dummy_input = np.zeros((batch_size, T_sim_max, 1), dtype=np.float32)
    dummy_input_tf = tf.convert_to_tensor(dummy_input, dtype=tf.float32)

    # Forward rollout
    y_pred = model.predict([dummy_input_tf, initial_states], verbose=0)[0]  # (T, n_out)

    # --------------------------------------------------
    # Prepend the *true* initial condition at t=0
    # --------------------------------------------------
    x0_full = np.asarray(initial_states, dtype=np.float32)[0]  # (8,)

    if y_pred.shape[1] == 8:
        y_plot = np.vstack([x0_full, y_pred])  # (T+1, 8)
    else:
        # training-mode output: subset [0,2,4,5,6]
        idx = [0, 2, 4, 5, 6]
        y_plot = np.vstack([x0_full[idx], y_pred])  # (T+1, 5)

    t_plot = np.arange(y_plot.shape[0], dtype=np.float32) * dt  # 0, dt, 2dt, ..., T*dt

    results[dt] = (t_plot, y_plot)

    print(f"Output shape (raw): {y_pred.shape} | Output shape (with x0): {y_plot.shape}")

# --------------------------------------------------
# Plot 1: trajectories vs time resolution
# --------------------------------------------------
n_out = next(iter(results.values()))[1].shape[1]
n_plot = min(n_out, len(CHANNEL_NAMES))

fig1, axes1 = plt.subplots(n_plot, 1, figsize=(9, 2.2 * n_plot), sharex=True)
if n_plot == 1:
    axes1 = [axes1]

for i in range(n_plot):
    ax = axes1[i]
    for dt, (t_plot, y_plot) in results.items():
        ax.plot(t_plot, y_plot[:, i], label=f"dt = {dt:g} s", linewidth=1.6)

    ax.set_ylabel(CHANNEL_NAMES[i], fontsize=11)
    ax.grid(True)

axes1[-1].set_xlabel("Time [s]", fontsize=11)
axes1[0].legend(loc="best", fontsize=9)
fig1.suptitle("PINN rollout vs. time resolution (includes x0 at t=0)", fontsize=13)
fig1.tight_layout(rect=[0, 0, 1, 0.96])

# --------------------------------------------------
# Plot 2: relative error vs dt = 1 reference
# --------------------------------------------------
t_ref, y_ref = results[DT_REF]

fig2, axes2 = plt.subplots(n_plot, 1, figsize=(9, 2.2 * n_plot), sharex=True)
if n_plot == 1:
    axes2 = [axes2]

for i in range(n_plot):
    ax = axes2[i]

    for dt, (t_plot, y_plot) in results.items():
        if dt == DT_REF:
            continue

        # Interpolate current trajectory onto reference time grid
        y_interp = np.interp(
            t_ref,
            t_plot,
            y_plot[:, i]
        )

        rel_err = np.abs(y_interp - y_ref[:, i]) / (np.abs(y_ref[:, i]) + EPS)

        ax.plot(t_ref, rel_err, label=f"dt = {dt:g} s", linewidth=1.6)

    ax.set_ylabel(f"rel.err {CHANNEL_NAMES[i]}", fontsize=10)
    ax.set_yscale("log")
    ax.grid(True, which="both")

axes2[-1].set_xlabel("Time [s]", fontsize=11)
axes2[0].legend(loc="best", fontsize=9)
fig2.suptitle("Relative error vs dt = 1 s reference (log scale)", fontsize=13)
fig2.tight_layout(rect=[0, 0, 1, 0.96])

plt.show()
