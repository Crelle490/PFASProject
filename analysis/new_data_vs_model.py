import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

# -----------------------------
# Project import setup
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from predictor.ode_runtime import build_model_from_config

# -----------------------------
# Read experimental data
# -----------------------------
data_path = PROJECT_ROOT / "data" / "UV_degradation_of_PFAS_SO3.csv"
df = pd.read_csv(data_path, sep=";")
df.columns = df.columns.astype(str).str.strip()

# Robust time column pick
time_candidates = ["Time", "time", "time (s)", "Time (s)", "time(s)", "t", "t (s)"]
time_col = next((c for c in time_candidates if c in df.columns), None)
if time_col is None:
    raise KeyError(f"Could not find a time column. Columns are:\n{df.columns.tolist()}")

t_data = pd.to_numeric(df[time_col], errors="coerce").to_numpy()
if np.any(np.isnan(t_data)):
    raise ValueError(f"Time column '{time_col}' contains non-numeric values after parsing.")

# Ensure increasing time (optional, but often helpful)
# df = df.sort_values(time_col).reset_index(drop=True)
# t_data = df[time_col].to_numpy()

# -----------------------------
# Simulation grid and model
# -----------------------------
t_final = float(t_data[-1])
DT = 1.0

# RNN step times: dt, 2dt, ... (we'll prepend t=0 later)
t_sim = np.arange(0.0, t_final + DT, DT, dtype=np.float32)  # includes 0..t_final
# build_model_from_config in your template expects t_sim length = horizon steps (often without 0)
# We'll pass the "step times" excluding 0 to match the usual rollout convention:
t_sim_steps = t_sim[1:]  # 1,2,...,t_final

cfg_dir = PROJECT_ROOT / "config"
initial_file = "initial_conditions_with_so3.yaml"
trained_k_yaml = cfg_dir / "trained_params.yaml"

model, dummy_input, initial_states = build_model_from_config(
    cfg_dir=cfg_dir,
    trained_k_yaml=trained_k_yaml,
    t_sim=t_sim_steps,
    dt=DT,
    initial_states=None,
    initial_file=initial_file
)

# Forward rollout
y_pred = model.predict([dummy_input, initial_states], verbose=0)[0]  # (T, n_out)

# Prepend x0 at t=0 to align with t_sim = [0,1,2,...]
x0_full = np.asarray(initial_states, dtype=np.float32)[0]  # (n_state,)
if y_pred.shape[1] == x0_full.shape[0]:
    y_plot = np.vstack([x0_full, y_pred])  # (T+1, n_out)
else:
    # If your model outputs a subset in "training mode", pick matching indices here
    # Update idx to match your build_model_from_config training-mode mapping.
    idx = [0, 2, 4, 5, 7]
    y_plot = np.vstack([x0_full[idx], y_pred])

t_plot = np.arange(y_plot.shape[0], dtype=np.float32) * DT  # 0..t_final

print(y_plot.shape)

# -----------------------------
# Map CSV columns to channels (edit if your CSV naming differs)
# -----------------------------
CHANNELS = [
    ("C7F15COO-", r"$C_{7F15}$"),
    ("C6F13COO-", r"$C_{6F13}$"),
    ("C5F11COO-", r"$C_{5F11}$"),
    ("C4F9COO-",  r"$C_{4F9}$"),
    ("C3F7COO-",  r"$C_{3F7}$"),
    ("C2F5COO-",  r"$C_{2F5}$"),
    ("CF3COO-",   r"$CF_3$"),
    ("F-",        r"$F^-$"),
]

# Keep only channels that exist in df
available = [(col, lab) for (col, lab) in CHANNELS if col in df.columns]
if len(available) == 0:
    raise KeyError(
        "None of the expected concentration columns were found.\n"
        f"Expected one of: {[c for c,_ in CHANNELS]}\n"
        f"Got columns: {df.columns.tolist()}"
    )

# Convert data columns to numeric
for col, _ in available:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# -----------------------------
# Plot: model vs data
# -----------------------------
n_plot = min(len(available), y_plot.shape[1])
fig, axes = plt.subplots(n_plot, 1, figsize=(10, 2.4 * n_plot), sharex=True)
if n_plot == 1:
    axes = [axes]


for i in range(n_plot):
    col, lab = available[i]
    ax = axes[i]
    print(i)
    # Data
    ax.plot(t_data, df[col].to_numpy(), marker="o", linestyle="none", markersize=3.5, label=f"data: {col}")

    # Model (channel i assumed aligned with CHANNELS order)
    ax.plot(t_plot, y_plot[:, i], linewidth=1.8, label="model")

    ax.set_ylabel(lab, fontsize=11)
    ax.grid(True)
    ax.legend(loc="best", fontsize=9)

axes[-1].set_xlabel("Time [s]", fontsize=11)
fig.suptitle("Model prediction vs experimental data", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
