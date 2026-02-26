import csv
from pathlib import Path
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


from so3_init_opt_robustness_new_model import build_model_with_k, set_cell_extra_params, get_cell_param_dict, _get_cell, make_constant_u_traj, prebuild_grid_tensors
from helper_functions import load_initials, load_trained_k, load_constants, load_cell_params

curent_dir = Path(__file__).resolve().parent
root_dir = curent_dir.parent


SO3_POINTS = np.array([0.0, 0.002, 0.004, 0.006, 0.008, 0.010], dtype=np.float32)
PH_POINTS  = np.array([5.75, 6.02, 6.205, 6.286, 6.485, 6.91], dtype=np.float32)

def SO3_to_pH(C_so3):
    C_so3 = np.asarray(C_so3, dtype=np.float32)
    if np.any((C_so3 < SO3_POINTS[0]) | (C_so3 > SO3_POINTS[-1])):
        #raise ValueError(f"C_so3 outside [{SO3_POINTS[0]}, {SO3_POINTS[-1]}]")
        C_so3 = np.asarray(C_so3, dtype=np.float32)
        return np.full_like(C_so3, 6.2, dtype=np.float32)
    return np.interp(C_so3, SO3_POINTS, PH_POINTS).astype(np.float32)

def to_int(x): 
    return int(x.strip())

def to_float(x):
    return float(x.strip().replace(",", "."))  # handles decimal comma too

path = root_dir / "data" / "exp2.csv"

# Each sequence_id maps to lists of measurements (+ constants)
experimental_data = defaultdict(lambda: {
    "time_s": [],
    "defluorination_pct": [],
    "c_pfoa0_M": None,
    "c_so3_0_M": None,
    "c_cl_0_M": None,
    "pH": None,
})

with open(path, newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f, delimiter=",")
    # strip header whitespace if any
    reader.fieldnames = [h.strip() for h in reader.fieldnames]

    for row in reader:
        if not row or row["sequence_id"] is None or row["sequence_id"].strip() == "":
            continue

        seq = to_int(row["sequence_id"])
        exp = experimental_data[seq]

        exp["time_s"].append(to_float(row["time (s)"]))
        exp["defluorination_pct"].append(to_float(row["defluorination_pct"]))

        # constants (same for all rows in a sequence)
        if exp["c_pfoa0_M"] is None:
            exp["c_pfoa0_M"] = to_float(row["c_pfoa0_M"])
            exp["c_so3_0_M"]  = to_float(row["c_so3_0_M"])
            exp["c_cl_0_M"]   = to_float(row["c_cl_0_M"])
            exp["pH"]         = to_float(row["pH"])

experimental_data = dict(experimental_data)
print(experimental_data)

cfg_dir = root_dir / "config"
trained_k_yaml = cfg_dir / "trained_params.yaml"

t_final = []
c_so3 = []
pH_for_exp = []
PFOA_c_init = []
for idx in experimental_data:
    t_final.append(experimental_data[idx]["time_s"][-1])
    c_so3.append(experimental_data[idx]["c_so3_0_M"])
    pH_for_exp.append(float(SO3_to_pH(c_so3[-1])))
    PFOA_c_init.append(experimental_data[idx]["c_pfoa0_M"])

print(t_final)
print(c_so3)
print(pH_for_exp)

dt = 5.0

t_sim = np.arange(0.0, max(t_final), dt, dtype=np.float32)

t_len = t_sim.size

k = load_trained_k(trained_k_yaml)

model, init, (pH, c_cl_init, c_so3_init, c_pfas_init) = build_model_with_k(cfg_dir=cfg_dir, k_vec=k,t_sim=t_sim,dt=dt)
model.trainable = False
cell_params = load_cell_params(cfg_dir / "trained_cell_params.yaml")
set_cell_extra_params(model, cell_params)
cell_nom = get_cell_param_dict(model)
cell = _get_cell(model)

# Warm up compile once (important)
u_warm = make_constant_u_traj(t_len, 0.0, c_cl_init, c_pfas_init)
_ = model([u_warm, init], training=False)


# Prebuild coarse tensors
u, x0, c_tf = prebuild_grid_tensors(
    t_len, c_so3, c_cl=c_cl_init, c_pfoa0=c_pfas_init, initial_states=init
)
y = model([u, x0], training=False)

# --- Ensure consistent ordering everywhere ---
seq_ids = sorted(experimental_data.keys())

# Rebuild these lists in the SAME order as seq_ids (important!)
c_so3 = [experimental_data[seq]["c_so3_0_M"] for seq in seq_ids]
pH_for_exp = [float(SO3_to_pH(cs)) for cs in c_so3]

# --- Run model output to numpy ---
y_np = y.numpy()
if y_np.ndim == 2:
    y_np = y_np[:, :, None]

# --- Choose which simulated channel to plot ---
DEF_IDX = 0  # <-- set correctly for your model (or compute from PFAS as discussed)
import matplotlib as mpl

mpl.rcParams.update({
    "text.usetex": True,

    # Font setup
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "cm",   # LaTeX math font (looks right with TNR)

    # Sizes
    "font.size": 16,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,

    # Improve layout
    "legend.fontsize": 14,
    "figure.titlesize": 16,
})
# --- Create figure with 6 subplots ---
fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
axes = axes.flatten()

for b, seq in enumerate(seq_ids):
    ax = axes[b]
    exp = experimental_data[seq]

    # Experimental
    t_meas = np.asarray(exp["time_s"], dtype=float)
    d_meas = np.asarray(exp["defluorination_pct"], dtype=float)

    # Simulated
    t_model = t_sim[:y_np.shape[1]]
    d_model = y_np[b, :, DEF_IDX]

    # red dots (measured), black line (simulated)
    ax.plot(
        t_meas, d_meas,
        marker="o", linestyle="None",
        color="red", markersize=5,
        label=r"\textbf{Measured}"
    )
    ax.plot(
        t_model, d_model,
        linestyle="-", linewidth=2.0,
        color="black",
        label=r"\textbf{Simulated}"
    )

    # LaTeX title (SO3 and pH)
    ax.set_title(
        rf"$C_{{\mathrm{{SO_3}}}} = {c_so3[b]:.3g}\,\mathrm{{M}},\quad \mathrm{{pH}}\approx {pH_for_exp[b]:.2f}$"
    )
    ax.grid(True)

# Axis labels on outer edges (LaTeX)
for ax in axes[3:]:
    ax.set_xlabel(r"Time~$t$~[s]")
for ax in axes[::3]:
    ax.set_ylabel(r"Defluorination~[\%]")

# Single legend for the whole figure
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2)

fig.tight_layout(rect=[0, 0, 1, 0.93])
plt.show()