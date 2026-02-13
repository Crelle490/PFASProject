# train.py
# Multi-batch training for defluorination (%F-) data.
# Each batch (sequence_id) has constant inputs: catalyst concentrations and pH, read from CSV.

import os, sys, time, yaml
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

try:
    from .model import create_model
except Exception:
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    from model import create_model

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass

DT_SIM = 5.0

def find_project_root(start: Path) -> Path:
    for p in [start] + list(start.parents):
        if (p / "config").is_dir():
            return p
    return start

def load_yaml_params(cfg_dir: Path):
    phys_candidates = [
        cfg_dir / "physichal_paramters.yaml",
        cfg_dir / "physical_parameters.yaml",
        cfg_dir / "physical_paramters.yaml",
    ]
    phys_path = next((p for p in phys_candidates if p.exists()), None)
    if phys_path is None:
        raise FileNotFoundError(f"Could not find any of: {', '.join(str(p) for p in phys_candidates)}")
    with open(phys_path, "r") as f:
        params = yaml.safe_load(f)

    init_path = cfg_dir / "initial_conditions.yaml"
    if not init_path.exists():
        raise FileNotFoundError(f"Missing required file: {init_path}")
    with open(init_path, "r") as f:
        init_vals = yaml.safe_load(f)
    return params, init_vals

# def display_trained_parameters(model):
#     rk_cell = None
#     for lyr in model.layers:
#         if hasattr(lyr, "cell"):
#             rk_cell = lyr.cell
#             break
#     if rk_cell is None:
#         print("Could not find RK cell.")
#         return
#     trained_params = {name: float(10.0 ** rk_cell.log_k_values[name].numpy())
#                       for name in rk_cell.log_k_values}
#     print("Trained parameters:")
#     for name, value in trained_params.items():
#         print(f"{name}: {value:.6e}")
#     return trained_params

def display_trained_parameters(model):
    rk_cell = None
    for lyr in model.layers:
        if hasattr(lyr, "cell"):
            rk_cell = lyr.cell
            break
    if rk_cell is None:
        print("Could not find RK cell.")
        return None

    out = {}

    # 1) log10-space kinetic params stored in log_k_values
    if hasattr(rk_cell, "log_k_values"):
        for name, var in rk_cell.log_k_values.items():
            out[name] = float((10.0 ** var.numpy()))

    # 2) extra log10 parameters (convert 10**)
    extra_log10 = ["log_Kh_mM", "log_gamma_scale", "log_krec"]
    #extra_log10 = []
    for name in extra_log10:
        if hasattr(rk_cell, name):
            out[name.replace("log_", "")] = float((10.0 ** getattr(rk_cell, name).numpy()))

    # 3) linear parameters (store raw)
    linear_params = ["theta_other", "theta1", "theta2", "phi0", "phi1", "phi2"]
    #linear_params = []
    for name in linear_params:
        if hasattr(rk_cell, name):
            out[name] = float(getattr(rk_cell, name).numpy())

    print("Trained parameters:")
    for k, v in out.items():
        print(f"{k}: {v:.6e}")

    return out

def _first_existing_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def main():
    here = Path(__file__).resolve().parent
    root = find_project_root(here)
    cfg_dir = root / "config"
    data_dir = root / "data"
    results_dir = root / "results"
    ckpt_dir = root / "checkpoints"
    results_dir.mkdir(exist_ok=True, parents=True)
    ckpt_dir.mkdir(exist_ok=True, parents=True)

    params, init_vals = load_yaml_params(cfg_dir)

    # Fallback defaults (used if CSV doesn't include these)
    pH_default = float(init_vals.get("pH", 7.0))
    c_cl_default = float(init_vals.get("c_cl_0", 0.0))
    c_so3_default = float(init_vals.get("c_so3_0", 0.0))

    # PFOA defluorination: 15 F atoms per molecule
    nF = int(init_vals.get("nF", 15))

    # Initial k guesses (keep your old values; you can later tune)
    k_vals = [5.64e+08, 5.69e+08, 2.04e+08, 5.30e+08,
              3.00e+08, 2.95e+08, 1.71e+08, 1.57e4, 1.0e6, 1.5e6]
    k1, k2, k3, k4, k5, k6, k7, betaj, k_cl, k_so3 = [np.float32(v) for v in k_vals]

    # Data CSV (defluorination)
    candidates = [
        data_dir / "Batch_Defluorination_data_formatted.csv",
        here / "Batch_Defluorination_data_formatted.csv",
        root / "Batch_Defluorination_data_formatted.csv",
    ]
    batch_csv = next((p for p in candidates if p.exists() and p.is_file()), None)
    if batch_csv is None:
        raise FileNotFoundError(f"Missing data file. Tried: {', '.join(str(p) for p in candidates)}")
    print(f"Using training data: {batch_csv}")

    df = pd.read_csv(batch_csv)
    if "sequence_id" not in df.columns or "time (s)" not in df.columns:
        raise ValueError("CSV must contain columns: 'sequence_id' and 'time (s)'.")

    # Identify defluorination column in percent/fraction
    f_col = _first_existing_col(df, ["F_pct", "F_percent", "defluorination_pct", "F-", "F"])
    if f_col is None:
        raise ValueError("Could not find defluorination column. Expected one of: "
                         "F_pct, F_percent, defluorination_pct, F-, F")

    # Per-batch constant inputs from CSV (preferred) or YAML defaults
    col_c_cl  = _first_existing_col(df, ["c_cl_0_M", "c_cl_0", "c_cl"])
    col_c_so3 = _first_existing_col(df, ["c_so3_0_M", "c_so3_0", "c_so3"])
    col_pH    = _first_existing_col(df, ["pH", "ph"])

    # Need initial PFOA concentration per batch for scaling.
    # If config defines c_pfoa0_M/c_pfoa_0_M, use it as a global override.
    col_c0 = _first_existing_col(df, ["c_pfoa0_M", "c_pfoa_0_M", "c_pfoa0", "c_pfoa_0",
                                     "c_pfas0_M", "c_pfas_0_M", "C0_PFOA", "PFOA0"])
    c0_forced = None
    if "c_pfoa0_M" in init_vals:
        c0_forced = float(init_vals["c_pfoa0_M"])
    elif "c_pfoa_0_M" in init_vals:
        c0_forced = float(init_vals["c_pfoa_0_M"])

    if c0_forced is None and col_c0 is None:
        raise ValueError("Missing initial PFOA concentration per batch. Please add a constant column like "
                         "'c_pfoa0_M' (M) to your CSV for each sequence_id), or set c_pfoa0_M in initial_conditions.yaml.")

    groups = df.groupby("sequence_id")

    t_true_list, y_true_list, initial_states_list = [], [], []
    exo_constants = []  # per batch: [c_cl, c_so3, pH, c_pfoa0]

    for seq_id, group in groups:
        g = group.sort_values(by="time (s)")
        t_seq = g["time (s)"].to_numpy(dtype=np.float32)

        f_vals = g[f_col].to_numpy(dtype=np.float32)
        f_max = np.nanmax(f_vals)
        if f_max <= 1.5:
            y_pct = 100.0 * f_vals
        else:
            y_pct = f_vals
        y_seq = y_pct.reshape(-1, 1)

        c_cl  = float(g[col_c_cl].iloc[0])  if col_c_cl  else c_cl_default
        c_so3 = float(g[col_c_so3].iloc[0]) if col_c_so3 else c_so3_default
        pH    = float(g[col_pH].iloc[0])    if col_pH    else pH_default

        if c0_forced is not None:
            c_pfoa0 = float(c0_forced)
        elif col_c0 is not None:
            c_pfoa0 = float(g[col_c0].iloc[0])
        else:
            raise ValueError("No valid source for c_pfoa0 after input checks.")

        t_true_list.append(t_seq)
        y_true_list.append(y_seq)
        exo_constants.append([c_cl, c_so3, pH, c_pfoa0])

        init_state = np.zeros((8,), dtype=np.float32)
        init_state[0] = c_pfoa0  # PFOA initial
        init_state[7] = 0.0      # F- initial
        initial_states_list.append(init_state)

    batch_size = len(t_true_list)

    # Fine-grained sim grids
    t_pinn_list, T_sim_list = [], []
    for t_seq in t_true_list:
        t_sim = np.arange(float(t_seq[0]), float(t_seq[-1]) + DT_SIM, DT_SIM, dtype=np.float32)
        t_pinn_list.append(t_sim)
        T_sim_list.append(len(t_sim))
    T_sim_max = max(T_sim_list)

    # Build exogenous input tensor: repeat constants along time
    exo_dim = 4
    exo_padded = []
    for i in range(batch_size):
        u0 = np.array(exo_constants[i], dtype=np.float32)
        T_i = len(t_pinn_list[i])
        exo_i = np.tile(u0[None, :], (T_i, 1))
        pad_len = T_sim_max - T_i
        if pad_len > 0:
            exo_i = np.vstack([exo_i, np.tile(u0[None, :], (pad_len, 1))])
        exo_padded.append(exo_i)

    exo_input_tf = tf.convert_to_tensor(np.stack(exo_padded, axis=0), dtype=tf.float32)

    # Pad experimental outputs
    T_exp_max = max(len(t) for t in t_true_list)
    y_true_padded = []
    for y_seq in y_true_list:
        pad_len = T_exp_max - y_seq.shape[0]
        y_seq_pad = np.pad(y_seq, ((0, pad_len), (0, 0)), mode='constant', constant_values=0.0)
        y_true_padded.append(y_seq_pad)
    y_train = tf.convert_to_tensor(np.stack(y_true_padded, axis=0), dtype=tf.float32)

    initial_states = tf.convert_to_tensor(np.stack(initial_states_list, axis=0), dtype=tf.float32)

    dataset = tf.data.Dataset.from_tensor_slices(((exo_input_tf, initial_states), y_train)).batch(batch_size)

    # Model: output defluorination percentage
    model = create_model(
        k1, k2, k3, k4, k5, k6, k7, betaj, k_cl, k_so3,
        params,
        c_cl_default, c_so3_default, pH_default,
        DT_SIM,
        initial_states, t_pinn_list, t_true_list,
        for_prediction=False,
        exo_dim=exo_dim,
        nF=nF,
        output_mode="defluorination_pct",
        exo_map={"c_cl":0, "c_so3":1, "pH":2, "c_pfoa0":3}
    )

    print("Predicting before training...")
    _ = model.predict([exo_input_tf, initial_states], verbose=0)

    print("Training...")
    start = time.time()
    model.fit(dataset, epochs=1000, verbose=1)
    print(f"Training Time: {time.time() - start:.2f} s")

    model.save_weights(str(ckpt_dir / "pinn_model.weights.h5"))

    y_pred = model.predict([exo_input_tf, initial_states], verbose=0)  # (batch, T_sim_max, 1)

    # Plot: predicted vs measured defluorination (%)
    plt.figure(figsize=(8, 5))
    for i in range(min(batch_size, 6)):
        plt.scatter(t_true_list[i], y_true_list[i][:, 0], s=18, alpha=0.85, label=f"data seq {i}")
        plt.plot(t_pinn_list[i], y_pred[i, :len(t_pinn_list[i]), 0], linewidth=1.3, label=f"pred seq {i}")
    plt.xlabel("Time (s)")
    plt.ylabel("Defluorination (%)")
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(str(results_dir / "Defluorination_fit.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # ============================================================
    # (B) NEW: subplots, one subplot per batch
    # ============================================================
    n_show = min(batch_size, 6)  # show first 6 by default
    ncols = 3
    nrows = int(np.ceil(n_show / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 3.2 * nrows), sharex=False, sharey=True)
    axes = np.array(axes).reshape(-1)  # flatten safely for nrows/ncols=1 cases

    for ax_i in range(nrows * ncols):
        ax = axes[ax_i]
        if ax_i >= n_show:
            ax.axis("off")
            continue

        # data + prediction for this sequence
        ax.scatter(t_true_list[ax_i], y_true_list[ax_i][:, 0], s=18, alpha=0.85, label="data")
        ax.plot(t_pinn_list[ax_i], y_pred[ax_i, :len(t_pinn_list[ax_i]), 0], linewidth=1.5, label="pred")

        ax.set_title(f"Sequence {ax_i}", fontsize=10)
        ax.grid(True)
        ax.set_xlabel("Time (s)")
        if ax_i % ncols == 0:
            ax.set_ylabel("Defluorination (%)")
        ax.legend(fontsize=8, loc="best")

    plt.tight_layout()
    plt.savefig(str(results_dir / "Defluorination_fit_subplots.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    trained_params = display_trained_parameters(model)
    if trained_params is not None:
        with open(cfg_dir / "trained_params.yaml", "w") as f:
            yaml.dump(trained_params, f, default_flow_style=False)

if __name__ == "__main__":
    main()
