# train.py (robust paths)
import os, sys, time, yaml
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# --- Robust import of create_model (package or flat file) ---
try:
    from .model import create_model  # if part of a package
except Exception:
    # Fallback: add this file's folder to sys.path and import absolute
    _here = Path(__file__).resolve().parent
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))
    from model import create_model

# Optional: enable memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass

# Simulation step for fine grid
DT_SIM = 1.0

def _find_project_root(start: Path) -> Path:
    """
    Walk up from `start` to find the first directory that contains a 'config' folder.
    If none found up to filesystem root, return `start`.
    """
    cur = start
    for p in [cur] + list(cur.parents):
        if (p / "config").is_dir():
            return p
    return start

def _load_yaml_params(cfg_dir: Path):
    # Tolerate common filename typos
    phys_candidates = [
        cfg_dir / "physichal_paramters.yaml",  # legacy misspelling
        cfg_dir / "physical_parameters.yaml",  # preferred
        cfg_dir / "physical_paramters.yaml",   # another typo
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

def display_trained_parameters(model):
    # Find the first RNN layer that has a 'cell'
    rk_cell = None
    for lyr in model.layers:
        if hasattr(lyr, "cell"):
            rk_cell = lyr.cell
            break
    if rk_cell is None:
        print("Could not find RK cell.")
        return
    trained_params = {name: float(10.0 ** rk_cell.log_k_values[name].numpy())
                      for name in rk_cell.log_k_values}
    print("Trained parameters:")
    for name, value in trained_params.items():
        print(f"{name}: {value:.6e}")
    return trained_params

def main():
    # --- Resolve project root, config, data, output dirs ---
    here = Path(__file__).resolve().parent
    project_root = _find_project_root(here)
    cfg_dir = project_root / "config"
    data_dir = project_root / "data"
    results_dir = project_root / "results"
    ckpt_dir = project_root / "checkpoints"
    results_dir.mkdir(exist_ok=True, parents=True)
    ckpt_dir.mkdir(exist_ok=True, parents=True)

    # --- Load YAML configs ---
    params, init_vals = _load_yaml_params(cfg_dir)

    pH   = float(init_vals["pH"])
    c_cl = float(init_vals["c_cl"])
    c_so3= float(init_vals["c_so3"])

    # Initial k guesses
    k_vals = [5.259223e+06, 5.175076e+08, 5.232472e+08, 5.551598e+08, 5.748630e+08, 5.647780e+08, 5.005279e+08]
    k1, k2, k3, k4, k5, k6, k7 = [np.float32(v) for v in k_vals]

    # --- Load data ---
    batch_csv = data_dir / "Batch_PFAS_data.csv"
    if not batch_csv.exists():
        raise FileNotFoundError(f"Data file not found: {batch_csv}")
    df = pd.read_csv(batch_csv)  # columns: sequence_id, time (s), PFAS...
    groups = df.groupby("sequence_id")
    columns = ["C7F15COO-", "C5F11COO-", "C3F7COO-", "C2F5COO-", "CF3COO-"]

    t_true_list, y_true_list, initial_states_list = [], [], []
    for seq_id, group in groups:
        g = group.sort_values(by="time (s)")
        t_seq = g["time (s)"].values
        y_seq = g[columns].values
        t_true_list.append(t_seq)
        y_true_list.append(y_seq)

        init_state = np.zeros((8,), dtype=np.float32)
        init_state[0] = y_seq[0, 0]  # first PFAS species initial concentration
        initial_states_list.append(init_state)

    batch_size = len(t_true_list)

    # --- Build per-sequence fine simulation grids ---
    t_pinn_list, T_sim_list = [], []
    for t_seq in t_true_list:
        t_sim = np.arange(t_seq[0], t_seq[-1] + DT_SIM, DT_SIM)
        t_pinn_list.append(t_sim)
        T_sim_list.append(len(t_sim))
    T_sim_max = max(T_sim_list)

    # Dummy input (time channel not used by the cell)
    dummy_input = np.zeros((batch_size, T_sim_max, 1), dtype=np.float32)
    dummy_input_tf = tf.convert_to_tensor(dummy_input, dtype=tf.float32)

    # Pad experimental outputs to common length
    T_exp_list = [len(t) for t in t_true_list]
    T_exp_max = max(T_exp_list)
    y_true_padded = []
    for y_seq in y_true_list:
        pad_len = T_exp_max - y_seq.shape[0]
        y_seq_pad = np.pad(y_seq, ((0, pad_len), (0, 0)), mode='constant', constant_values=0.0)
        y_true_padded.append(y_seq_pad)
    y_train = tf.convert_to_tensor(np.stack(y_true_padded, axis=0), dtype=tf.float32)

    # Initial states tensor
    initial_states = tf.convert_to_tensor(np.stack(initial_states_list, axis=0), dtype=tf.float32)

    # Dataset (one big batch or adjust as needed)
    dataset = tf.data.Dataset.from_tensor_slices(((dummy_input_tf, initial_states), y_train))
    dataset = dataset.batch(batch_size)

    # --- Build model (for training: for_prediction=False -> outputs 5 PFAS species) ---
    model = create_model(k1, k2, k3, k4, k5, k6, k7,
                         params, c_cl, c_so3, pH, DT_SIM,
                         initial_states, t_pinn_list, t_true_list,
                         for_prediction=False)

    # --- Train ---
    print("Predicting before training...")
    _ = model.predict([dummy_input_tf, initial_states], verbose=0)

    print("Training...")
    start = time.time()
    model.fit(dataset, epochs=200, verbose=1)
    print(f"Training Time: {time.time() - start:.2f} s")

    # Save weights
    model.save_weights(str(ckpt_dir / "pinn_model.weights.h5"))

    # Predict after training
    y_pred = model.predict([dummy_input_tf, initial_states], verbose=0)  # (batch, T_sim_max, 5)

    # --- Plot first two sequences (if available) ---
    plt.figure(figsize=(9, 6))
    labels = ['C7F15COO-', 'C5F11COO-', 'C3F7COO-', 'C2F5COO-', 'CF3COO-']
    for i in range(len(labels)):
        plt.subplot(3, 2, i+1)
        if batch_size >= 1:
            plt.scatter(t_true_list[0], y_true_list[0][:, i], color='gray', label='Raw Data seq 0', marker='o', alpha=0.7)
            plt.plot(t_pinn_list[0], y_pred[0, :len(t_pinn_list[0]), i], label='Pred seq 0')
        if batch_size >= 2:
            plt.scatter(t_true_list[1], y_true_list[1][:, i], color='lightgray', label='Raw Data seq 1', marker='x', alpha=0.7)
            plt.plot(t_pinn_list[1], y_pred[1, :len(t_pinn_list[1]), i], label='Pred seq 1')
        plt.xlabel('Time (s)')
        plt.ylabel(labels[i])
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    plt.savefig(str(results_dir / "MulipleBatch_fixedD.png"), dpi=600, bbox_inches='tight')
    plt.close()

    # --- Dump trained params next to config ---
    trained_params = display_trained_parameters(model)
    if trained_params is not None:
        with open(cfg_dir / "trained_params.yaml", "w") as f:
            yaml.dump(trained_params, f, default_flow_style=False)

if __name__ == "__main__":
    main()
