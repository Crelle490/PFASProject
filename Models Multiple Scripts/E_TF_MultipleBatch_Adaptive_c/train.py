# train.py
# Robust path resolution; trains multi-sequence model and saves outputs
# run python -m E_TF_MultipleBatch_Adaptive_c.train inside the 'Models Multiple Scripts' folder

import os, sys, time, yaml
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# Robust import whether run as a package or a flat script
try:
    from .model import create_model
except Exception:
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    from model import create_model

# Optional GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass

DT_SIM = 1.0

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

def display_trained_parameters(model):
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
    here = Path(__file__).resolve().parent
    root = find_project_root(here)
    cfg_dir = root / "config"
    data_dir = root / "data"
    results_dir = root / "results"
    ckpt_dir = root / "checkpoints"
    results_dir.mkdir(exist_ok=True, parents=True)
    ckpt_dir.mkdir(exist_ok=True, parents=True)

    params, init_vals = load_yaml_params(cfg_dir)
    pH = float(init_vals["pH"])
    c_cl = float(init_vals["c_cl"])
    c_so3 = float(init_vals["c_so3"])

    # Initial k guesses
    k_vals = [5.259223e+06, 5.175076e+08, 5.232472e+08, 5.551598e+08, 5.748630e+08, 5.647780e+08, 5.005279e+08]
    k1, k2, k3, k4, k5, k6, k7 = [np.float32(v) for v in k_vals]

    # Data
    batch_csv = data_dir / "Batch_PFAS_data.csv"
    if not batch_csv.exists():
        raise FileNotFoundError(f"Missing data file: {batch_csv}")
    df = pd.read_csv(batch_csv)
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
        init_state[0] = y_seq[0, 0]
        initial_states_list.append(init_state)

    batch_size = len(t_true_list)

    # Fine-grained sim grids
    t_pinn_list, T_sim_list = [], []
    for t_seq in t_true_list:
        t_sim = np.arange(t_seq[0], t_seq[-1] + DT_SIM, DT_SIM)
        t_pinn_list.append(t_sim)
        T_sim_list.append(len(t_sim))
    T_sim_max = max(T_sim_list)

    # Dummy input (time channel not used)
    dummy_input = np.zeros((batch_size, T_sim_max, 1), dtype=np.float32)
    dummy_input_tf = tf.convert_to_tensor(dummy_input, dtype=tf.float32)

    # Pad experimental outputs
    T_exp_max = max(len(t) for t in t_true_list)
    y_true_padded = []
    for y_seq in y_true_list:
        pad_len = T_exp_max - y_seq.shape[0]
        y_seq_pad = np.pad(y_seq, ((0, pad_len), (0, 0)), mode='constant', constant_values=0.0)
        y_true_padded.append(y_seq_pad)
    y_train = tf.convert_to_tensor(np.stack(y_true_padded, axis=0), dtype=tf.float32)

    # Initial states
    initial_states = tf.convert_to_tensor(np.stack(initial_states_list, axis=0), dtype=tf.float32)

    # Dataset
    dataset = tf.data.Dataset.from_tensor_slices(((dummy_input_tf, initial_states), y_train)).batch(batch_size)

    # Model
    model = create_model(k1, k2, k3, k4, k5, k6, k7,
                         params, c_cl, c_so3, pH, DT_SIM,
                         initial_states, t_pinn_list, t_true_list,
                         for_prediction=False)

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

    # Plot first two sequences
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
    plt.savefig(str(results_dir / "EAdaptive_Multibatch.png"), dpi=600, bbox_inches='tight')
    plt.close()

    # Dump trained params
    trained_params = display_trained_parameters(model)
    if trained_params is not None:
        with open(cfg_dir / "trained_params.yaml", "w") as f:
            yaml.dump(trained_params, f, default_flow_style=False)

if __name__ == "__main__":
    main()
