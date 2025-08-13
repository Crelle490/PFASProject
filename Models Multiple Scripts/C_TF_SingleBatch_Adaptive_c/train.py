# cinn_pfas/train.py
# Entry point that mirrors your "working single-file" script but imports the split modules.
# cinn_pfas/train.py

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path   # <-- add this

from .model import create_model

def _project_root() -> Path:
    """PFASProject root: two levels up from this file (…/PFASProject)."""
    return Path(__file__).resolve().parents[2]

def plot_results(t_true, t_pinn, y_train, y_pred_before, y_pred_after):
    outputs_to_plot = [0, 1, 2, 3, 4]
    labels = ['Output y0', 'Output y1', 'Output y2', 'Output y3', 'Output y4']

    plt.figure(figsize=(9, 6))
    num_plots = len(outputs_to_plot)
    rows = (num_plots + 1) // 2  # two-column layout

    for idx, (i, label) in enumerate(zip(outputs_to_plot, labels), start=1):
        plt.subplot(rows, 2, idx)
        plt.scatter(t_true, y_train[0, :, i], color='gray', label='Raw Data', marker='o', alpha=0.7)
        plt.plot(t_pinn, y_pred_before[0, :, i], label='Before Training')
        plt.plot(t_pinn, y_pred_after[0, :, i], label='After Training')
        plt.xlabel('Time (s)')
        plt.ylabel(label)
        plt.grid(True)
        plt.legend()

    plt.tight_layout()

    # --- SAVE TO PFASProject/results ---
    results_dir = _project_root() / "results"   # <-- absolute path at repo root
    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / "SingleBatch_AdaptiveC.png"

    plt.savefig(out, dpi=600, bbox_inches='tight')   # save BEFORE show
    print(f"Saved plot to: {out}")
    plt.show()


def _find_rnn_cell(model):
    """Return the first Keras RNN cell found (depth-first)."""
    stack = [model]
    while stack:
        m = stack.pop()
        for layer in getattr(m, "layers", []):
            # Direct RNN layer
            if hasattr(layer, "cell"):
                return layer.cell
            # Nested models / wrappers
            if hasattr(layer, "layers"):
                stack.append(layer)
    raise RuntimeError(
        f"No RNN layer with a `.cell` found. Layers seen: {[type(l).__name__ for l in model.layers]}"
    )

def display_trained_parameters(model):
    rk_cell = _find_rnn_cell(model)
    trained_params = {
        name: float(10.0 ** rk_cell.log_k_values[name].numpy())
        for name in rk_cell.log_k_values
    }
    print("Trained parameters:")
    for name, value in trained_params.items():
        print(f"{name}: {value:.6e}")


def main():
    # Reaction rate constants (initial guesses)
    k_values = [5.259223e+06, 5.175076e+08, 5.232472e+08,
                5.551598e+08, 5.748630e+08, 5.647780e+08, 5.005279e+08]
    k1, k2, k3, k4, k5, k6, k7 = [np.float32(v) for v in k_values]

    # Physical & experimental parameters
    params = {
        "l": 0.24,  # cm
        "I0_185": 2.07e-6,
        "I0_254": 5.19e-4,
        "c_h2o": 55.6,
        "epsilon_h2o_185": 0.0324,
        "phi_h2o_185": 0.045,
        "epsilon_h2o_254": 0.032,
        "phi_h2o_254": 0.0,
        "epsilon_oh_m_185": 3200.0,
        "phi_oh_m_185": 0.11,
        "epsilon_cl_185": 3540.0,
        "phi_cl_185": 0.43,
        "epsilon_so3_185": 3729.5,
        "phi_so3_185": 0.85,
        "epsilon_so3_254": 21.22,
        "phi_so3_254": 0.11,
        "epsilon_pfas_185": 2689.5,
        "epsilon_pfas_254": 28.8,
    }

    pH = 5.7
    c_cl = 0.0
    c_so3 = 0.0

    # Data
    data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'PFAS_data.csv')
    data_path = os.path.abspath(data_path)  # Optional: make it absolute
    df = pd.read_csv(data_path)
    t_true = df['time (s)'].values
    t_pinn = np.arange(t_true[0], np.round(t_true[-1]), 1.0)

    # Targets (select 5 outputs as in working script)
    y_train = df[['C7F15COO-', 'C5F11COO-', 'C3F7COO-', 'C2F5COO-', 'CF3COO-']].values
    y_train = y_train[np.newaxis, :, :]  # [batch, time, outputs]
    batch_input = t_pinn[np.newaxis, :, np.newaxis]  # [batch, time, 1]

    # Initial state for RK cell (8 species)
    initial_state = np.array([[df['C7F15COO-'][0], 0, 0, 0, 0, 0, 0, 0]], dtype='float32')

    # Build model
    model = create_model(k1, k2, k3, k4, k5, k6, k7,
                         params, c_cl, c_so3, pH, dt=1.0,
                         initial_state=initial_state,
                         batch_input_shape=batch_input.shape,
                         t_pinn=t_pinn, t_true=t_true)

    # Predict before training
    y_pred_before = model.predict(batch_input)

    # Train
    start = time.time()
    model.fit(batch_input, y_train, epochs=200, steps_per_epoch=1, verbose=1)
    print(f"Training Time: {time.time() - start:.2f} seconds")

    # Predict after training
    y_pred_after = model.predict(batch_input)

    # Plot & report
    plot_results(t_true, t_pinn, y_train, y_pred_before, y_pred_after)
    display_trained_parameters(model)

if __name__ == '__main__':
    main()
