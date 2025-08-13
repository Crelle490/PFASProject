# cinn_pfas/train.py
# Inside 'Models Multiple Scripts': python -m A_TF_SingleBatch_Fixed_c.train

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from A_TF_SingleBatch_Fixed_c.model import create_model


def main():
    # Load experimental data (adjust the path as needed)
    data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'PFAS_data.csv')
    data_path = os.path.abspath(data_path)  # Optional: make it absolute
    df = pd.read_csv(data_path)

    # Time series
    t_true = df['time (s)'].values # experimental time series
    t_pinn = np.arange(t_true[0], np.round(t_true[-1]), 1.0) # predict time series with same span 

    # Extract selective specimens
    y_train = df[['C7F15COO-', 'C5F11COO-', 'C3F7COO-', 'C2F5COO-', 'CF3COO-']].values[np.newaxis, :, :]
    batch_input = t_pinn[np.newaxis, :, np.newaxis] # shape: (1, time_steps, 1) = (batch, time, features=time_value)

    # Define kinetic parameters and initial state
    k_values = [5.259223e+06, 5.175076e+08, 5.232472e+08, 5.551598e+08, 5.748630e+08, 5.647780e+08, 5.005279e+08] # Initial guess
    k1, k2, k3, k4, k5, k6, k7 = [np.float32(v) for v in k_values]
    c_eaq = np.float32(3.69e-12) # hard-coded hydrated electrons
    initial_state = np.array([[1.96e-07, 0, 0, 0, 0, 0, 0, 0]], dtype='float32') # IC

    # Create and compile the model
    model = create_model(k1, k2, k3, k4, k5, k6, k7, c_eaq, 1.0, initial_state, batch_input.shape, t_pinn, t_true)

    # Predict before training for comparison
    y_pred_before = model.predict(batch_input)

    # Train the model
    start_time = time.time()
    model.fit(batch_input, y_train, epochs=200, steps_per_epoch=1, verbose=1)
    end_time = time.time()
    print(f"Training Time: {end_time - start_time:.2f} seconds")

    # Predict after training
    y_pred = model.predict(batch_input)

    # Plot the results
    outputs_to_plot = [0, 1, 2, 3, 4]
    labels = ['Output y0', 'Output y1', 'Output y2', 'Output y3', 'Output y4']
    plt.figure(figsize=(9, 6))
    num_plots = len(outputs_to_plot)
    rows = (num_plots + 1) // 2

    for idx, (i, label) in enumerate(zip(outputs_to_plot, labels), start=1):
        plt.subplot(rows, 2, idx)
        plt.scatter(t_true, y_train[0, :, i], color='gray', label='Raw Data', marker='o', alpha=0.7)
        plt.plot(t_pinn, y_pred_before[0, :, i], color='r', label='Before Training')
        plt.plot(t_pinn, y_pred[0, :, i], 'b', label='After Training')
        plt.xlabel('Time (s)')
        plt.ylabel(label)
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    results_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
    os.makedirs(results_dir, exist_ok=True)  # Ensure folder exists
    output_path = os.path.join(results_dir, 'Aminimal_model.png')
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.show()

    # Extract and print the trained kinetic parameters
    # find the RNN layer robustly
    rnn_layer = next((lyr for lyr in model.layers if hasattr(lyr, "cell")), None)
    if rnn_layer is None:
        raise RuntimeError(
            f"Could not find an RNN layer with a `.cell`. Layers: {[type(l).__name__ for l in model.layers]}"
        )

    rk_cell = rnn_layer.cell
    trained_params = {
        name: float(10.0 ** rk_cell.log_k_values[name].numpy())
        for name in rk_cell.log_k_values
    }
    print("Trained parameters:")
    for name, value in trained_params.items():
        print(f"{name}: {value:.6e}") # log --> linear



if __name__ == "__main__":
    main()
