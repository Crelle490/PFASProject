# cinn_pfas/train.py
# Inside 'Models Multiple Scripts': python -m B_TO_SingleBatch_Fixed_c.train

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from B_TO_SingleBatch_Fixed_c.model import PINNModel
from B_TO_SingleBatch_Fixed_c.integrator import RungeKuttaIntegratorCell
from B_TO_SingleBatch_Fixed_c.loss import create_loss_fn


def main():
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    # Load experimental data
    data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'PFAS_data.csv')
    data_path = os.path.abspath(data_path)  # Optional: make it absolute
    df = pd.read_csv(data_path)

    t_true = df['time (s)'].values
    t_pinn = np.arange(t_true[0], np.round(t_true[-1]), 1.0)

    # y_train shape: (1, len(t_true), 5)
    y_train_np = df[['C7F15COO-', 'C5F11COO-', 'C3F7COO-', 'C2F5COO-', 'CF3COO-']].values[np.newaxis, :, :]
    y_train = torch.tensor(y_train_np, dtype=torch.float32, device=device)

    # Define kinetic parameters and initial state
    k_values = [5.259223e+06, 5.175076e+08, 5.232472e+08, 5.551598e+08, 5.748630e+08, 5.647780e+08, 5.005279e+08]
    k1, k2, k3, k4, k5, k6, k7 = [np.float32(v) for v in k_values]
    c_eaq = np.float32(3.69e-12)
    initial_state = np.array([[1.96e-07, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32)

    num_steps = len(t_pinn)
    batch_size = y_train.shape[0]

    # Create the integrator cell and PINN model
    integrator_cell = RungeKuttaIntegratorCell(k1, k2, k3, k4, k5, k6, k7, c_eaq, dt=1.0, initial_state=initial_state)
    integrator_cell.to(device)
    model = PINNModel(integrator_cell, num_steps=num_steps).to(device)

    # Create loss function and optimizer
    loss_fn = create_loss_fn(t_pinn, t_true)
    optimizer = optim.RMSprop(model.parameters(), lr=5e-2)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[70, 150, 250], gamma=0.2)

    # Predict before training for comparison
    model.eval()
    with torch.no_grad():
        y_pred_before = model(torch.zeros(batch_size, num_steps, 1, device=device))

    # Training loop
    num_epochs = 150
    model.train()
    start_time = time.time()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        dummy_input = torch.zeros(batch_size, num_steps, 1, device=device)  # Dummy input; not used in integration
        y_pred = model(dummy_input)
        loss = loss_fn(y_pred, y_train)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if epoch % 1 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
    end_time = time.time()
    print(f"Training Time: {end_time - start_time:.2f} seconds")

    # Predict after training
    model.eval()
    with torch.no_grad():
        y_pred = model(torch.zeros(batch_size, num_steps, 1, device=device))

    # Convert predictions to numpy arrays for plotting
    y_pred_before_np = y_pred_before.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()

    # Plot results for each output
    outputs_to_plot = [0, 1, 2, 3, 4]
    labels = ['Output y0', 'Output y1', 'Output y2', 'Output y3', 'Output y4']
    plt.figure(figsize=(9, 6))
    num_plots = len(outputs_to_plot)
    rows = (num_plots + 1) // 2

    for idx, (i, label) in enumerate(zip(outputs_to_plot, labels), start=1):
        plt.subplot(rows, 2, idx)
        plt.scatter(t_true, y_train_np[0, :, i], color='gray', label='Raw Data', marker='o', alpha=0.7)
        plt.plot(t_pinn, y_pred_before_np[0, :, i], color='r', label='Before Training')
        plt.plot(t_pinn, y_pred_np[0, :, i], 'b', label='After Training')
        plt.xlabel('Time (s)')
        plt.ylabel(label)
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    results_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'Results')
    os.makedirs(results_dir, exist_ok=True)  # Ensure folder exists
    output_path = os.path.join(results_dir, 'Bminimal_model.png')

    plt.show()

    # Print the trained kinetic parameters
    trained_params = {
        'k1': torch.pow(10.0, integrator_cell.log_k1).item(),
        'k2': torch.pow(10.0, integrator_cell.log_k2).item(),
        'k3': torch.pow(10.0, integrator_cell.log_k3).item(),
        'k4': torch.pow(10.0, integrator_cell.log_k4).item(),
        'k5': torch.pow(10.0, integrator_cell.log_k5).item(),
        'k6': torch.pow(10.0, integrator_cell.log_k6).item(),
        'k7': torch.pow(10.0, integrator_cell.log_k7).item(),
    }
    print("Trained parameters:")
    for name, value in trained_params.items():
        print(f"{name}: {value:.6e}")


if __name__ == "__main__":
    main()
