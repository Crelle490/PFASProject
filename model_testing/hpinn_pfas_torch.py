import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim


# ---------------------------
# Runge-Kutta Integrator Cell
# ---------------------------
class RungeKuttaIntegratorCell(nn.Module):
    def __init__(self, k1, k2, k3, k4, k5, k6, k7, c_eaq, dt, initial_state):
        super().__init__()
        self.c_eaq = c_eaq
        self.dt = dt
        # Save initial state as a tensor (non-trainable)
        self.register_buffer('initial_state', torch.tensor(initial_state, dtype=torch.float32))
        self.state_size = 8
        # Trainable parameters stored in log10 space
        self.log_k1 = nn.Parameter(torch.tensor(np.log10(k1), dtype=torch.float32))
        self.log_k2 = nn.Parameter(torch.tensor(np.log10(k2), dtype=torch.float32))
        self.log_k3 = nn.Parameter(torch.tensor(np.log10(k3), dtype=torch.float32))
        self.log_k4 = nn.Parameter(torch.tensor(np.log10(k4), dtype=torch.float32))
        self.log_k5 = nn.Parameter(torch.tensor(np.log10(k5), dtype=torch.float32))
        self.log_k6 = nn.Parameter(torch.tensor(np.log10(k6), dtype=torch.float32))
        self.log_k7 = nn.Parameter(torch.tensor(np.log10(k7), dtype=torch.float32))

    def forward(self, y):
        # Use a list of parameters instead of a dictionary with string keys.
        log_ks = [self.log_k1, self.log_k2, self.log_k3, self.log_k4,
                  self.log_k5, self.log_k6, self.log_k7]
        ks = [10 ** p for p in log_ks]

        # 4th order Runge-Kutta integration
        k1_val = self._fun(y, ks) * self.dt
        k2_val = self._fun(y + 0.5 * k1_val, ks) * self.dt
        k3_val = self._fun(y + 0.5 * k2_val, ks) * self.dt
        k4_val = self._fun(y + k3_val, ks) * self.dt
        y_next = y + (k1_val + 2 * k2_val + 2 * k3_val + k4_val) / 6.0

        # Select outputs: columns 0, 2, 4, 5, and 6
        y1 = y_next[:, 0:1]
        y3 = y_next[:, 2:3]
        y5 = y_next[:, 4:5]
        y6 = y_next[:, 5:6]
        y7 = y_next[:, 6:7]
        output = torch.cat([y1, y3, y5, y6, y7], dim=-1)
        return output, y_next

    def _fun(self, y, ks):
        # Extract the first 7 components from y
        y_vars = [y[:, i:i + 1] for i in range(7)]
        # Compute reaction rates using the corresponding kinetic parameter from the list
        rates = [ks[i] * self.c_eaq * y_vars[i] for i in range(7)]
        dy1 = -rates[0]
        dy2 = rates[0] - rates[1]
        dy3 = rates[1] - rates[2]
        dy4 = rates[2] - rates[3]
        dy5 = rates[3] - rates[4]
        dy6 = rates[4] - rates[5]
        dy7 = rates[5] - rates[6]
        dy8 = 2 * sum(rates)
        return torch.cat([dy1, dy2, dy3, dy4, dy5, dy6, dy7, dy8], dim=-1)


# ---------------------------
# Interpolation Function
# ---------------------------
def interpolate_predictions(t_pinn, t_true, y_pred):
    """
    Interpolate y_pred at t_true time points using t_pinn.
    y_pred: Tensor of shape (batch, num_steps, output_dim)
    """
    device = y_pred.device
    t_pinn_tensor = torch.tensor(t_pinn, dtype=torch.float32, device=device)
    t_true_tensor = torch.tensor(t_true, dtype=torch.float32, device=device)
    indices = torch.searchsorted(t_pinn_tensor, t_true_tensor, right=False) - 1
    indices = torch.clamp(indices, 0, len(t_pinn_tensor) - 2)
    t0 = t_pinn_tensor[indices]
    t1 = t_pinn_tensor[indices + 1]

    batch_size = y_pred.shape[0]
    indices_expanded = indices.unsqueeze(0).expand(batch_size, -1)
    y0 = torch.gather(y_pred, 1,
                      indices_expanded.unsqueeze(-1).expand(batch_size, indices_expanded.shape[1], y_pred.shape[2]))
    y1 = torch.gather(y_pred, 1, (indices_expanded + 1).unsqueeze(-1).expand(batch_size, indices_expanded.shape[1],
                                                                             y_pred.shape[2]))
    w = (t_true_tensor - t0) / (t1 - t0)
    w = w.unsqueeze(0).unsqueeze(-1)
    y_pred_interp = y0 + w * (y1 - y0)
    return y_pred_interp


# ---------------------------
# Custom Loss Function
# ---------------------------
def create_loss_fn(t_pinn, t_true):
    def my_loss_fn(y_pred, y_true):
        # Interpolate predictions to experimental time points
        y_pred_interp = interpolate_predictions(t_pinn, t_true, y_pred)
        # Compute maximum absolute values for each output over batch and time
        y_max = torch.max(torch.abs(y_true), dim=1)[0]
        y_max = torch.max(y_max, dim=0)[0]
        max_y_max = torch.max(y_max)
        weights = max_y_max / y_max
        coefficient = 1.0 / max_y_max
        state_losses = torch.mean((y_true - y_pred_interp) ** 2, dim=(0, 1))
        weighted_loss = torch.sum(weights * state_losses)
        scaled_loss = 10 * (coefficient ** 2) * weighted_loss
        return scaled_loss

    return my_loss_fn


# ---------------------------
# PINN Model
# ---------------------------
class PINNModel(nn.Module):
    def __init__(self, integrator_cell, num_steps):
        """
        Unroll the integrator cell for a fixed number of time steps.
        """
        super(PINNModel, self).__init__()
        self.cell = integrator_cell
        self.num_steps = num_steps

    def forward(self, dummy_input):
        # dummy_input is not used; integration depends on the cell's initial state.
        batch_size = dummy_input.shape[0]
        # Remove unsqueeze; initial_state already has shape (1, 8)
        state = self.cell.initial_state.expand(batch_size, -1)
        outputs = []
        for _ in range(self.num_steps):
            output, state = self.cell(state)
            outputs.append(output)
        return torch.stack(outputs, dim=1)


# ---------------------------
# Main Training Script
# ---------------------------
if __name__ == "__main__":
    # Define kinetic parameters
    k_values = [5.259223e+06, 5.175076e+08, 5.232472e+08,
                5.551598e+08, 5.748630e+08, 5.647780e+08, 5.005279e+08]
    k1, k2, k3, k4, k5, k6, k7 = [np.float32(v) for v in k_values]
    c_eaq = np.float32(3.69e-12)

    # Load experimental data from CSV (ensure file exists in ./data/)
    df = pd.read_csv('./data/PFAS_data.csv')
    t_true = df['time (s)'].values
    t_pinn = np.arange(t_true[0], np.round(t_true[-1]), 1.0)
    # y_train: shape (batch, time_steps, outputs)
    y_train_np = df[['C7F15COO-', 'C5F11COO-', 'C3F7COO-', 'C2F5COO-', 'CF3COO-']].values[np.newaxis, :, :]
    y_train = torch.tensor(y_train_np, dtype=torch.float32)
    # Dummy input to determine number of time steps (shape: [batch, time_steps, 1])
    batch_input_np = t_pinn[np.newaxis, :, np.newaxis]
    batch_input = torch.tensor(batch_input_np, dtype=torch.float32)

    initial_state = np.array([[1.96e-07, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32)
    num_steps = batch_input.shape[1]

    # Create the integrator cell and PINN model
    integrator_cell = RungeKuttaIntegratorCell(k1, k2, k3, k4, k5, k6, k7, c_eaq, dt=1.0, initial_state=initial_state)
    model = PINNModel(integrator_cell, num_steps=num_steps)

    # Create loss function and optimizer (using a multi-step scheduler similar to PiecewiseConstantDecay)
    loss_fn = create_loss_fn(t_pinn, t_true)
    optimizer = optim.RMSprop(model.parameters(), lr=5e-2)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[70, 150, 250], gamma=0.2)

    # Evaluate model before training
    model.eval()
    with torch.no_grad():
        y_pred_before = model(batch_input)

    # Training loop
    model.train()
    start_time = time.time()
    num_epochs = 200
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        y_pred = model(batch_input)
        loss = loss_fn(y_pred, y_train)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
    end_time = time.time()
    print(f"Training Time: {end_time - start_time:.2f} seconds")

    # Evaluate model after training
    model.eval()
    with torch.no_grad():
        y_pred = model(batch_input)

    # Plotting results
    y_pred_before_np = y_pred_before.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()

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
        plt.xlabel('Time (t)')
        plt.ylabel(label)
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    os.makedirs("Results", exist_ok=True)
    output_file = "Results/HPINN_minimal_model_exp.png"
    plt.savefig(output_file, dpi=600, bbox_inches='tight')
    plt.show()

    # Print trained parameter values
    trained_params = {
        'k1': (10 ** integrator_cell.log_k1).item(),
        'k2': (10 ** integrator_cell.log_k2).item(),
        'k3': (10 ** integrator_cell.log_k3).item(),
        'k4': (10 ** integrator_cell.log_k4).item(),
        'k5': (10 ** integrator_cell.log_k5).item(),
        'k6': (10 ** integrator_cell.log_k6).item(),
        'k7': (10 ** integrator_cell.log_k7).item(),
    }
    print("Trained parameters:")
    for name, value in trained_params.items():
        print(f"{name}: {value:.6e}")
