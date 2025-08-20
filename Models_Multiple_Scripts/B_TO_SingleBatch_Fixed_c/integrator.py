# cinn_pfas/integrator.py

import numpy as np
import torch
import torch.nn as nn


class RungeKuttaIntegratorCell(nn.Module):
    def __init__(self, k1, k2, k3, k4, k5, k6, k7, c_eaq, dt, initial_state):
        super().__init__()
        self.c_eaq = c_eaq
        self.dt = dt
        # Register initial_state as a buffer (non-trainable tensor)
        self.register_buffer('initial_state', torch.tensor(initial_state, dtype=torch.float32))
        self.state_size = 8
        # Initialize trainable log parameters (in log10 space)
        self.log_k1 = nn.Parameter(torch.tensor(np.log10(k1), dtype=torch.float32))
        self.log_k2 = nn.Parameter(torch.tensor(np.log10(k2), dtype=torch.float32))
        self.log_k3 = nn.Parameter(torch.tensor(np.log10(k3), dtype=torch.float32))
        self.log_k4 = nn.Parameter(torch.tensor(np.log10(k4), dtype=torch.float32))
        self.log_k5 = nn.Parameter(torch.tensor(np.log10(k5), dtype=torch.float32))
        self.log_k6 = nn.Parameter(torch.tensor(np.log10(k6), dtype=torch.float32))
        self.log_k7 = nn.Parameter(torch.tensor(np.log10(k7), dtype=torch.float32))

    def forward(self, y):
        # Compute current parameters by converting from log scale
        params = {
            'k1': torch.pow(10.0, self.log_k1),
            'k2': torch.pow(10.0, self.log_k2),
            'k3': torch.pow(10.0, self.log_k3),
            'k4': torch.pow(10.0, self.log_k4),
            'k5': torch.pow(10.0, self.log_k5),
            'k6': torch.pow(10.0, self.log_k6),
            'k7': torch.pow(10.0, self.log_k7),
        }
        k1_val = self._fun(y, params) * self.dt
        k2_val = self._fun(y + 0.5 * k1_val, params) * self.dt
        k3_val = self._fun(y + 0.5 * k2_val, params) * self.dt
        k4_val = self._fun(y + k3_val, params) * self.dt
        y_next = y + (k1_val + 2 * k2_val + 2 * k3_val + k4_val) / 6.0

        # Select outputs (e.g., specific states)
        y_1 = y_next[:, 0:1]
        y_3 = y_next[:, 2:3]
        y_5 = y_next[:, 4:5]
        y_6 = y_next[:, 5:6]
        y_7 = y_next[:, 6:7]
        output = torch.cat([y_1, y_3, y_5, y_6, y_7], dim=-1)
        return output, y_next

    def _fun(self, y, params):
        # y has shape (batch, 8)
        y_vars = [y[:, i:i + 1] for i in range(7)]
        k_vars = [params[f'k{i + 1}'] for i in range(7)]
        rates = [k * self.c_eaq * y_var for k, y_var in zip(k_vars, y_vars)]
        dy1 = -rates[0]
        dy2 = rates[0] - rates[1]
        dy3 = rates[1] - rates[2]
        dy4 = rates[2] - rates[3]
        dy5 = rates[3] - rates[4]
        dy6 = rates[4] - rates[5]
        dy7 = rates[5] - rates[6]
        dy8 = 2 * sum(rates)
        return torch.cat([dy1, dy2, dy3, dy4, dy5, dy6, dy7, dy8], dim=-1)


def interpolate_predictions(t_pinn, t_true, y_pred):
    """
    Interpolate y_pred at t_true time points using t_pinn.

    Args:
        t_pinn (np.array, list, or torch.Tensor): Time points for the PINN predictions.
        t_true (np.array, list, or torch.Tensor): Experimental time points.
        y_pred (torch.Tensor): Predictions of shape (batch, num_steps, output_dim).

    Returns:
        torch.Tensor: Interpolated predictions with shape (batch, len(t_true), output_dim).
    """
    device = y_pred.device
    # Convert t_pinn and t_true to tensors (or move them to device) if needed.
    if not torch.is_tensor(t_pinn):
        t_pinn = torch.tensor(t_pinn, dtype=torch.float32, device=device)
    else:
        t_pinn = t_pinn.to(device)
    if not torch.is_tensor(t_true):
        t_true = torch.tensor(t_true, dtype=torch.float32, device=device)
    else:
        t_true = t_true.to(device)

    # Find indices for interpolation
    indices = torch.searchsorted(t_pinn, t_true, right=False) - 1
    indices = indices.clamp(0, t_pinn.numel() - 2)  # shape: (L,)

    # Get lower and upper time bounds from t_pinn
    t0 = t_pinn[indices]        # shape: (L,)
    t1 = t_pinn[indices + 1]      # shape: (L,)

    # Use advanced indexing to extract predictions
    # y_pred has shape (batch, num_steps, output_dim)
    # y0 and y1 will have shape (batch, L, output_dim)
    y0 = y_pred[:, indices, :]
    y1 = y_pred[:, indices + 1, :]

    # Compute weights (shape: (L,))
    w = (t_true - t0) / (t1 - t0)
    # Reshape w for broadcasting over batch and output dimensions
    w = w.view(1, -1, 1)

    # Interpolate linearly between y0 and y1
    y_pred_interp = y0 + w * (y1 - y0)
    return y_pred_interp
