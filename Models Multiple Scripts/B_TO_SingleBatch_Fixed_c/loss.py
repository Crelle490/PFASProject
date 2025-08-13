# cinn_pfas/loss.py

import torch
from .integrator import interpolate_predictions


def create_loss_fn(t_pinn, t_true):
    def loss_fn(y_pred, y_true):
        """
        Custom loss function with dynamic scaling and weighting.

        Args:
            y_pred: Tensor of shape (batch, num_steps, output_dim)
            y_true: Tensor of shape (batch, len(t_true), output_dim)

        Returns:
            A scalar loss.
        """
        y_pred_interp = interpolate_predictions(t_pinn, t_true, y_pred)
        if y_pred_interp.shape != y_true.shape:
            raise ValueError("Interpolated predictions and true values have different shapes.")

        # Compute maximum absolute values for each output over batch and time
        y_max = torch.max(torch.abs(y_true), dim=1)[0]
        y_max = torch.max(y_max, dim=0)[0]
        max_y_max = torch.max(y_max)
        weights = max_y_max / y_max
        coefficient = 1.0 / max_y_max

        # Compute per-state mean squared error
        state_losses = torch.mean((y_true - y_pred_interp) ** 2, dim=(0, 1))
        weighted_loss = torch.sum(weights * state_losses)
        scaled_loss = 10 * (coefficient ** 2) * weighted_loss
        return scaled_loss

    return loss_fn
