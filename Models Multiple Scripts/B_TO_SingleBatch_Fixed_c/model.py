# cinn_pfas/model.py

import torch
import torch.nn as nn
from .integrator import RungeKuttaIntegratorCell

class PINNModel(nn.Module):
    def __init__(self, integrator_cell, num_steps):
        """
        Args:
            integrator_cell (nn.Module): Instance of RungeKuttaIntegratorCell.
            num_steps (int): Number of time steps to integrate.
        """
        super().__init__()
        self.cell = integrator_cell
        self.num_steps = num_steps

    def forward(self, x=None):
        """
        The input x is unused because integration depends only on the initial state.
        Returns:
            Tensor of shape (batch, num_steps, output_dim)
        """
        outputs = []
        # Use the cell's initial_state and expand to batch size if needed
        if x is not None:
            batch_size = x.shape[0]
        else:
            batch_size = 1
        state = self.cell.initial_state.repeat(batch_size, 1)
        for _ in range(self.num_steps):
            output, state = self.cell(state)
            outputs.append(output)
        return torch.stack(outputs, dim=1)
