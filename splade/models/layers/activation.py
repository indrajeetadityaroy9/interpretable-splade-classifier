"""Activation functions including dReLU for Turbo Sparse."""

import torch
import torch.nn as nn

class DReLU(nn.Module):
    """Dynamic ReLU (dReLU) with learnable threshold for hard sparsity.
    
    f(x) = max(0, x - theta) where theta is a learnable parameter.
    Used in Turbo Sparse / SPLADE v3 to enforce true 0s for inference efficiency.
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Assuming x is [Batch, Seq, Dim] or [Batch, Dim]
        # Broadcasting theta across batch/seq
        return torch.relu(x - self.theta)