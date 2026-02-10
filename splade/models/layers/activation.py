import torch
import torch.nn as nn
import torch.nn.functional as F


class DReLU(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x - self.theta)


class GatedJumpReLU(nn.Module):
    """Gated JumpReLU: hard binary gate (Heaviside) with sigmoid STE.

    Forward pass uses binary gating (preserves exact DLA identity).
    Backward pass uses sigmoid surrogate for gradient flow to gate_threshold.
    No magnitude shrinkage: active features keep full relu(x) magnitude.
    """

    def __init__(self, dim: int, init_threshold: float = 1.0):
        super().__init__()
        self.dim = dim
        self.gate_threshold = nn.Parameter(torch.ones(dim) * init_threshold)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        diff = x - self.gate_threshold
        gate_binary = (diff > 0).float()
        gate_sigmoid = torch.sigmoid(diff)
        # STE: binary in forward, sigmoid gradient in backward
        gate = gate_binary.detach() - gate_sigmoid.detach() + gate_sigmoid
        out = F.relu(x) * gate
        return out, gate_sigmoid
