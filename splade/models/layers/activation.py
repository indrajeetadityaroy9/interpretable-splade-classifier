import torch
import torch.nn as nn
import torch.nn.functional as F

from splade.training.constants import JUMPRELU_BANDWIDTH


class _JumpReLUSTE(torch.autograd.Function):
    """JumpReLU with sigmoid straight-through estimator (Rajamanoharan et al. 2024).

    Forward:  z · H(z - θ)           (exact binary gate, DLA-compatible)
    Backward: Uses σ'((z - θ) / ε)   (smooth gradient through θ)

    where ε is a bandwidth hyperparameter controlling the STE sharpness.
    """

    @staticmethod
    def forward(ctx, z: torch.Tensor, log_threshold: torch.Tensor, bandwidth: float):
        theta = log_threshold.exp()
        gate = (z > theta).float()
        ctx.save_for_backward(z, theta, gate)
        ctx.bandwidth = bandwidth
        return z * gate, gate

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor, grad_gate: torch.Tensor):
        z, theta, gate = ctx.saved_tensors
        eps = ctx.bandwidth

        # Gradient w.r.t. z: pass through where gate is open
        grad_z = grad_output * gate

        # Gradient w.r.t. log_threshold (via chain rule through exp):
        # d/d(log_θ) of H(z - θ) ≈ -σ'((z - θ)/ε) · θ/ε
        # where σ'(x) = σ(x)(1-σ(x))
        scaled = (z - theta) / eps
        sigmoid_deriv = torch.sigmoid(scaled) * (1 - torch.sigmoid(scaled))
        # The upstream gradient for the gate dimension is: grad_output * z (from z * gate)
        # plus any direct grad_gate
        upstream = grad_output * z + grad_gate
        grad_log_theta = -(upstream * sigmoid_deriv * theta / eps).sum(
            dim=list(range(len(z.shape) - 1))
        )

        return grad_z, grad_log_theta, None


class JumpReLUGate(nn.Module):
    """JumpReLU gate (Rajamanoharan et al. 2024).

    Produces exact binary gates during BOTH training and evaluation,
    ensuring the DLA identity logit[c] = Σ s[j]·W_eff[c,j] + b_eff[c]
    holds exactly at all times, not just at eval.

    The learnable threshold θ_j (stored as log_θ for numerical stability)
    controls per-dimension sparsity. Gradients flow through θ via a
    sigmoid straight-through estimator with bandwidth ε.

    Returns 3-tuple: (output, gate_mask, l0_probs)
      - gate_mask: binary {0,1} tensor (exact gates, not probabilities)
      - l0_probs: σ((z - θ) / ε), differentiable proxy for E[gates]
        used by the L0 sparsity loss
    """

    def __init__(self, dim: int, init_log_threshold: float = 0.0):
        super().__init__()
        self.dim = dim
        self.log_threshold = nn.Parameter(
            torch.full((dim,), init_log_threshold)
        )

    def forward(
        self, x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = F.relu(x)

        if self.training:
            output, gate = _JumpReLUSTE.apply(
                z, self.log_threshold, JUMPRELU_BANDWIDTH,
            )
        else:
            theta = self.log_threshold.exp()
            gate = (z > theta).float()
            output = z * gate

        # L0 proxy: differentiable expected gate opening probability
        # P(z_j > θ_j) approximated via sigmoid for gradient flow to θ
        # When z is large relative to θ, this is ~1; when small, ~0
        theta = self.log_threshold.exp()
        l0_probs = torch.sigmoid(
            (z - theta) / JUMPRELU_BANDWIDTH
        )

        return output, gate, l0_probs
