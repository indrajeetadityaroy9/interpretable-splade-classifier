"""CircuitState and unified masking for CIS.

CircuitState wraps the forward pass outputs as a NamedTuple (torch.compile
compatible without pytree registration). circuit_mask() provides a single
temperature-parameterized masking function used by both training (soft) and
evaluation (hard).
"""

from typing import NamedTuple

import torch


class CircuitState(NamedTuple):
    """Produced by the forward pass, consumed by losses and metrics."""

    logits: torch.Tensor  # [B, C]
    sparse_vector: torch.Tensor  # [B, V]
    W_eff: torch.Tensor  # [B, C, V]
    b_eff: torch.Tensor  # [B, C]


def circuit_mask(
    attribution: torch.Tensor,
    circuit_fraction: float,
    temperature: float,
) -> torch.Tensor:
    """Unified soft/hard circuit mask via temperature-parameterized sigmoid.

    Training uses temperature ~10 (soft, differentiable).
    Evaluation uses temperature ~1e6 (hard, effectively binary).
    Both use the SAME circuit_fraction, ensuring consistent circuit boundaries.

    Args:
        attribution: [B, V] absolute attribution magnitudes for the target class.
        circuit_fraction: fraction of dimensions to retain (e.g. 0.1 = top 10%).
        temperature: sigmoid temperature. Higher = sharper mask.

    Returns:
        [B, V] mask in [0, 1].
    """
    n = attribution.shape[-1]
    k = max(1, int(circuit_fraction * n))
    kth_index = max(1, n - k)
    threshold = torch.kthvalue(attribution, kth_index, dim=-1).values
    return torch.sigmoid(temperature * (attribution - threshold.unsqueeze(-1)))
