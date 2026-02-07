"""SPLADE aggregation operators."""

import torch

_EPS = {
    torch.float32: {"div": 1e-6, "log": 1e-7},
    torch.bfloat16: {"div": 1e-3, "log": 1e-4},
}


def splade_aggregate(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Aggregate token logits into sparse vectors with shape [B, V]."""
    activated = torch.log1p(torch.relu(logits.float()))
    activated = activated.masked_fill(~mask.unsqueeze(-1).bool(), 0.0)
    return activated.max(dim=1).values
