"""Feature absorption: alignment of free decoder features with vocabulary directions."""

import torch
import torch.nn.functional as F
from torch import Tensor


@torch.no_grad()
def feature_absorption_rate(
    W_dec_B: Tensor,
    W_vocab: Tensor,
) -> dict[str, float]:
    """Estimate absorption of vocabulary directions into free decoder features."""
    B_norm = F.normalize(W_dec_B, dim=0)
    V_norm = F.normalize(W_vocab, dim=0)

    cos_matrix = B_norm.T @ V_norm
    max_cos_per_free = cos_matrix.abs().max(dim=1).values

    n_free = W_dec_B.shape[1]

    return {
        "n_free": n_free,
        "max_alignment": max_cos_per_free.max().item(),
        "mean_alignment": max_cos_per_free.mean().item(),
        "std_alignment": max_cos_per_free.std().item(),
    }
