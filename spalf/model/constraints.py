import math
import torch
from torch import Tensor


def compute_orthogonality_violation(z: Tensor, W_dec_A: Tensor, W_dec_B: Tensor, tau_ortho: float) -> Tensor:
    """Differentiable co-activation orthogonality violation via column sampling.

    Samples k≈√F columns to avoid materializing O(F²) Gram matrices; unbiased estimate
    of the activation-weighted off-diagonal coherence.
    """
    V = W_dec_A.shape[1]
    F_total = V + W_dec_B.shape[1]
    k = min(F_total, math.ceil(math.sqrt(F_total)))

    idx = torch.randint(F_total, (k,), device="cuda")
    cols_a = W_dec_A[:, idx.clamp(max=V - 1)]
    cols_b = W_dec_B[:, (idx - V).clamp(min=0)]
    W_sub = torch.where((idx < V).unsqueeze(0), cols_a, cols_b)
    W_hat = W_sub / W_sub.norm(dim=0, keepdim=True)

    G_sq = (W_hat.T @ W_hat).pow(2)
    active = (z[:, idx] > 0).float()
    C = active.T @ active

    weighted = G_sq * C
    num = weighted.sum() - weighted.diagonal().sum()
    denom = C.sum() - C.diagonal().sum() + torch.finfo(C.dtype).eps
    return num / denom - tau_ortho
