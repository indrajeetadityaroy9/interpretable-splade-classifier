"""Dead latent counting: features with zero activations across the evaluation set.

Evaluation budget: 10^7 tokens, matching the dead latent definition from
Rajamanoharan et al. (2024) and Gao et al. (2024).
"""


import torch

from spalf.data.activation_store import ActivationStore
from spalf.model import StratifiedSAE
from spalf.whitening.whitener import SoftZCAWhitener

# 10^7 tokens: a feature is dead if it fires zero times over this budget.
_DEAD_EVAL_TOKENS: int = 10_000_000


@torch.no_grad()
def count_dead_latents(
    sae: StratifiedSAE,
    whitener: SoftZCAWhitener,
    store: ActivationStore,
) -> dict[str, float]:
    """Count features that never activate, split by anchored and free strata."""
    device = next(sae.parameters()).device
    tokens_per_batch = store.batch_size * store.seq_len
    n_eval_batches = _DEAD_EVAL_TOKENS // tokens_per_batch

    ever_active = torch.zeros(sae.F, dtype=torch.bool, device=device)
    total_tokens = 0

    for _ in range(n_eval_batches):
        x = store.next_batch().to(device)
        x_tilde = whitener.forward(x)
        _, _, gate_mask, _, _ = sae(x_tilde)
        ever_active |= gate_mask.bool().any(dim=0)
        total_tokens += tokens_per_batch

    dead = ~ever_active
    dead_anchored = dead[: sae.V]
    dead_free = dead[sae.V :]

    return {
        "n_dead": dead.sum().item(),
        "frac_dead": dead.float().mean().item(),
        "n_dead_anchored": dead_anchored.sum().item(),
        "frac_dead_anchored": dead_anchored.float().mean().item(),
        "n_dead_free": dead_free.sum().item(),
        "frac_dead_free": dead_free.float().mean().item(),
        "eval_tokens": total_tokens,
    }
