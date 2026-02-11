"""Virtual Polysemy Expansion (VPE).

Expands polysemous tokens into multiple virtual sense slots,
resolving semantic superposition in the sparse bottleneck.

For K polysemous tokens with M senses each, adds K*(M-1) virtual
dimensions to the sparse vector. Hidden-state projections disambiguate
which sense is active per position, using winner-take-all hard assignment
with straight-through estimator for gradients.

DLA preservation: each virtual slot s_virtual[k,m] = original_logit[k] * gate[k,m].
Winner-take-all zeros all but one slot, so exactly one slot carries the full
original magnitude. The classifier operates on the expanded vector [V + K*(M-1)],
preserving logit[c] = sum_j s_expanded[j] * W_eff[c,j] + b_eff[c].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VirtualExpander(nn.Module):
    """Expands polysemous tokens into virtual sense slots.

    Args:
        backbone_hidden_dim: Hidden dimension H from backbone (e.g. 768).
        polysemous_token_ids: List of K token IDs to expand.
        num_senses: M, number of sense slots per token.
    """

    def __init__(
        self,
        backbone_hidden_dim: int,
        polysemous_token_ids: list[int],
        num_senses: int = 4,
    ):
        super().__init__()
        self.token_ids = polysemous_token_ids
        self.num_senses = num_senses
        K = len(polysemous_token_ids)
        self.num_virtual_slots = K * (num_senses - 1)

        self.sense_proj = nn.Linear(backbone_hidden_dim, K * num_senses, bias=False)
        self._token_to_idx = {tid: i for i, tid in enumerate(polysemous_token_ids)}

    def forward(
        self,
        hidden_states: torch.Tensor,
        mlm_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Expand MLM logits with virtual sense slots.

        Args:
            hidden_states: [B, L, H] backbone hidden states.
            mlm_logits: [B, L, V] original MLM logits.

        Returns:
            [B, L, V + K*(M-1)] expanded logits.
        """
        B, L, V = mlm_logits.shape
        K = len(self.token_ids)
        M = self.num_senses

        sense_scores = self.sense_proj(hidden_states).view(B, L, K, M)

        # Winner-take-all with STE: hard argmax forward, soft gradient backward
        sense_hard = F.one_hot(sense_scores.argmax(dim=-1), M).float()
        sense_soft = F.softmax(sense_scores, dim=-1)
        sense_gate = sense_hard.detach() - sense_soft.detach() + sense_soft

        token_ids_t = torch.tensor(self.token_ids, device=mlm_logits.device)
        poly_logits = mlm_logits[:, :, token_ids_t]  # [B, L, K]

        sense_logits = poly_logits.unsqueeze(-1) * sense_gate  # [B, L, K, M]

        # Slot 0 replaces original position; slots 1..M-1 become virtual dims
        mlm_logits_expanded = mlm_logits.clone()
        mlm_logits_expanded[:, :, token_ids_t] = sense_logits[:, :, :, 0]

        virtual_logits = sense_logits[:, :, :, 1:].reshape(B, L, K * (M - 1))

        return torch.cat([mlm_logits_expanded, virtual_logits], dim=-1)
