"""B-cos layers and classifier for exact DLA at arbitrary depth.

B-cos networks (Boehle et al., CVPR 2022) replace standard linear layers
with cosine-similarity-based transforms. The key property: for any depth,
the output is expressible as a linear function of the input:

    output = W_dyn(x) @ x

where W_dyn is the input-dependent dynamic weight matrix. This gives exact
DLA (W_eff = W_dyn, b_eff = 0) for any number of layers.

For B=2 (default):
    f_i(x) = |cos(w_i, x)|^(B-1) * (w_norm_i^T x) = cos_i * (w_norm_i^T x)
    W_dyn = diag(cos_sim) @ W_norm

Multi-layer composition:
    W_eff = W_dyn_L @ ... @ W_dyn_2 @ W_dyn_1
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class BcosLinear(nn.Module):
    """B-cos linear layer with exact dynamic weight decomposition.

    No bias — B-cos networks are inherently bias-free, which is what
    enables exact W_eff decomposition at any depth.
    """

    def __init__(self, in_features: int, out_features: int, B: int = 2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.B = B
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_norm = F.normalize(self.weight, dim=1)
        proj = F.linear(x, w_norm)  # [B, out_features]
        x_norm = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        cos_sim = proj / x_norm
        return cos_sim.abs().pow(self.B - 1) * proj

    def get_dynamic_weight(self, x: torch.Tensor) -> torch.Tensor:
        """Compute W_dyn such that forward(x) == W_dyn @ x (batched).

        Returns:
            [batch, out_features, in_features] dynamic weight matrix.
        """
        w_norm = F.normalize(self.weight, dim=1)
        proj = F.linear(x, w_norm)
        x_norm = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        cos_sim = proj / x_norm  # [B, out]
        scale = cos_sim.abs().pow(self.B - 1)  # [B, out]
        return scale.unsqueeze(-1) * w_norm.unsqueeze(0)  # [B, out, in]


class BcosClassifier(nn.Module):
    """Multi-layer B-cos classifier with exact W_eff for any depth.

    Provides the same (logits, W_eff, b_eff) interface as the ReLU
    classifier in SpladeModel.classifier_forward().
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_labels: int,
        num_layers: int = 2,
        B: int = 2,
    ):
        super().__init__()
        sizes = [vocab_size] + [hidden_size] * (num_layers - 1) + [num_labels]
        self.layers = nn.ModuleList([
            BcosLinear(sizes[i], sizes[i + 1], B=B)
            for i in range(num_layers)
        ])

    def forward(self, sparse_vector: torch.Tensor) -> torch.Tensor:
        h = sparse_vector
        for layer in self.layers:
            h = layer(h)
        return h

    def classifier_forward(
        self, sparse_vector: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning (logits, W_eff, b_eff).

        W_eff = W_dyn_L @ ... @ W_dyn_1 (exact for any depth).
        b_eff = 0 (B-cos has no bias).
        """
        h = sparse_vector
        W_eff = None

        for layer in self.layers:
            W_dyn = layer.get_dynamic_weight(h)  # [B, out, in]
            h = layer(h)
            W_eff = W_dyn if W_eff is None else torch.bmm(W_dyn, W_eff)

        b_eff = torch.zeros(
            h.shape, device=h.device, dtype=h.dtype,
        )
        return h, W_eff, b_eff
