"""Unified CIS circuit losses — completeness, separation, sharpness.

All losses consume W_eff from the forward pass via the shared DLA function
(mechanistic/attribution.py:compute_attribution_tensor) and use the unified
circuit_mask() from circuits/core.py.

Loss weighting uses uncertainty weighting (Kendall et al., 2018): each loss
gets a learnable log-variance parameter. Tasks with high uncertainty are
automatically down-weighted, eliminating manual CC/SEP/SHARP weight constants.

Centroid EMA uses a 20-step window (decay ≈ 0.905), derived from semantics
rather than arbitrary tuning.
"""

import math
from typing import Callable

import torch
import torch.nn.functional as F

from splade.circuits.core import circuit_mask
from splade.mechanistic.attribution import compute_attribution_tensor
from splade.training.constants import CIRCUIT_FRACTION, CIRCUIT_TEMPERATURE
from splade.utils.cuda import DEVICE

# Centroid EMA: 20-step effective window → decay = 1 - 2/(20+1) ≈ 0.905
_CENTROID_EMA_WINDOW = 20
_CENTROID_EMA_DECAY = 1.0 - 2.0 / (_CENTROID_EMA_WINDOW + 1)


class UncertaintyWeighting(torch.nn.Module):
    """Learnable multi-task loss weighting via homoscedastic uncertainty.

    Each task i gets a learnable log-variance log(σ²_i). The weighted loss is:
        L_total = Σ_i [L_i / (2σ²_i) + ½ log(σ²_i)]

    The log(σ²) regularizer prevents all weights from diverging to infinity.
    Tasks with noisy/unstable gradients automatically get down-weighted.

    Reference: Kendall, Gal & Cipolla, "Multi-Task Learning Using Uncertainty
    to Weigh Losses" (arXiv:1705.07115), CVPR 2018.
    """

    def __init__(self, num_tasks: int = 3):
        super().__init__()
        # Initialize log(σ²) = 0 → σ² = 1 → equal weighting initially
        self.log_vars = torch.nn.Parameter(torch.zeros(num_tasks, device=DEVICE))

    def forward(self, *losses: torch.Tensor) -> torch.Tensor:
        total = torch.tensor(0.0, device=DEVICE)
        for i, loss in enumerate(losses):
            # Kendall et al.: L_i / (2σ²_i) + (1/2)log(σ²_i)
            # With log_var = log(σ²): precision = exp(-log_var) = 1/σ²
            precision = torch.exp(-self.log_vars[i])  # 1/σ²
            total = total + 0.5 * precision * loss + 0.5 * self.log_vars[i]
        return total


class AttributionCentroidTracker:
    """Maintains EMA of per-class mean absolute attribution vectors.

    Used by the separation loss to encourage distinct per-class circuits
    without requiring all classes to appear in every mini-batch.

    EMA decay is derived from a 20-step effective window (≈ 0.905),
    making the semantics explicit rather than tuning an opaque constant.
    """

    def __init__(
        self,
        num_classes: int,
        vocab_size: int,
    ):
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.momentum = _CENTROID_EMA_DECAY
        self.centroids = torch.zeros(num_classes, vocab_size, device=DEVICE)
        self._initialized = torch.zeros(num_classes, dtype=torch.bool, device=DEVICE)

    @torch.no_grad()
    def update(
        self,
        sparse_vector: torch.Tensor,
        W_eff: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        for c in range(self.num_classes):
            mask = labels == c
            if not mask.any():
                continue
            class_sparse = sparse_vector[mask]
            class_W_eff = W_eff[mask]
            class_labels = torch.full(
                (class_sparse.shape[0],), c, device=sparse_vector.device,
            )
            attr = (
                compute_attribution_tensor(class_sparse, class_W_eff, class_labels)
                .abs()
                .mean(dim=0)
            )
            if self._initialized[c]:
                self.centroids[c].lerp_(attr, 1.0 - self.momentum)
            else:
                self.centroids[c].copy_(attr)
                self._initialized[c] = True

    def get_normalized_centroids(self) -> torch.Tensor:
        """Return L2-normalized centroids for cosine similarity computation."""
        norms = self.centroids.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return self.centroids / norms


def compute_completeness_loss(
    sparse_vector: torch.Tensor,
    W_eff: torch.Tensor,
    labels: torch.Tensor,
    classifier_forward_fn: Callable[[torch.Tensor], torch.Tensor],
    circuit_fraction: float = CIRCUIT_FRACTION,
    temperature: float = CIRCUIT_TEMPERATURE,
) -> torch.Tensor:
    """Differentiable circuit completeness via unified soft masking.

    Uses circuit_mask() from core.py with the same fraction and temperature
    parameters used throughout the system.
    """
    attr_magnitude = compute_attribution_tensor(sparse_vector, W_eff, labels).abs()
    soft_mask = circuit_mask(attr_magnitude, circuit_fraction, temperature)
    masked_sparse = sparse_vector * soft_mask
    masked_logits = classifier_forward_fn(masked_sparse)
    return F.cross_entropy(masked_logits, labels)


def compute_separation_loss(
    centroid_tracker: AttributionCentroidTracker,
) -> torch.Tensor:
    """Mean pairwise cosine similarity between class attribution centroids."""
    centroids = centroid_tracker.get_normalized_centroids()
    n = centroids.shape[0]
    if n < 2:
        return torch.tensor(0.0, device=DEVICE)
    sim_matrix = centroids @ centroids.T
    mask = torch.triu(torch.ones(n, n, device=DEVICE, dtype=torch.bool), diagonal=1)
    return sim_matrix[mask].mean()


def compute_sharpness_loss(
    sparse_vector: torch.Tensor,
    W_eff: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """1 - Hoyer sparsity of attribution magnitudes. O(n), no sorting."""
    attr = compute_attribution_tensor(sparse_vector, W_eff, labels).abs()
    n = attr.shape[-1]
    sqrt_n = math.sqrt(n)
    l1 = attr.sum(dim=-1)
    l2 = attr.norm(dim=-1)
    hoyer = (sqrt_n - l1 / l2.clamp(min=1e-8)) / (sqrt_n - 1.0)
    return (1.0 - hoyer).mean()
