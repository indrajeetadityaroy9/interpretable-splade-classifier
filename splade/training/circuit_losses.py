"""CIS differentiable circuit losses — always active during training.

Three auxiliary losses that jointly optimize task accuracy, sparsity, and
circuit quality as part of the Circuit-Integrated SPLADE (CIS) framework:

  1. Circuit completeness: soft-masked sparse vector should still classify
     correctly, encouraging decision-relevant information to concentrate
     in a small subset of vocabulary dimensions.
  2. Circuit separation: per-class attribution centroids should be dissimilar,
     encouraging distinct circuits for distinct classes.
  3. Attribution sharpness: attribution magnitude should be concentrated
     (high Gini coefficient), not diffuse across the vocabulary.

All losses use W_eff (the effective weight matrix from the ReLU MLP
classifier) for exact per-sample DLA attribution:

    W_eff(s) = W2 @ diag(D(s)) @ W1

where D(s) is the activation mask from the ReLU hidden layer.
"""

from typing import Callable

import torch
import torch.nn.functional as F

from splade.training.constants import (
    CENTROID_MOMENTUM,
    CIRCUIT_FRACTION,
    CIRCUIT_TEMPERATURE,
)
from splade.utils.cuda import DEVICE


class AttributionCentroidTracker:
    """Maintains EMA of per-class mean absolute attribution vectors.

    Used by the separation loss to encourage distinct per-class circuits
    without requiring all classes to appear in every mini-batch.
    """

    def __init__(
        self,
        num_classes: int,
        vocab_size: int,
        momentum: float = CENTROID_MOMENTUM,
    ):
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.momentum = momentum
        self.centroids = torch.zeros(num_classes, vocab_size, device=DEVICE)
        self._initialized = torch.zeros(num_classes, dtype=torch.bool, device=DEVICE)

    @torch.no_grad()
    def update(
        self,
        sparse_vector: torch.Tensor,
        W_eff: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        """Update centroids with current batch attributions.

        Args:
            sparse_vector: [batch, V] sparse activations.
            W_eff: [batch, C, V] effective weight matrix from ReLU MLP.
            labels: [batch] integer class labels.
        """
        for c in range(self.num_classes):
            mask = labels == c
            if not mask.any():
                continue
            class_sparse = sparse_vector[mask]
            class_weights = W_eff[mask, c, :]
            attr = (class_sparse * class_weights).abs().mean(dim=0)
            if self._initialized[c]:
                self.centroids[c].lerp_(attr, 1.0 - self.momentum)
            else:
                self.centroids[c].copy_(attr)
                self._initialized[c] = True

    def get_normalized_centroids(self) -> torch.Tensor:
        """Return L2-normalized centroids for cosine similarity computation."""
        norms = self.centroids.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return self.centroids / norms


class CircuitLossSchedule:
    """Delayed quadratic warmup for circuit losses.

    Circuit losses are disabled during early training (warmup_fraction of
    total steps) to let the model learn basic representations first, then
    ramp up quadratically.
    """

    def __init__(self, total_steps: int, warmup_fraction: float):
        self.total_steps = total_steps
        self.delay_steps = int(warmup_fraction * total_steps)
        self._step = 0

    def step(self) -> float:
        """Return current circuit loss weight multiplier in [0, 1]."""
        self._step += 1
        if self._step < self.delay_steps:
            return 0.0
        remaining = self.total_steps - self.delay_steps
        if remaining <= 0:
            return 1.0
        progress = (self._step - self.delay_steps) / remaining
        return min(1.0, progress * progress)


def compute_circuit_completeness_loss(
    sparse_vector: torch.Tensor,
    W_eff: torch.Tensor,
    labels: torch.Tensor,
    classifier_forward_fn: Callable[[torch.Tensor], torch.Tensor],
    circuit_fraction: float = CIRCUIT_FRACTION,
    temperature: float = CIRCUIT_TEMPERATURE,
) -> torch.Tensor:
    """Differentiable circuit completeness via soft top-k masking.

    Computes per-sample DLA using W_eff for the target class, builds a soft
    mask retaining only the top circuit_fraction of dimensions (by attribution
    magnitude), forwards the masked sparse vector through the ReLU MLP
    classifier, and returns the cross-entropy loss against the true labels.

    Args:
        sparse_vector: [batch, V] non-negative sparse activations.
        W_eff: [batch, C, V] effective weight matrix from ReLU MLP.
        labels: [batch] integer class labels.
        classifier_forward_fn: callable(sparse) -> logits for the ReLU MLP.
        circuit_fraction: fraction of vocab dimensions to retain.
        temperature: sigmoid temperature (higher = sharper mask).

    Returns:
        Scalar loss.
    """
    # Per-sample DLA attribution for the target class
    batch_indices = torch.arange(sparse_vector.shape[0], device=sparse_vector.device)
    target_weights = W_eff[batch_indices, labels, :]  # [batch, V]
    attr_magnitude = (sparse_vector * target_weights).abs()

    # Soft top-k: keep top circuit_fraction of dimensions
    k = max(1, int(circuit_fraction * sparse_vector.shape[-1]))
    threshold = torch.kthvalue(
        attr_magnitude, attr_magnitude.shape[-1] - k, dim=-1,
    ).values.unsqueeze(-1)

    soft_mask = torch.sigmoid(temperature * (attr_magnitude - threshold))

    # Forward masked sparse through ReLU MLP classifier
    masked_sparse = sparse_vector * soft_mask
    masked_logits = classifier_forward_fn(masked_sparse)

    return F.cross_entropy(masked_logits, labels)


def compute_circuit_separation_loss(
    centroid_tracker: AttributionCentroidTracker,
) -> torch.Tensor:
    """Mean pairwise cosine similarity between class attribution centroids.

    Lower similarity means more distinct per-class circuits.  The loss is
    the mean of the upper-triangle of the cosine similarity matrix.

    Uses detached centroids (EMA statistics), so gradients flow only
    through the current batch's contribution to the separation signal.

    Args:
        centroid_tracker: tracker with up-to-date centroids.

    Returns:
        Scalar loss (mean pairwise cosine similarity).
    """
    centroids = centroid_tracker.get_normalized_centroids()
    n = centroids.shape[0]
    if n < 2:
        return torch.tensor(0.0, device=DEVICE)

    sim_matrix = centroids @ centroids.T
    mask = torch.triu(torch.ones(n, n, device=DEVICE, dtype=torch.bool), diagonal=1)
    return sim_matrix[mask].mean()


def compute_attribution_sharpness_loss(
    sparse_vector: torch.Tensor,
    W_eff: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """1 - Gini coefficient of the attribution magnitude distribution.

    The Gini coefficient measures inequality: a high Gini means attribution
    is concentrated on few dimensions (sharp, interpretable).  The loss is
    1 - Gini, so minimizing it maximizes sharpness.

    Args:
        sparse_vector: [batch, V].
        W_eff: [batch, C, V] effective weight matrix from ReLU MLP.
        labels: [batch].

    Returns:
        Scalar loss (1 - mean Gini).
    """
    batch_indices = torch.arange(sparse_vector.shape[0], device=sparse_vector.device)
    target_weights = W_eff[batch_indices, labels, :]
    attr = (sparse_vector * target_weights).abs()

    # Sort ascending for Gini computation
    sorted_attr, _ = attr.sort(dim=-1)
    n = sorted_attr.shape[-1]

    # Gini = (2 * sum(i * x_i)) / (n * sum(x_i)) - (n+1)/n
    # where i is 1-indexed and x_i is sorted ascending
    index = torch.arange(1, n + 1, device=sorted_attr.device, dtype=sorted_attr.dtype)
    weighted_sum = (index.unsqueeze(0) * sorted_attr).sum(dim=-1)
    total_sum = sorted_attr.sum(dim=-1).clamp(min=1e-8)

    gini = (2.0 * weighted_sum) / (n * total_sum) - (n + 1.0) / n

    return (1.0 - gini).mean()
