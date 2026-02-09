import torch

from splade.training.constants import DF_MOMENTUM
from splade.utils.cuda import DEVICE


class DocumentFrequencyTracker:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.df_counts = torch.zeros(vocab_size, device=DEVICE)
        self.doc_count = 0.0

    def update(self, sparse_vectors: torch.Tensor) -> None:
        self.df_counts += (sparse_vectors.detach() > 0).sum(dim=0)
        self.doc_count += sparse_vectors.shape[0]

    def get_weights(self) -> torch.Tensor:
        """DF-based importance weights: high-DF tokens get upweighted for penalization.

        Uses normalized document frequency directly — tokens appearing in more
        documents are penalized more heavily. Scale-invariant by construction.
        """
        df_ratio = self.df_counts / max(self.doc_count, 1.0)
        return df_ratio

    def get_stats(self) -> dict:
        df_ratio = self.df_counts / max(self.doc_count, 1.0)
        return {
            "doc_count": self.doc_count,
            "top1_df_pct": df_ratio.max().item() * 100,
            "mean_df_pct": df_ratio.mean().item() * 100,
        }

    def reset(self) -> None:
        self.df_counts.zero_()
        self.doc_count = 0.0

    def soft_reset(self) -> None:
        self.df_counts *= DF_MOMENTUM
        self.doc_count *= DF_MOMENTUM


def compute_df_flops_reg(activations: torch.Tensor, df_weights: torch.Tensor) -> torch.Tensor:
    """Scale-invariant DF-FLOPS regularization via L1/L2 ratio.

    Uses the L1/L2 ratio of DF-weighted activations as a sparsity penalty.
    This is scale-invariant (Rahimi et al., 2019): multiplying activations
    by a constant does not change the penalty, eliminating the need for
    manually tuned alpha/beta shape parameters.

    The DF weighting ensures high-document-frequency tokens are penalized
    more heavily, encouraging the model to use rare, discriminative tokens.
    """
    mean_act = activations.abs().mean(dim=0)
    weighted = df_weights * mean_act
    l1 = weighted.sum()
    l2 = weighted.norm()
    # L1/L2 ratio: 1.0 when uniform (dense), 1/sqrt(n) when one-hot (sparse)
    # Minimizing this encourages sparsity. Normalized by sqrt(n) to [0, 1].
    n = weighted.shape[0]
    return l1 / (l2.clamp(min=1e-8) * n ** 0.5)
