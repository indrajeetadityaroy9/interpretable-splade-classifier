import torch

from splade.training.constants import DF_ALPHA, DF_BETA, DF_MOMENTUM
from splade.utils.cuda import DEVICE


class DocumentFrequencyTracker:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.df_counts = torch.zeros(vocab_size, device=DEVICE)
        self.doc_count = 0.0
        self._log_2 = torch.log(torch.tensor(2.0, device=DEVICE))
        log_alpha = torch.log(torch.tensor(DF_ALPHA, device=DEVICE))
        log_alpha = torch.where(log_alpha.abs() < 1e-7, torch.tensor(1e-7, device=DEVICE), log_alpha)
        self._cached_log_alpha_2 = self._log_2 / log_alpha

    def update(self, sparse_vectors: torch.Tensor) -> None:
        self.df_counts += (sparse_vectors.detach() > 0).sum(dim=0)
        self.doc_count += sparse_vectors.shape[0]

    def get_weights(self) -> torch.Tensor:
        df_ratio = self.df_counts / max(self.doc_count, 1.0)
        x_clamped = df_ratio.clamp(min=1e-8)
        x_pow = x_clamped.pow(self._cached_log_alpha_2)
        inner = (x_pow - 1.0).clamp(min=0.0)
        return 1.0 / (1.0 + inner.pow(DF_BETA))

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


class DFFlopsRegFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, activations: torch.Tensor, df_weights: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(activations, df_weights)
        mean_act = activations.abs().mean(dim=0)
        return (df_weights * mean_act ** 2).sum()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        activations, df_weights = ctx.saved_tensors
        batch_size = activations.shape[0]
        mean_act = activations.abs().mean(dim=0)
        sign = torch.where(activations != 0, torch.sign(activations), torch.zeros_like(activations))
        grad_activations = (
            grad_output
            * 2.0
            * df_weights.unsqueeze(0)
            * mean_act.unsqueeze(0)
            * sign
            / batch_size
        )
        return grad_activations, None


