import torch
from torch import Tensor


def _lw_nonlinear_shrinkage(eigenvalues: Tensor, n: int) -> Tensor:
    """Ledoit-Wolf 2020 analytical nonlinear shrinkage (Eq. 4.3 + 4.9)."""
    p = len(eigenvalues)
    c = p / n
    lam = eigenvalues.to(torch.float64).clone()
    lam += torch.finfo(torch.float64).eps * torch.arange(p, 0, -1, dtype=torch.float64, device="cuda")

    diffs = lam.unsqueeze(0) - lam.unsqueeze(1)
    diffs.fill_diagonal_(1.0)
    h = (1.0 / diffs).sum(dim=0) / p - 1.0 / p
    m = -(1.0 - c) / lam + c * h
    return (lam / (1.0 - c - c * lam * m).abs().pow(2)).clamp_min(
        torch.finfo(torch.float64).eps).to(eigenvalues.dtype)


class FrequentDirections:
    """Streaming FD sketch (Liberty 2013); l=√d rows (Ghashami 2016 Thm 3)."""

    def __init__(self, d: int) -> None:
        self.d = d
        self._l = min(d, int(d ** 0.5))
        self._sketch = torch.zeros(2 * self._l, d, dtype=torch.float64, device="cuda")
        self._next_row = 0
        self._n = 0
        self._mean = torch.zeros(d, dtype=torch.float64, device="cuda")
        self._sum_sq = torch.zeros(1, dtype=torch.float64, device="cuda")
        self._prev_sv: Tensor | None = None
        self._converged = False

    def update(self, x: Tensor) -> None:
        x = x.to(torch.float64)
        B = x.shape[0]
        # Welford mean + parallel M2.
        batch_mean = x.mean(dim=0)
        delta = batch_mean - self._mean
        new_n = self._n + B
        self._mean = self._mean + delta * (B / new_n)
        self._sum_sq += (x - batch_mean).pow(2).sum() + delta.pow(2).sum() * (self._n * B / new_n)
        self._n = new_n

        centered = x - self._mean
        pos = 0
        while pos < B:
            chunk = min(2 * self._l - self._next_row, B - pos)
            self._sketch[self._next_row:self._next_row + chunk] = centered[pos:pos + chunk]
            self._next_row += chunk
            pos += chunk
            if self._next_row >= 2 * self._l:
                self._compress()

    def _compress(self) -> None:
        _, s, Vt = torch.linalg.svd(self._sketch, full_matrices=False)
        if len(s) > self._l:
            s_shrunk = (s[:self._l].pow(2) - s[self._l].pow(2)).clamp_min(0.0).sqrt()
        else:
            s_shrunk = s[:self._l]
        self._sketch[:self._l] = s_shrunk.unsqueeze(1) * Vt[:self._l]
        self._sketch[self._l:] = 0.0
        self._next_row = self._l

        if self._n >= self.d and self._prev_sv is not None and \
           (s_shrunk - self._prev_sv).norm() / s_shrunk.norm() < 1.0 / self.d:
            self._converged = True
        self._prev_sv = s_shrunk.clone()

    @property
    def trace(self) -> float:
        return (self._sum_sq / (self._n - 1)).item()

    def get_eigendecomposition(self) -> tuple[Tensor, Tensor]:
        if self._next_row > self._l:
            self._compress()
        _, s, Vt = torch.linalg.svd(self._sketch[:self._l], full_matrices=False)
        return s.pow(2) / (self._n - 1), Vt.T


class SoftZCAWhitener:
    """Soft-ZCA whitening with LW2020 per-eigenvalue shrinkage."""

    def __init__(self, mean: Tensor, eigenvalues: Tensor, eigenvectors: Tensor,
                 reg_eigenvalues: Tensor, n_samples: int, total_trace: float = 0.0) -> None:
        self.d = mean.shape[0]
        self._n_samples = n_samples
        self._total_trace = total_trace
        self.mean = mean.float()
        self._eigenvalues = eigenvalues.float()
        self._eigenvectors = eigenvectors.float()
        self._reg_eigenvalues = reg_eigenvalues.float()

        l = eigenvectors.shape[1]
        # Spectral gap: largest relative drop between consecutive eigenvalues.
        self._k = int((reg_eigenvalues[:-1] / reg_eigenvalues[1:]).argmax().item()) + 1
        self.is_low_rank = l < self.d or self._k < l

        if self.is_low_rank:
            self._U_k = self._eigenvectors[:, :self._k]
            if l < self.d and total_trace > 0.0:
                sketch_trace = self._reg_eigenvalues.sum().item()
                tail_sum = self._reg_eigenvalues[self._k:].sum().item()
                self._lambda_bar = (tail_sum + total_trace - sketch_trace) / (self.d - self._k)
            else:
                self._lambda_bar = float(self._reg_eigenvalues[self._k:].mean())
            self._scale_k = self._reg_eigenvalues[:self._k].rsqrt()
            self._scale_tail = 1.0 / self._lambda_bar ** 0.5
            self._diff_scale = self._scale_k - self._scale_tail
            self._diff_scale_sq = self._scale_k.pow(2) - self._scale_tail ** 2
            self._scale_tail_sq = self._scale_tail ** 2
        else:
            scales = self._reg_eigenvalues.rsqrt()
            self._W_white = self._eigenvectors @ torch.diag(scales) @ self._eigenvectors.T
            self._precision = self._W_white.T @ self._W_white

    @property
    def effective_rank(self) -> int:
        """Shannon effective rank = exp(entropy of normalized spectrum)."""
        p = self._eigenvalues / self._eigenvalues.sum()
        return int(torch.exp(-(p * p.log()).sum()).item())

    @property
    def noise_fraction(self) -> float:
        sketch_total = self._eigenvalues.sum().item()
        noise = self._eigenvalues[self._k:].sum().item()
        if self._total_trace > 0.0:
            return max((noise + self._total_trace - sketch_total) / self._total_trace, 1.0 / self.d)
        return max(noise / sketch_total, 1.0 / self.d)

    @classmethod
    def from_sketch(cls, sketch: FrequentDirections) -> "SoftZCAWhitener":
        eigenvalues, eigenvectors = sketch.get_eigendecomposition()
        eigenvalues = eigenvalues.clamp_min(torch.finfo(eigenvalues.dtype).eps * eigenvalues.max())
        return cls(sketch._mean, eigenvalues, eigenvectors,
                   _lw_nonlinear_shrinkage(eigenvalues, sketch._n),
                   sketch._n, sketch.trace)

    def forward(self, x: Tensor) -> Tensor:
        centered = x - self.mean
        if self.is_low_rank:
            proj = centered @ self._U_k
            return (proj * self._diff_scale) @ self._U_k.T + centered * self._scale_tail
        return centered @ self._W_white.T

    def compute_mahalanobis_sq(self, diff: Tensor) -> Tensor:
        if self.is_low_rank:
            proj = diff @ self._U_k
            return (proj.pow(2) * self._diff_scale_sq).sum(dim=1) + self._scale_tail_sq * diff.pow(2).sum(dim=1)
        return (diff @ self._precision * diff).sum(dim=1)

    def state_dict(self) -> dict:
        return {"mean": self.mean, "eigenvalues": self._eigenvalues,
                "eigenvectors": self._eigenvectors, "reg_eigenvalues": self._reg_eigenvalues,
                "n_samples": torch.tensor(self._n_samples),
                "total_trace": torch.tensor(self._total_trace)}

    def load_state_dict(self, sd: dict) -> None:
        self.__init__(sd["mean"], sd["eigenvalues"], sd["eigenvectors"],
                      sd["reg_eigenvalues"],
                      int(sd["n_samples"].item()), float(sd["total_trace"].item()))
