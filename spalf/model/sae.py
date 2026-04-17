import math
import torch
import torch.nn as nn
from torch import Tensor

from spalf.model.kernel import FusedJumpReLUFunction

# IQR-to-σ ratio for standard normal: 2·Φ⁻¹(0.75) = 2·√2·erfinv(0.5).
_IQR_SIGMA: float = 2.0 * math.sqrt(2.0) * math.erfinv(0.5)


class StratifiedSAE(nn.Module):
    """Sparse autoencoder with anchored and free decoder strata."""

    def __init__(self, d: int, F: int, V: int) -> None:
        super().__init__()
        self.d, self.F, self.V = d, F, V
        self.F_free = F - V

        self.W_enc = nn.Parameter(torch.empty(F, d))
        self.b_enc = nn.Parameter(torch.zeros(F))
        self.W_dec_A = nn.Parameter(torch.empty(d, V))
        self.W_dec_B = nn.Parameter(torch.empty(d, self.F_free))
        self.b_dec = nn.Parameter(torch.zeros(d))
        self.log_threshold = nn.Parameter(torch.zeros(F))

        self.register_buffer("gamma", torch.ones(F))
        self.register_buffer("gamma_init", torch.ones(F))
        self.register_buffer("dead_counter", torch.zeros(F, dtype=torch.long))

    def forward(self, x_tilde: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        pre_act = x_tilde @ self.W_enc.T + self.b_enc
        z, gate_mask, l0_probs = FusedJumpReLUFunction.apply(pre_act, self.log_threshold, self.gamma)
        x_hat = z[:, :self.V] @ self.W_dec_A.T + z[:, self.V:] @ self.W_dec_B.T + self.b_dec

        # Discretization penalty u²/γ over Moreau transition zone (dimensionless).
        u = pre_act - self.log_threshold.detach().exp()
        in_zone = (u > -(2.0 * self.gamma).sqrt()) & (u <= 0.0)
        disc_penalty = (u.pow(2) / self.gamma) * in_zone.float()
        return x_hat, z, gate_mask, l0_probs, disc_penalty

    @torch.no_grad()
    def recalibrate_gamma(self, pre_act: Tensor) -> None:
        """Density-matched Moreau bandwidth: c = q/(IQR_σ·φ(z_q))."""
        thresholds = self.log_threshold.exp()

        global_iqr = torch.quantile(pre_act, 0.75, dim=0) - torch.quantile(pre_act, 0.25, dim=0)
        mask = (pre_act > (thresholds - global_iqr)) & (pre_act < (thresholds + global_iqr))

        masked = pre_act.masked_fill(~mask, float("nan"))
        iqr = torch.nanquantile(masked, 0.75, dim=0) - torch.nanquantile(masked, 0.25, dim=0)
        iqr = torch.where(iqr.isnan(), global_iqr, iqr)

        q = (pre_act > thresholds).float().mean(dim=0).clamp_min(1.0 / self.F)
        z_q = torch.erfinv(1.0 - 2.0 * q) * math.sqrt(2)
        phi_zq = torch.exp(-z_q.pow(2) / 2.0) / math.sqrt(2 * math.pi)
        c = (q / (_IQR_SIGMA * phi_zq)).clamp_max(1.0)

        self.gamma.copy_((c * iqr).pow(2) / 2.0)

    @torch.no_grad()
    def normalize_free_decoder(self) -> None:
        self.W_dec_B.div_(self.W_dec_B.norm(dim=0, keepdim=True))

    @torch.no_grad()
    def update_dead_counts(self, gate_mask: Tensor) -> None:
        fired = gate_mask.any(dim=0)
        self.dead_counter[fired] = 0
        self.dead_counter[~fired] += 1

    @torch.no_grad()
    def resample_dead_features(self, x_tilde: Tensor, x_hat: Tensor, dead_threshold: int, L0_target: int) -> int:
        """Reinitialize dead FREE features toward reconstruction error."""
        dead_mask = self.dead_counter > dead_threshold
        dead_mask[:self.V] = False
        dead_indices = dead_mask.nonzero(as_tuple=True)[0]
        n_dead = dead_indices.shape[0]
        if n_dead == 0:
            return 0

        residuals = x_tilde - x_hat
        k = min(n_dead, residuals.shape[0])
        _, top_idx = residuals.norm(dim=1).topk(k)
        directions = residuals[top_idx]
        directions = directions / directions.norm(dim=1, keepdim=True)

        active = dead_indices[:k]
        self.W_enc.data[active] = directions
        self.W_dec_B.data[:, active - self.V] = directions.T

        pre_acts = x_tilde @ directions.T
        thresholds = torch.quantile(pre_acts, 1.0 - L0_target / self.F, dim=0)
        self.log_threshold.data[active] = thresholds.clamp_min(torch.finfo(pre_acts.dtype).eps).log()
        self.dead_counter[active] = 0
        return n_dead
