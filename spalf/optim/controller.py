"""Dual controller: EMA filtering, PI dual update, non-monotone penalty, active-subset aware."""

import math
import torch
from torch import Tensor


class DualController:
    def __init__(self, initial_violations: Tensor, rho_0: float) -> None:
        n = initial_violations.shape[0]
        self.n_constraints = n

        self._ema1 = torch.zeros(n, device="cuda")
        self._lambdas = torch.zeros(n, device="cuda")
        self._v_prev = torch.zeros(n, device="cuda")
        self._etas = initial_violations.abs().rsqrt()
        self._rhos = torch.full((n,), rho_0, device="cuda")

        self._n_updates = 0
        self._log_bc = 0.0
        self._n_dual_updates = 0
        self._next_slow_step = 0

    def _adaptive_beta(self, n_active: int) -> float:
        """Harmonic gain t/(t+1) with n_active burn-in (Robbins-Monro)."""
        t = max(self._n_updates, n_active)
        return t / (t + 1)

    def update(self, active_violations: Tensor, idx: list[int]) -> None:
        """Per-step EMA update on the active subset only."""
        self._n_updates += 1
        beta = self._adaptive_beta(len(idx))
        self._log_bc += math.log(beta)
        self._ema1[idx] = beta * self._ema1[idx] + (1.0 - beta) * active_violations.detach()

    def step(self, idx: list[int]) -> None:
        """PI dual update + non-monotone ρ recompute on active slots only."""
        v = self.v_ema[idx]
        self._lambdas[idx] = (
            self._lambdas[idx] + self._rhos[idx] * (2.0 * v - self._v_prev[idx])
        ).clamp_min(0.0)
        self._v_prev[idx] = v.clone()
        self._rhos[idx] = self._etas[idx] * v.abs().rsqrt()

    def activate(self, index: int, initial_violation: float) -> None:
        """Bring a dormant constraint online: reset its η, ema1, v_prev."""
        self._etas[index] = abs(initial_violation) ** -0.5
        self._ema1[index] = 0.0
        self._v_prev[index] = 0.0

    def should_do_slow_update(self, step: int) -> bool:
        """Triangular spacing: ~√(2T) slow updates over T steps."""
        if step >= self._next_slow_step:
            self._n_dual_updates += 1
            self._next_slow_step = step + self._n_dual_updates
            return True
        return False

    @property
    def v_ema(self) -> Tensor:
        return self._ema1 / (1.0 - math.exp(self._log_bc))

    def state_dict(self) -> dict:
        return {
            "ema1": self._ema1, "n_updates": self._n_updates, "log_bc": self._log_bc,
            "lambdas": self._lambdas, "v_prev": self._v_prev,
            "etas": self._etas, "rhos": self._rhos,
            "n_dual_updates": self._n_dual_updates, "next_slow_step": self._next_slow_step,
        }

    def load_state_dict(self, sd: dict) -> None:
        self._ema1 = sd["ema1"].cuda()
        self._lambdas = sd["lambdas"].cuda()
        self._v_prev = sd["v_prev"].cuda()
        self._etas = sd["etas"].cuda()
        self._rhos = sd["rhos"].cuda()
        self._n_updates = int(sd["n_updates"])
        self._log_bc = float(sd["log_bc"])
        self._n_dual_updates = int(sd["n_dual_updates"])
        self._next_slow_step = int(sd["next_slow_step"])
