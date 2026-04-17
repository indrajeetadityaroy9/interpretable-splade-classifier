"""Dual controller: EMA filtering, PI dual update, monotone penalty."""

import math
import torch
from torch import Tensor


class DualController:
    def __init__(self, initial_violations: Tensor, rho_0: float, n_primal: int) -> None:
        n = initial_violations.shape[0]
        self.n_constraints = n
        self.n_primal = n_primal

        self._ema1 = torch.zeros(n, device="cuda")
        self._lambdas = torch.zeros(n, device="cuda")
        self._v_prev = torch.zeros(n, device="cuda")
        self._etas = initial_violations.abs().rsqrt()
        self._rhos = torch.full((n,), rho_0, device="cuda")

        self._n_updates = 0
        self._log_bc = 0.0
        self._n_dual_updates = 0
        self._next_slow_step = 0

    def _adaptive_beta(self) -> float:
        t = max(self._n_updates, self.n_constraints)
        return t / (t + 1)

    def update(self, violations: Tensor) -> None:
        self._n_updates += 1
        beta = self._adaptive_beta()
        self._log_bc += math.log(beta)
        self._ema1.mul_(beta).add_(violations.detach(), alpha=1.0 - beta)

    def step(self) -> None:
        v = self.v_ema
        self._lambdas = (self._lambdas + self._rhos * (2.0 * v - self._v_prev)).clamp_min(0.0)
        self._v_prev = v.clone()
        self._rhos = self._etas * v.abs().rsqrt()

    def recalibrate(self, index: int, initial_violation: float) -> None:
        self._etas[index] = abs(initial_violation) ** -0.5
        self._ema1[index] = 0.0
        self._v_prev[index] = 0.0

    def should_do_slow_update(self, step: int) -> bool:
        """Triangular spacing: interval grows as n_dual_updates, ~√(2T) total."""
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
            "n_primal": self.n_primal,
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
        self.n_primal = int(sd["n_primal"])
