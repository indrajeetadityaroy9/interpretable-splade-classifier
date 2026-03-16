"""Dual controller: TEMA filtering, PI dual update, non-monotone CAPU."""

import math

import torch
from torch import Tensor

class DualController:
    """TEMA + PI dual + non-monotone CAPU."""

    def __init__(
        self,
        initial_violations: Tensor,
        rho_0: float,
        total_steps: int,
        n_primal: int,
    ) -> None:
        self.n_constraints = initial_violations.shape[0]
        self.n_primal = n_primal
        self._total_steps = total_steps

        self._ema1 = torch.zeros(self.n_constraints, device="cuda")
        self._ema2 = torch.zeros(self.n_constraints, device="cuda")
        self._ema3 = torch.zeros(self.n_constraints, device="cuda")
        self._n_updates = 0
        self._log_bc = 0.0

        self._lambdas = torch.zeros(self.n_constraints, device="cuda")
        self._v_prev = torch.zeros(self.n_constraints, device="cuda")

        iv = initial_violations.abs()
        self._etas = iv.rsqrt()
        self._v_bar = torch.ones(self.n_constraints, device="cuda")
        self._rhos = torch.full((self.n_constraints,), rho_0, device="cuda")
        self._rho_0 = rho_0
        self._rho_floor = rho_0 * (iv.mean() / iv)

        self._n_dual_updates = 0
        self._next_slow_step = 0

    def _adaptive_beta(self) -> float:
        """Harmonic gain: t/(t+1) with n_constraints burn-in (Robbins-Monro)."""
        t = max(self._n_updates, self.n_constraints)
        return t / (t + 1)

    def update(self, violations: Tensor) -> None:
        """Per-step EMA and v_bar update from raw violations."""
        v = violations.detach()
        self._n_updates += 1
        beta = self._adaptive_beta()
        self._log_bc += math.log(beta)

        self._ema1 = beta * self._ema1 + (1.0 - beta) * v
        self._ema2 = beta * self._ema2 + (1.0 - beta) * self._ema1
        self._ema3 = beta * self._ema3 + (1.0 - beta) * self._ema2
        self._v_bar = beta * self._v_bar + (1.0 - beta) * v.pow(2)

    def step(self) -> None:
        """PI dual update + non-monotone CAPU."""
        v = self.v_ema
        self._lambdas = torch.clamp(
            self._lambdas + self._rhos * (2.0 * v - self._v_prev),
            min=0.0,
        )
        self._v_prev = v.clone()
        self._rhos = torch.clamp(self._etas * self._v_bar.rsqrt(), min=self._rho_floor)

    def recalibrate(self, index: int, initial_violation: float) -> None:
        """Reset constraint state at index (used at KL onset)."""
        self._etas[index] = abs(initial_violation) ** -0.5
        self._v_bar[index] = 1.0
        self._rho_floor[index] = self._rho_0
        self._ema1[index] = 0.0
        self._ema2[index] = 0.0
        self._ema3[index] = 0.0
        self._v_prev[index] = 0.0

    def should_do_slow_update(self, step: int) -> bool:
        """Triangular spacing: interval grows as n_dual_updates, ~sqrt(2T) total."""
        if step >= self._next_slow_step:
            self._n_dual_updates += 1
            self._next_slow_step = step + self._n_dual_updates
            return True
        return False

    @property
    def v_ema(self) -> Tensor:
        """Bias-corrected TEMA: 3·EMA₁ − 3·EMA₂ + EMA₃."""
        bc = 1.0 - math.exp(self._log_bc)
        e1 = self._ema1 / bc
        e2 = self._ema2 / bc
        e3 = self._ema3 / bc
        return 3.0 * e1 - 3.0 * e2 + e3

    def state_dict(self) -> dict:
        return {
            "ema1": self._ema1,
            "ema2": self._ema2,
            "ema3": self._ema3,
            "n_updates": self._n_updates,
            "log_bc": self._log_bc,
            "lambdas": self._lambdas,
            "v_prev": self._v_prev,
            "etas": self._etas,
            "v_bar": self._v_bar,
            "rhos": self._rhos,
            "rho_0": self._rho_0,
            "rho_floor": self._rho_floor,
            "n_dual_updates": self._n_dual_updates,
            "next_slow_step": self._next_slow_step,
            "total_steps": self._total_steps,
            "n_primal": self.n_primal,
        }

    def load_state_dict(self, sd: dict) -> None:
        self._ema1 = sd["ema1"].cuda()
        self._ema2 = sd["ema2"].cuda()
        self._ema3 = sd["ema3"].cuda()
        self._n_updates = int(sd["n_updates"])
        self._log_bc = float(sd["log_bc"])
        self._lambdas = sd["lambdas"].cuda()
        self._v_prev = sd["v_prev"].cuda()
        self._etas = sd["etas"].cuda()
        self._v_bar = sd["v_bar"].cuda()
        self._rhos = sd["rhos"].cuda()
        self._rho_0 = float(sd["rho_0"])
        self._rho_floor = sd["rho_floor"].cuda()
        self._n_dual_updates = int(sd["n_dual_updates"])
        self._next_slow_step = int(sd["next_slow_step"])
        self._total_steps = int(sd["total_steps"])
        self.n_primal = int(sd["n_primal"])
