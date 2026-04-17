"""Constraint abstraction: Ctx, Constraint, ConstraintSet, and violation functions."""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


@dataclass
class Ctx:
    """Per-step forward-pass state passed to constraint violation functions."""
    sae: Any
    whitener: Any
    W_vocab: Tensor
    x: Tensor
    x_hat: Tensor
    z: Tensor
    kl_div: Tensor | None = None


@dataclass
class Constraint:
    name: str
    fn: Callable[[Ctx], Tensor]                          # returns raw violation scalar
    tau: float
    onset: Callable[[Tensor, Ctx], bool] | None = None    # None ⇒ always-active primal
    needs_kl: bool = False
    active: bool = True


class ConstraintSet:
    def __init__(self, constraints: list[Constraint]) -> None:
        self.cs = constraints

    def active_indices(self) -> list[int]:
        return [i for i, c in enumerate(self.cs) if c.active]

    def primal_indices(self) -> list[int]:
        return [i for i, c in enumerate(self.cs) if c.onset is None]

    def active_needs_kl(self) -> bool:
        return any(c.active and c.needs_kl for c in self.cs)

    def violations(self, ctx: Ctx) -> Tensor:
        """Stack active-constraint violations (fn(ctx) − tau) in constraint order."""
        return torch.stack([c.fn(ctx) - c.tau for c in self.cs if c.active])

    def index_of(self, name: str) -> int:
        for i, c in enumerate(self.cs):
            if c.name == name:
                return i
        raise KeyError(name)

    def by_name(self, name: str) -> Constraint:
        return self.cs[self.index_of(name)]


def compute_orthogonality_violation(z: Tensor, W_dec_A: Tensor, W_dec_B: Tensor) -> Tensor:
    """Differentiable co-activation orthogonality via column sampling (raw, unshifted)."""
    V = W_dec_A.shape[1]
    F_total = V + W_dec_B.shape[1]
    k = min(F_total, math.ceil(math.sqrt(F_total)))

    idx = torch.randint(F_total, (k,), device="cuda")
    cols_a = W_dec_A[:, idx.clamp(max=V - 1)]
    cols_b = W_dec_B[:, (idx - V).clamp(min=0)]
    W_sub = torch.where((idx < V).unsqueeze(0), cols_a, cols_b)
    W_hat = W_sub / W_sub.norm(dim=0, keepdim=True)

    G_sq = (W_hat.T @ W_hat).pow(2)
    active = (z[:, idx] > 0).float()
    C = active.T @ active

    weighted = G_sq * C
    num = weighted.sum() - weighted.diagonal().sum()
    denom = C.sum() - C.diagonal().sum() + torch.finfo(C.dtype).eps
    return num / denom
