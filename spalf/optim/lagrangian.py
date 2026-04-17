import torch
from torch import Tensor


def compute_augmented_lagrangian(l0_corr: Tensor, violations: Tensor, lambdas: Tensor, rhos: Tensor) -> Tensor:
    """AL-CoLe: l0_corr + Σ ρ_i · Ψ(g_i, λ_i/ρ_i) where Ψ(g,y)=(max(0,2g+y)²-y²)/4."""
    y = lambdas / rhos
    psi = ((2.0 * violations + y).clamp_min(0.0).pow(2) - y.pow(2)) / 4.0
    return l0_corr + (rhos * psi).sum()
