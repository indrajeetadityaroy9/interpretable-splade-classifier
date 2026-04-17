import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _fused_jumprelu_fwd_kernel(pre_act_ptr, theta_ptr, z_ptr, gate_ptr, l0_ptr,
                               n_elements, F: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(pre_act_ptr + offsets, mask=mask, other=0.0)
    theta = tl.load(theta_ptr + (offsets % F), mask=mask, other=0.0)
    gate = (x > theta).to(tl.float32)
    tl.store(z_ptr + offsets, x * gate, mask=mask)
    tl.store(gate_ptr + offsets, gate, mask=mask)
    tl.store(l0_ptr + offsets, gate, mask=mask)


@triton.jit
def _moreau_ste_bwd_kernel(grad_l0_ptr, pre_act_ptr, theta_ptr, gamma_ptr,
                           grad_pre_act_ptr, grad_theta_ptr,
                           n_elements, F: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """L0 backward: Moreau proximal gradient of Heaviside. Zone: -√(2γ) < u ≤ 0."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    grad_l0 = tl.load(grad_l0_ptr + offsets, mask=mask, other=0.0)
    x = tl.load(pre_act_ptr + offsets, mask=mask, other=0.0)
    feat_idx = offsets % F
    theta = tl.load(theta_ptr + feat_idx, mask=mask, other=0.0)
    gamma = tl.load(gamma_ptr + feat_idx, mask=mask, other=1.0)

    u = x - theta
    in_zone = (u > -tl.sqrt(2.0 * gamma)) & (u <= 0.0)
    moreau = tl.where(in_zone, -u / gamma, 0.0)

    tl.store(grad_pre_act_ptr + offsets, grad_l0 * moreau, mask=mask)
    tl.atomic_add(grad_theta_ptr + feat_idx, -grad_l0 * moreau, mask=mask)


@triton.jit
def _recon_ste_bwd_kernel(grad_z_ptr, pre_act_ptr, theta_ptr, gamma_ptr,
                          grad_theta_ptr,
                          n_elements, F: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """Reconstruction STE gradient to θ (JumpReLU paper Eq. 11) via Moreau kernel."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    grad_z = tl.load(grad_z_ptr + offsets, mask=mask, other=0.0)
    x = tl.load(pre_act_ptr + offsets, mask=mask, other=0.0)
    feat_idx = offsets % F
    theta = tl.load(theta_ptr + feat_idx, mask=mask, other=0.0)
    gamma = tl.load(gamma_ptr + feat_idx, mask=mask, other=1.0)

    u = x - theta
    in_zone = (u > -tl.sqrt(2.0 * gamma)) & (u <= 0.0)
    moreau = tl.where(in_zone, -u / gamma, 0.0)
    tl.atomic_add(grad_theta_ptr + feat_idx, -grad_z * x * moreau, mask=mask)


class FusedJumpReLUFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, pre_act: Tensor, log_threshold: Tensor, gamma: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        B, F = pre_act.shape
        n_elements = B * F
        # [256, 2048]: lower bound = full warp occupancy; upper = shared-mem limit.
        block = max(256, min(2048, triton.next_power_of_2(F)))

        theta = log_threshold.exp()
        z = torch.empty_like(pre_act)
        gate = torch.empty_like(pre_act)
        l0 = torch.empty_like(pre_act)

        grid = ((n_elements + block - 1) // block,)
        _fused_jumprelu_fwd_kernel[grid](pre_act, theta, z, gate, l0, n_elements, F=F, BLOCK_SIZE=block)

        ctx.save_for_backward(pre_act, theta, gamma, gate)
        ctx.F, ctx.n_elements, ctx.block = F, n_elements, block
        return z, gate.detach(), l0

    @staticmethod
    def backward(ctx, grad_z: Tensor, _grad_gate: Tensor, grad_l0: Tensor):
        pre_act, theta, gamma, gate = ctx.saved_tensors
        F, n_elements, block = ctx.F, ctx.n_elements, ctx.block

        grad_pre_act_from_l0 = torch.zeros_like(pre_act)
        grad_theta = torch.zeros(F, device="cuda", dtype=pre_act.dtype)
        grid = ((n_elements + block - 1) // block,)

        _moreau_ste_bwd_kernel[grid](grad_l0, pre_act, theta, gamma,
                                     grad_pre_act_from_l0, grad_theta,
                                     n_elements, F=F, BLOCK_SIZE=block)
        _recon_ste_bwd_kernel[grid](grad_z, pre_act, theta, gamma, grad_theta,
                                    n_elements, F=F, BLOCK_SIZE=block)

        return grad_z * gate + grad_pre_act_from_l0, grad_theta * theta, None
