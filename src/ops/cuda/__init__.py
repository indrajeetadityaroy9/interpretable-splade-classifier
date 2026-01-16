"""
SPLADE CUDA Kernels Python Interface

Provides high-performance CUDA implementations with automatic fallback.

Usage:
    from src.ops.cuda import splade_aggregate_cuda, flops_reg_cuda, CUDA_AVAILABLE

    if CUDA_AVAILABLE:
        output = splade_aggregate_cuda(logits, attention_mask)
        loss = flops_reg_cuda(activations)
"""

import torch
from typing import Optional, Tuple
import sys
from pathlib import Path

# Try to import the compiled CUDA extension
CUDA_AVAILABLE = False
CUDA_INFO = None

# Add the cuda directory to path for the compiled extension
_cuda_dir = Path(__file__).parent
if str(_cuda_dir) not in sys.path:
    sys.path.insert(0, str(_cuda_dir))

try:
    import splade_cuda_kernels as _cuda_kernels
    CUDA_AVAILABLE = True
    CUDA_INFO = _cuda_kernels.get_cuda_info()
except ImportError:
    _cuda_kernels = None


def splade_aggregate_cuda(
    logits: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    CUDA implementation of fused SPLADE aggregation.

    Computes: output[b, v] = max_s(log1p(relu(logits[b, s, v])) * mask[b, s])

    Args:
        logits: MLM logits [batch_size, seq_len, vocab_size]
        attention_mask: Attention mask [batch_size, seq_len]

    Returns:
        Sparse document vectors [batch_size, vocab_size]

    Raises:
        RuntimeError: If CUDA kernels are not available
    """
    if not CUDA_AVAILABLE:
        raise RuntimeError(
            "CUDA kernels not available. Build with: "
            "python src/ops/cuda/setup_cuda.py build_ext --inplace"
        )

    return _cuda_kernels.splade_aggregate(logits, attention_mask)


def flops_reg_cuda(activations: torch.Tensor) -> torch.Tensor:
    """
    CUDA implementation of FLOPS regularization.

    Computes: L_FLOPS = sum_j (mean_i |activations[i,j]|)^2

    Args:
        activations: Sparse activation tensor [batch_size, vocab_size]

    Returns:
        Scalar loss tensor

    Raises:
        RuntimeError: If CUDA kernels are not available
    """
    if not CUDA_AVAILABLE:
        raise RuntimeError(
            "CUDA kernels not available. Build with: "
            "python src/ops/cuda/setup_cuda.py build_ext --inplace"
        )

    return _cuda_kernels.flops_reg(activations)


def topk_cuda(
    vectors: torch.Tensor,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    CUDA implementation of top-k extraction.

    Args:
        vectors: Sparse vectors [batch_size, vocab_size]
        k: Number of top elements to extract

    Returns:
        Tuple of (values [batch, k], indices [batch, k])

    Raises:
        RuntimeError: If CUDA kernels are not available
    """
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA kernels not available.")

    return _cuda_kernels.topk(vectors, k)


def sparse_stats_cuda(
    vectors: torch.Tensor,
    threshold: float = 1e-6,
) -> torch.Tensor:
    """
    CUDA implementation of sparsity statistics.

    Args:
        vectors: Sparse vectors [batch_size, vocab_size]
        threshold: Values with abs < threshold are considered zero

    Returns:
        Statistics [batch_size, 3] containing [num_nonzero, sum_abs, max_abs]
    """
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA kernels not available.")

    return _cuda_kernels.sparse_stats(vectors, threshold)


def get_backend_info() -> str:
    """Get information about the CUDA backend."""
    if CUDA_AVAILABLE:
        return f"CUDA Kernels Available\n{CUDA_INFO}"
    else:
        return "CUDA Kernels Not Available (using Triton/PyTorch fallback)"


__all__ = [
    "CUDA_AVAILABLE",
    "CUDA_INFO",
    "splade_aggregate_cuda",
    "flops_reg_cuda",
    "topk_cuda",
    "sparse_stats_cuda",
    "get_backend_info",
]
