"""
Optimized GPU operations for SPLADE.

Backend Priority:
1. CUDA C++ kernels (fastest, requires compilation)
2. Triton kernels (fast, JIT compiled)
3. PyTorch fallback (always available)

Usage:
    from src.ops import splade_aggregate, flops_reg

    # Automatic backend selection (best available)
    doc_vectors = splade_aggregate(logits, attention_mask)

    # Force specific backend
    doc_vectors = splade_aggregate(logits, attention_mask, backend="pytorch")

Supported Operations:
    - splade_aggregate: Fused log1p + relu + mask + max_pool
    - flops_reg: FLOPS regularization loss

Hardware Requirements:
    - CUDA compute capability >= 7.0 (Volta or newer)
    - For CUDA kernels: Build with setup_cuda.py
    - For Triton: pip install triton>=2.0
"""

# Check CUDA C++ kernels availability
try:
    from .cuda import CUDA_AVAILABLE, CUDA_INFO
except ImportError:
    CUDA_AVAILABLE = False
    CUDA_INFO = None

# Check Triton availability
try:
    import triton
    TRITON_AVAILABLE = True
    TRITON_VERSION = triton.__version__
except ImportError:
    TRITON_AVAILABLE = False
    TRITON_VERSION = None

# Import public API
from .splade_kernels import splade_aggregate, flops_reg


__all__ = [
    "splade_aggregate",
    "flops_reg",
    "CUDA_AVAILABLE",
    "TRITON_AVAILABLE",
]
