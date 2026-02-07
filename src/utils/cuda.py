"""Runtime device defaults and seeding."""

import os
import random

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy
import torch

CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA_AVAILABLE else "cpu")
COMPUTE_DTYPE = torch.bfloat16 if CUDA_AVAILABLE else torch.float32
AUTOCAST_DEVICE_TYPE = "cuda" if CUDA_AVAILABLE else "cpu"
AUTOCAST_ENABLED = CUDA_AVAILABLE


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if CUDA_AVAILABLE:
        torch.cuda.manual_seed_all(seed)


if CUDA_AVAILABLE:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
