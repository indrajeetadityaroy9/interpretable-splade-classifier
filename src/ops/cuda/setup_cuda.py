"""
Build script for SPLADE CUDA extensions.

Usage:
    python src/ops/cuda/setup_cuda.py build_ext --inplace

Or install as package:
    pip install -e . --config-settings="--build-option=--cuda"
"""

import os
import sys
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Get the directory containing this script
CUDA_DIR = Path(__file__).parent.absolute()
SRC_DIR = CUDA_DIR.parent.parent.parent  # Project root


def get_cuda_extension():
    """Configure the CUDA extension with optimized compiler flags."""

    # Source files
    sources = [
        str(CUDA_DIR / "bindings.cpp"),
        str(CUDA_DIR / "splade_kernels.cu"),
    ]

    # Get pybind11 include path
    try:
        import pybind11
        pybind11_include = pybind11.get_include()
    except ImportError:
        pybind11_include = None

    # Include directories
    include_dirs = [str(CUDA_DIR)]
    if pybind11_include:
        include_dirs.append(pybind11_include)

    # Compiler flags
    extra_compile_args = {
        "cxx": [
            "-O3",
            "-std=c++17",
        ],
        "nvcc": [
            "-O3",
            "-std=c++17",
            # Architecture flags for modern GPUs
            "-gencode=arch=compute_70,code=sm_70",  # V100
            "-gencode=arch=compute_75,code=sm_75",  # T4
            "-gencode=arch=compute_80,code=sm_80",  # A100
            "-gencode=arch=compute_86,code=sm_86",  # RTX 3090
            "-gencode=arch=compute_89,code=sm_89",  # RTX 4090
            "-gencode=arch=compute_90,code=sm_90",  # H100
            # Optimization flags
            "--use_fast_math",
            "--extra-device-vectorization",
            "-lineinfo",  # For profiling
            # Suppress warnings
            "-Xcompiler", "-fno-strict-aliasing",
        ],
    }

    return CUDAExtension(
        name="splade_cuda_kernels",
        sources=sources,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
    )


if __name__ == "__main__":
    setup(
        name="splade_cuda_kernels",
        version="0.1.0",
        description="CUDA kernels for SPLADE neural sparse classifier",
        ext_modules=[get_cuda_extension()],
        cmdclass={"build_ext": BuildExtension},
        python_requires=">=3.9",
    )
