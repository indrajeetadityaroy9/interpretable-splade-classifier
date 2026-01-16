"""
SPLADE Classifier Setup

Automatically builds CUDA C++ kernels if nvcc is available.
Falls back gracefully to Triton/PyTorch if CUDA compilation fails.
"""

import os
import sys
import subprocess
from pathlib import Path
from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext

# Check if CUDA is available
def cuda_available():
    """Check if nvcc compiler is available."""
    try:
        result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def get_cuda_extensions():
    """Get CUDA extension configuration if available."""
    if not cuda_available():
        print("NOTE: nvcc not found. CUDA kernels will not be built.")
        print("      The package will use Triton/PyTorch fallback.")
        return []

    try:
        from torch.utils.cpp_extension import CUDAExtension
    except ImportError:
        print("NOTE: PyTorch not installed. CUDA kernels will not be built.")
        print("      Install PyTorch first, then reinstall this package.")
        return []

    # Get pybind11 include path
    try:
        import pybind11
        pybind11_include = pybind11.get_include()
    except ImportError:
        pybind11_include = None

    # Use relative paths (required by setuptools)
    cuda_dir = "src/ops/cuda"

    include_dirs = [cuda_dir]
    if pybind11_include:
        include_dirs.append(pybind11_include)

    return [
        CUDAExtension(
            name="splade_cuda_kernels",
            sources=[
                f"{cuda_dir}/bindings.cpp",
                f"{cuda_dir}/splade_kernels.cu",
            ],
            include_dirs=include_dirs,
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "-gencode=arch=compute_70,code=sm_70",  # V100
                    "-gencode=arch=compute_75,code=sm_75",  # T4
                    "-gencode=arch=compute_80,code=sm_80",  # A100
                    "-gencode=arch=compute_86,code=sm_86",  # RTX 3090
                    "-gencode=arch=compute_89,code=sm_89",  # RTX 4090
                    "-gencode=arch=compute_90,code=sm_90",  # H100
                    "--use_fast_math",
                    "-Xcompiler", "-fno-strict-aliasing",
                ],
            },
        )
    ]


class BuildExtOptional(build_ext):
    """
    Custom build_ext that makes CUDA compilation optional.

    If CUDA compilation fails, it prints a warning but doesn't
    fail the entire installation. Users can still use Triton/PyTorch.
    """

    def build_extensions(self):
        if not self.extensions:
            return

        try:
            super().build_extensions()
            print("\n" + "="*60)
            print("CUDA kernels built successfully!")
            print("SPLADE will use CUDA C++ backend (7.4x faster than PyTorch)")
            print("="*60 + "\n")
        except Exception as e:
            print("\n" + "="*60)
            print(f"WARNING: CUDA kernel compilation failed: {e}")
            print("")
            print("The package will still work using Triton/PyTorch fallback.")
            print("For best performance, ensure:")
            print("  1. CUDA toolkit is installed (nvcc available)")
            print("  2. PyTorch is installed with CUDA support")
            print("  3. pybind11 is installed: pip install pybind11")
            print("")
            print("Then reinstall: pip install -e . --no-build-isolation")
            print("="*60 + "\n")

    def copy_extensions_to_source(self):
        """Copy built extensions to the source tree for editable installs."""
        if not self.extensions:
            return

        try:
            import shutil
            # Get the built extension
            for ext in self.extensions:
                fullname = self.get_ext_fullname(ext.name)
                filename = self.get_ext_filename(fullname)

                # Source: build directory
                src_path = os.path.join(self.build_lib, filename)

                # Destination: src/ops/cuda/ directory (relative path)
                dst_dir = os.path.join(os.path.dirname(__file__) or ".", "src", "ops", "cuda")
                dst_path = os.path.join(dst_dir, os.path.basename(filename))

                if os.path.exists(src_path):
                    shutil.copy2(src_path, dst_path)
                    print(f"Copied {filename} to {dst_dir}")
        except Exception as e:
            print(f"Warning: Could not copy extension to source: {e}")


def get_cmdclass():
    """Get command class, using PyTorch's BuildExtension if available."""
    try:
        from torch.utils.cpp_extension import BuildExtension

        class CombinedBuildExt(BuildExtOptional, BuildExtension):
            """Combines optional build with PyTorch's BuildExtension."""
            pass

        return {"build_ext": CombinedBuildExt}
    except ImportError:
        return {"build_ext": BuildExtOptional}


# Get extensions (empty list if CUDA not available)
ext_modules = get_cuda_extensions()

setup(
    ext_modules=ext_modules,
    cmdclass=get_cmdclass(),
)
