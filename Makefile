# SPLADE Classifier Makefile
#
# Build and development commands

.PHONY: help install install-dev build-cuda clean test benchmark

# Default target
help:
	@echo "SPLADE Classifier - Build Commands"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  install       Install package with dependencies"
	@echo "  install-dev   Install with development dependencies"
	@echo "  build-cuda    Build CUDA C++ kernels (requires CUDA toolkit)"
	@echo "  clean         Remove build artifacts"
	@echo "  test          Run tests"
	@echo "  benchmark     Run benchmark comparison"
	@echo ""

# Install package
install:
	pip install -r requirements.txt
	pip install -e .

# Install with dev dependencies
install-dev: install
	pip install pytest pytest-benchmark

# Build CUDA kernels
build-cuda:
	@echo "Building CUDA kernels..."
	@if command -v nvcc >/dev/null 2>&1; then \
		cd src/ops/cuda && \
		(test -f ../../../.venv/bin/python && ../../../.venv/bin/python setup_cuda.py build_ext --inplace || python setup_cuda.py build_ext --inplace) && \
		echo "CUDA kernels built successfully!"; \
	else \
		echo "Error: nvcc not found. Please install CUDA toolkit."; \
		exit 1; \
	fi

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf src/*.egg-info/
	rm -rf src/ops/cuda/build/
	rm -f src/ops/*.so
	rm -f src/ops/cuda/*.so
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

# Run tests (when available)
test:
	python -c "from src.ops import get_available_backends; print('Available backends:', get_available_backends())"
	python -c "from src.models import SPLADEClassifier; print('SPLADEClassifier: OK')"
	python -c "from src.data import load_classification_data; print('Data loading: OK')"

# Run benchmark
benchmark:
	python -m src.benchmark --dataset ag_news --epochs 3 --seeds 3 --max_samples 1000

# Quick smoke test
smoke-test:
	python -m src.train --dataset ag_news --epochs 1 --max_train_samples 100

# Check CUDA availability
check-cuda:
	@python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
	@python -c "from src.ops import get_backend_info; print(get_backend_info())"
