#!/bin/bash
set -e

echo "Starting Reproduction Pipeline..."

# 1. Main Benchmarks
echo "Reproducing Main Results..."
for config in experiments/main/*.yaml; do
    echo "Running $config"
    python -m splade.scripts.eval --config "$config"
done

# 2. Ablations
echo "Reproducing Ablations..."
for config in experiments/ablations/*.yaml; do
    echo "Running $config"
    python -m splade.scripts.eval --config "$config"
done

# 3. Sensitivity
echo "Reproducing Sensitivity Analysis..."
for config in experiments/sensitivity/*.yaml; do
    echo "Running $config"
    python -m splade.scripts.eval --config "$config"
done

echo "Reproduction Complete. Results are in results/"
