# SPLADE Classifier
A fast, interpretable sparse text classifier with an sklearn-compatible API. Built on [SPLADE](https://arxiv.org/abs/2107.05720) (Sparse Lexical and Expansion Model), this library provides neural sparse representations that outperform traditional TF-IDF while remaining fully interpretable.

## Features

- **Sklearn-compatible API** - `fit()`, `predict()`, `transform()` just like scikit-learn
- **Interpretable** - See exactly which vocabulary terms drive each prediction
- **Sparse & Fast** - 78%+ sparsity with CUDA/Triton-accelerated inference (7.4x speedup)
- **Better than TF-IDF** - Outperforms traditional sparse baselines while maintaining interpretability

## Installation

```bash
# Standard install (auto-builds CUDA kernels if nvcc available)
pip install .

# With Triton acceleration
pip install ".[gpu]"

# Development install
pip install -e . --no-build-isolation
make build-cuda  # Build CUDA kernels
```

## Quick Start

```python
from src.models import SPLADEClassifier

# Train
clf = SPLADEClassifier()
clf.fit(train_texts, train_labels, epochs=5)

# Predict
predictions = clf.predict(test_texts)

# Explain predictions
clf.print_explanation("This movie was fantastic!")
```

## Benchmark Results

Evaluated on AG News (4-class news classification) with 2,000 training samples (preliminary results):

| Model | Accuracy | F1 Score | Sparsity | Interpretable |
|-------|----------|----------|----------|---------------|
| **SPLADE (Ours)** | **90.1%** | **0.901** | 78.1% | Yes |
| TF-IDF + LogReg | 85.0% | 0.848 | 99.7% | Yes |
| BERT-base (reference) | ~94% | ~0.94 | N/A | No |

**Key tradeoffs:**
- SPLADE vs TF-IDF: +5.1% accuracy while remaining interpretable
- SPLADE vs BERT: ~4% lower accuracy, but provides term-level explanations

See `notebooks/splade_sklearn_benchmark.ipynb` for the comparison methodology.

## API Reference

### SPLADEClassifier

```python
clf = SPLADEClassifier(
    model_name="distilbert-base-uncased",  # Backbone model
    max_length=128,                         # Max sequence length
    batch_size=32,                          # Training batch size
    learning_rate=2e-5,                     # Learning rate
    flops_lambda=1e-4,                      # Sparsity regularization
    random_state=42,                        # Reproducibility seed
)
```

#### Methods

| Method | Description |
|--------|-------------|
| `fit(X, y, epochs=5)` | Train on texts and labels |
| `predict(X)` | Get class predictions |
| `predict_proba(X)` | Get class probabilities |
| `transform(X)` | Get sparse SPLADE vectors |
| `score(X, y)` | Compute accuracy |
| `save(path)` / `load(path)` | Model persistence |

#### Interpretability

| Method | Description |
|--------|-------------|
| `explain(text)` | Get top weighted terms |
| `explain_prediction(text)` | Full prediction breakdown |
| `compare_texts(text1, text2)` | Compare representations |
| `print_explanation(text)` | Pretty-print explanation |

## How It Works

SPLADE uses a masked language model (DistilBERT) to produce sparse vocabulary-sized vectors:

1. **Encode**: Text → DistilBERT → Token logits `[batch, seq_len, vocab_size]`
2. **Activate**: `log(1 + ReLU(logits))` for log-saturation
3. **Pool**: Max-pool over sequence → Sparse document vector
4. **Classify**: Linear layer on sparse vector

The resulting vectors are:
- **Sparse**: ~98% zeros (efficient storage/retrieval)
- **Interpretable**: Each dimension = vocabulary term weight
- **Expandable**: Semantically related terms get non-zero weights

## Performance Optimization

GPU kernels provide significant speedup (3-tier backend: CUDA C++ → Triton → PyTorch):

| Backend | SPLADE Aggregation | Speedup |
|---------|-------------------|---------|
| PyTorch (baseline) | 1.28 ms | 1.0x |
| Triton | 0.22 ms | 5.9x |
| **CUDA C++** | **0.17 ms** | **7.4x** |

*Measured on NVIDIA H100. Backend is auto-selected (best available).*

```bash
# Check available backends
python -c "from src.ops import get_backend_info; print(get_backend_info())"
```
