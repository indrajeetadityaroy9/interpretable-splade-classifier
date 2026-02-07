# Interpretable SPLADE Classifier

Canonical research pipeline for SPLADE-based text classification and faithfulness benchmarking.

## Canonical Execution Paths

- Training: `python -m scripts.train`
- Evaluation: `python -m scripts.eval`
- Inference API: `src/models/classifier.py::SPLADEClassifier`

The codebase is intentionally single-path:

- Dataset: SST-2 only
- Training orchestration: one script
- Evaluation orchestration: one script (single-seed full benchmark)
- Inference: `predict`, `predict_proba`, `explain`

## Install

```bash
pip install -e .
```

## Train

```bash
python -m scripts.train \
  --train-samples 2000 \
  --test-samples 200 \
  --epochs 2 \
  --batch-size 64 \
  --seed 42
```

## Evaluate

```bash
python -m scripts.eval \
  --train-samples 2000 \
  --test-samples 200 \
  --epochs 2 \
  --batch-size 32 \
  --seed 42
```

Evaluation includes:

- SPLADE native explanations
- Attention / LIME / Integrated Gradients adapter baselines on the same model
- ERASER-family metrics, normalized AOPC, soft perturbation metrics
- F-Fidelity fine-tuned copy metrics
- Adversarial sensitivity metrics

## Inference API

```python
from src.models.classifier import SPLADEClassifier

model = SPLADEClassifier(num_labels=2)
model.fit(train_texts, train_labels, epochs=2)

predictions = model.predict(test_texts)
probabilities = model.predict_proba(test_texts)
explanations = model.explain("This movie was fantastic", top_k=10)
sparse_vectors = model.transform(test_texts)
```

## Core Modules

- `scripts/train.py`: canonical training entrypoint
- `scripts/eval.py`: canonical evaluation entrypoint
- `src/models/classifier.py`: model, training loop, inference API
- `src/models/components.py`: SPLADE aggregation and Triton backward kernel
- `src/training/losses.py`: DF-FLOPS regularization
- `src/training/optim.py`: LR schedule, AGC, early stopping
- `src/training/finetune.py`: F-Fidelity fine-tuning copy
- `src/evaluation/faithfulness.py`: faithfulness metrics
- `src/evaluation/adversarial.py`: adversarial sensitivity metrics
- `src/evaluation/benchmark.py`: benchmark config, execution, result tables
- `src/baselines/splade_adapters.py`: adapter baselines on the same model
- `src/data/loader.py`: SST-2 loader
