# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Lexical-SAE (Circuit-Integrated SPLADE): a supervised exact sparse autoencoder for text classification. Combines SPLADE's sparse lexical bottleneck with exact per-token attribution via Direct Logit Attribution (DLA). Key SOTA claims: (1) zero reconstruction error unlike post-hoc SAEs, (2) intrinsic alignment via training-time constraints, (3) surgical concept unlearning with mathematical guarantees. A 2-layer ReLU MLP creates piecewise-linear structure enabling zero-cost algebraic attribution.

## Commands

```bash
# Install
pip install -e .

# Run experiments
python -m splade.scripts.run_experiment --config experiments/paper/sst2.yaml
python -m splade.scripts.run_ablation --config experiments/paper/sst2_ablation.yaml
python -m splade.scripts.run_multi_dataset --config experiments/paper/multi_dataset.yaml

# Quick verification (toy-sized, fast)
python -m splade.scripts.run_experiment --config experiments/verify_full.yaml

# Hero experiments
python -m splade.scripts.run_faithfulness --config experiments/civilcomments.yaml  # Exp A: removability
python -m splade.scripts.run_surgery --config experiments/surgery.yaml             # Exp B: bias lobotomy
python -m splade.scripts.run_long_context --config experiments/long_context.yaml   # Exp C: needle in haystack

# Tests
pytest tests/
pytest tests/ -k "not slow"          # skip network-dependent tests
pytest tests/test_circuits.py -k "test_name"  # single test
```

## Architecture

### Forward Pass
```
Input → BERT → Vocab Transform (Dense+GELU+LN) → MLM Logits → DReLU → Sparse Vector (max-pool over sequence) → ReLU MLP → Logits
```
The ReLU MLP (`classifier_fc1` → `classifier_fc2`) also returns `W_eff` and `b_eff` such that `logit[b,c] = sum_j[s[b,j] * W_eff[b,c,j]] + b_eff[b,c]` holds exactly. This is the core DLA identity.

### Training Objective
Three circuit losses optimized via GECO constrained optimization:
- **Completeness**: Circuit (top-k% attributed dims) must preserve predictions
- **Separation**: Per-class attribution centroids must be distinct (cosine similarity)
- **Sharpness**: Attributions must concentrate on few dims (Hoyer sparsity)

GECO constrains CE ≤ τ (auto-derived from warmup or set via `target_accuracy` config) while minimizing the circuit objective.

### Key Modules
- **`splade/models/splade.py`** — `CISModel`: BERT encoder + SPLADE head + ReLU MLP classifier. `classifier_forward()` returns logits + W_eff + b_eff.
- **`splade/circuits/core.py`** — `CircuitState` (logits, sparse_vector, W_eff, b_eff) and `circuit_mask()` (temperature-parameterized soft/hard top-k masking).
- **`splade/circuits/geco.py`** — `GECOController`: dataset-size-invariant Lagrangian optimization. Single multiplier constraining CE loss while minimizing circuit objectives.
- **`splade/circuits/losses.py`** — Three circuit losses: completeness, separation, sharpness.
- **`splade/mechanistic/attribution.py`** — `compute_attribution_tensor()`: the single canonical DLA function used by both training losses and evaluation metrics.
- **`splade/training/loop.py`** — Two-phase training: warmup (CE only, sets GECO threshold tau_ce) then main (GECO-constrained circuit optimization with EMA and early stopping).
- **`splade/evaluation/mechanistic.py`** — Tiered evaluation: Tier 1 (accuracy + DLA verification), Tier 2 (completeness + sparsity), Tier 3 (separation + example circuits). Full metrics (ERASER, explainer comparison, layerwise, SAE) still computed for JSON output.
- **`splade/evaluation/faithfulness.py`** — Removability metric: measures prediction flip rate when top-k DLA-attributed sparse dims are zeroed. Compares DLA vs gradient/IG/attention baselines.
- **`splade/intervene.py`** — Surgical intervention API: `suppress_token_globally()` (permanent weight surgery), `SuppressedCISModel` (reversible wrapper), `evaluate_bias()` (per-identity FPR analysis).
- **`splade/training/constants.py`** — Internal hyperparameters (LR schedule, EMA decay, etc.).

### Config Knobs
Three training knobs exposed via `TrainingConfig` in YAML configs:
- `target_accuracy`: GECO tau override (None = auto from warmup 25th percentile)
- `sparsity_target`: circuit fraction (default 0.1 = top 10% of active dims)
- `warmup_fraction`: fraction of training for CE-only warmup (default 0.2)

### Design Principles
- **Single attribution function**: `compute_attribution_tensor()` is called by training losses AND evaluation — no divergence between training proxy and eval metric.
- **GECO manages trade-offs**: Single Lagrangian multiplier auto-adapts; no manual loss weight tuning.
- **Sparse bottleneck faithfulness**: ~100-200 non-zero dims out of ~30K vocabulary. Zeroing entries causes no distribution shift, enabling clean ERASER evaluation.

## Config System

Configs are YAML files loaded via `splade/config/load.py` into dataclasses in `splade/config/schema.py`. Experiment configs live in `experiments/`. The optional `training:` section controls the 3 knobs above.

## Archived Code

`archive/` contains the B-cos variant (alternative architecture for arbitrary-depth DLA). Not part of the core CIS contribution.

## Datasets

SST-2, AG News, IMDB, Yelp, E-SNLI, CivilComments — all loaded from HuggingFace `datasets` via `splade/data/loader.py`. CivilComments includes identity annotations for bias analysis.
