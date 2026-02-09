# Lexical-SAE

**A Supervised Exact Sparse Autoencoder for Interpretable Text Classification**

Lexical-SAE repurposes SPLADE's sparse lexical bottleneck as an intrinsically interpretable classifier with three properties that post-hoc sparse autoencoders cannot provide: **(1)** zero reconstruction error (algebraic identity, not approximation), **(2)** intrinsic alignment via training-time circuit constraints, and **(3)** surgical concept removal with mathematical guarantees.

The feature dictionary is the tokenizer vocabulary itself. No feature splitting ambiguity, no "what is feature #1405?" --- if the model learns a spurious correlation between "Muslim" and toxicity, you can identify it by name, surgically zero its contribution, and mathematically verify the bias is removed.

---

## Method

The ReLU activation mask `D(s) = diag(1[W1*s + b1 > 0])` yields an exact per-input effective weight matrix:

```
W_eff(s) = W2 @ diag(D(s)) @ W1

logit_c  = sum_j [ s_j * W_eff(s)[c,j] ] + b_eff(s)_c      (algebraic identity, not an approximation)
```

This **Direct Logit Attribution (DLA)** decomposes every prediction into per-token contributions at zero cost. Verification error is ~0.001 (machine precision for BF16).

### Architecture

```
Input -> BERT/ModernBERT -> Vocab Transform (Dense+GELU+LN) -> MLM Logits -> DReLU
      -> Sparse Vector (max-pool, ~100-200 of ~30K dims active)
      -> ReLU MLP (fc1 -> fc2) -> Logits + W_eff + b_eff
```

The sparse bottleneck is a **Faithfulness Measurable Model** ([Madsen et al., 2024](https://arxiv.org/abs/2310.01538)): zeroing `s_j` entries causes no distribution shift, unlike input-space token erasure.

### Training

Three circuit losses optimized via GECO constrained optimization:

```
minimize     L_completeness + L_separation + L_sharpness
subject to   L_CE <= tau_ce
```

| Loss | Objective | Mechanism |
|------|-----------|-----------|
| **Completeness** | Circuit-masked predictions match full | DLA &rarr; soft top-k mask &rarr; reclassify &rarr; CE |
| **Separation** | Per-class circuits use distinct tokens | EMA centroids &rarr; mean pairwise cosine (orthogonality) |
| **Sharpness** | Attributions concentrate on few dims | Hoyer sparsity of attribution magnitudes |

The constraint threshold `tau_ce` is set automatically from the 25th percentile of warmup CE. A single GECO Lagrangian multiplier ([Rezende & Viola, 2018](https://arxiv.org/abs/1810.00597)) replaces manual loss weighting.

**Key invariant**: `compute_attribution_tensor()` is a single function called by both training losses and evaluation metrics. There is no separate training-time vs. evaluation-time attribution.

### Surgical Intervention

Two mechanisms for verifiable concept removal at the sparse bottleneck:

| Mechanism | Scope | Reversible | Guarantee |
|-----------|-------|------------|-----------|
| **Global suppression** (`suppress_token_globally`) | Zeros vocab_projector weights + DReLU threshold | No | `s[token_id] = 0` for all inputs, mathematically provable |
| **Inference-time masking** (`SuppressedCISModel`) | Masks sparse vector before classifier | Yes | DLA identity preserved with fewer active dims |

Since `logit[c] = sum_j s[j] * W_eff[c,j] + b_eff[c]`, zeroing `s[j]` removes token j's contribution to ALL classes with mathematical certainty.

---

## Installation

```bash
git clone https://github.com/<repo>/interpretable-splade-classifier.git
cd interpretable-splade-classifier
pip install -e .
```

**Requirements**: Python >= 3.10, PyTorch >= 2.1, CUDA GPU.

---

## Reproducing Results

### Core Experiments

```bash
# Main experiment (SST-2, full split)
python -m splade.scripts.run_experiment --config experiments/paper/sst2.yaml

# Ablation: Baseline (no circuit losses) vs Full Lexical-SAE
python -m splade.scripts.run_ablation --config experiments/paper/sst2_ablation.yaml

# Multi-dataset benchmark (SST-2, AG News, IMDB)
python -m splade.scripts.run_multi_dataset --config experiments/paper/multi_dataset.yaml
```

### Hero Experiments

```bash
# Experiment A: Faithfulness Stress Test
# Compares DLA removability vs gradient/IG/attention baselines
python -m splade.scripts.run_faithfulness --config experiments/civilcomments.yaml

# Experiment B: Surgical Bias Removal ("Lobotomy")
# Suppresses identity-correlated tokens, measures FPR gap reduction
python -m splade.scripts.run_surgery --config experiments/surgery.yaml

# Experiment C: Long-Context Needle in Haystack
# Tests sparse bottleneck signal preservation at increasing document lengths
python -m splade.scripts.run_long_context --config experiments/long_context.yaml
```

### Quick Verification

```bash
python -m splade.scripts.run_experiment --config experiments/verify_full.yaml
```

---

## Datasets

All loaded automatically from HuggingFace `datasets`:

| Dataset | Task | Classes | Train | Test |
|---------|------|---------|-------|------|
| SST-2 | Sentiment | 2 | 67,349 | 872 |
| AG News | Topic | 4 | 120,000 | 7,600 |
| IMDB | Sentiment | 2 | 25,000 | 25,000 |
| Yelp | Polarity | 2 | 560,000 | 38,000 |
| CivilComments | Toxicity | 2 | 1,804,874 | 97,320 |

CivilComments includes 24 identity group annotations for bias analysis (male, female, transgender, muslim, christian, jewish, black, white, etc.).

---

## Project Structure

```
splade/
  models/
    splade.py              # CISModel: BERT/ModernBERT + sparse bottleneck + ReLU MLP -> CircuitState
    layers/activation.py   # DReLU with learnable thresholds
  circuits/
    core.py                # CircuitState NamedTuple, circuit_mask()
    geco.py                # GECOController: dataset-size-invariant Lagrangian optimization
    losses.py              # Completeness, separation (orthogonality), sharpness
    metrics.py             # Circuit extraction, completeness, cosine/Jaccard separation
  mechanistic/
    attribution.py         # compute_attribution_tensor(): canonical DLA for all consumers
    layerwise.py           # Per-BERT-layer contribution decomposition
    sae.py                 # Sparse autoencoder baseline
  evaluation/
    eraser.py              # ERASER comprehensiveness, sufficiency, AOPC
    faithfulness.py        # Removability metric: prediction flip rate under DLA-guided ablation
    baselines.py           # Gradient, IG, attention attribution methods
    compare_explainers.py  # Side-by-side ERASER comparison across explainers
    mechanistic.py         # Tiered evaluation pipeline (Performance -> Faithfulness -> Interpretability)
  intervene.py             # Surgical intervention: suppress_token_globally, SuppressedCISModel, evaluate_bias
  training/
    loop.py                # GECO-integrated training with EMA, early stopping, lambda pinning alerts
    optim.py               # LR range test, gradient centralization
    constants.py           # Internal hyperparameters (not user-configurable)
  data/loader.py           # HuggingFace dataset loading + CivilComments identity annotations
  inference.py             # Batched inference, prediction, explanation API
  pipelines.py             # Shared setup_and_train() pipeline
  config/
    schema.py              # Dataclass config: 3 training knobs (target_accuracy, sparsity_target, warmup_fraction)
    load.py                # YAML -> Config
  scripts/
    run_experiment.py      # Train + mechanistic evaluation
    run_ablation.py        # Baseline vs Full Lexical-SAE comparison
    run_multi_dataset.py   # Cross-dataset benchmark
    run_faithfulness.py    # Experiment A: removability comparison
    run_surgery.py         # Experiment B: surgical bias removal
    run_long_context.py    # Experiment C: needle in haystack
experiments/
  paper/                   # Publication configs (full splits)
  civilcomments.yaml       # CivilComments experiment config
  surgery.yaml             # Bias removal experiment config
  long_context.yaml        # Long-context experiment config
  verify*.yaml             # Quick verification configs (toy-sized)
tests/                     # Unit tests
archive/                   # B-cos variant (not part of core contribution)
```

---

## Configuration

Three training knobs exposed via YAML `training:` section:

| Knob | Default | Effect |
|------|---------|--------|
| `target_accuracy` | `null` (auto) | GECO tau_ce override. Null = auto from warmup 25th percentile |
| `sparsity_target` | `0.1` | Circuit fraction: top 10% of active sparse dims |
| `warmup_fraction` | `0.2` | Fraction of training for CE-only warmup before circuit losses activate |

All other hyperparameters (LR schedule, EMA decay, GECO dynamics, etc.) are derived automatically or hardwired.

---

## Design Decisions

**Feature dictionary = tokenizer vocabulary.** Unlike post-hoc SAEs which learn an opaque feature dictionary requiring interpretation, Lexical-SAE operates directly in vocabulary space. Each sparse dimension corresponds to a known token. This makes surgical intervention auditable: if the model relies on an identity term like "Muslim" to predict toxicity, you can identify that spurious correlation by name and remove it with a single operation --- no guessing what "feature #1405" represents.

**Zero reconstruction error.** The DLA identity `logit_c = sum_j s_j * W_eff[c,j] + b_eff_c` holds exactly (verified to BF16 machine precision). Post-hoc SAEs incur reconstruction error that compounds through downstream layers.

**Intrinsic alignment.** Circuit structure (completeness, separation, sharpness) is optimized during training, not extracted post-hoc. The same `compute_attribution_tensor()` function drives both training losses and evaluation metrics.

**GECO over fixed loss weights.** A single Lagrangian multiplier adapts automatically. Lambda pinning detection alerts when the accuracy-interpretability trade-off becomes infeasible.

**Sparse bottleneck as FMM.** ~100-200 non-zero dimensions out of ~30K vocabulary. Zeroing entries for faithfulness evaluation or surgical intervention causes no distribution shift.

---

## Limitations

- **Text classification only.** The architecture requires a sparse vocabulary bottleneck; not applicable to generation or retrieval.
- **Vocabulary-level granularity.** Attributions identify vocabulary tokens, not input spans. A clean vocab filter masks subwords (`##ing`) and special tokens for human-readable output.
- **Input-dependent W_eff.** The effective weight matrix varies per input due to the ReLU activation mask; explanations are per-sample, not global.
- **Encoder family support.** Tested with DistilBERT, BERT-base, and ModernBERT. Other encoder families may require MLM head path adaptation.

---

## References

### Core Method

- Formal, T., Piwowarski, B., & Clinchant, S. (2021). SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking. *SIGIR*. [`arXiv:2107.05720`](https://arxiv.org/abs/2107.05720)
- Balestriero, R. & Baraniuk, R. (2018). A Spline Theory of Deep Networks. *ICML*. [`arXiv:1802.09210`](https://arxiv.org/abs/1802.09210)
- Rezende, D. J. & Viola, F. (2018). Taming VAEs. [`arXiv:1810.00597`](https://arxiv.org/abs/1810.00597)

### Evaluation

- DeYoung, J., et al. (2020). ERASER: A Benchmark to Evaluate Rationalized NLP Models. *ACL*. [`arXiv:1911.03429`](https://arxiv.org/abs/1911.03429)
- Madsen, A., et al. (2024). Are Faithfulness Measures Faithful? [`arXiv:2310.01538`](https://arxiv.org/abs/2310.01538)
- Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic Attribution for Deep Networks. *ICML*. [`arXiv:1703.01365`](https://arxiv.org/abs/1703.01365)

### Related Work

- Elhage, N., et al. (2021). A Mathematical Framework for Transformer Circuits. *Anthropic*.
- Conmy, A., et al. (2023). Towards Automated Circuit Discovery for Mechanistic Interpretability. *NeurIPS*. [`arXiv:2304.14997`](https://arxiv.org/abs/2304.14997)
- Marks, S., et al. (2024). Sparse Feature Circuits. *ICLR*. [`arXiv:2403.19647`](https://arxiv.org/abs/2403.19647)
- Lei, T., et al. (2025). Sparse Attention Post-Training. [`arXiv:2512.05865`](https://arxiv.org/abs/2512.05865)
- Lieberum, T., et al. (2025). Open Problems in Mechanistic Interpretability. [`arXiv:2501.16496`](https://arxiv.org/abs/2501.16496)
- Chen, J., et al. (2025). Rethinking Circuit Completeness. [`arXiv:2505.10039`](https://arxiv.org/abs/2505.10039)
