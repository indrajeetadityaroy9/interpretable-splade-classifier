# Circuit-Integrated SPLADE: Faithful-by-Construction Text Classification with Mechanistic Training Losses

Official implementation for *Circuit-Integrated SPLADE (CIS): Faithful-by-Construction Text Classification via Sparse Lexical Bottlenecks and Mechanistic Training Losses*.

---

## Abstract

Post-hoc explanation methods for text classifiers provide approximations with no formal guarantee that explanations reflect the model's internal reasoning. We propose **Circuit-Integrated SPLADE (CIS)**, a framework that repurposes the SPLADE sparse lexical bottleneck (Formal et al., 2021) as a classification architecture where every class logit decomposes *exactly* as a weighted sum over vocabulary-level activations. The key enabling mechanism is a **ReLU MLP classifier** that preserves exact Direct Logit Attribution (DLA) through an input-dependent effective weight matrix:

```
W_eff(s) = W2 @ diag(D(s)) @ W1        (piecewise-linear decomposition)

logit_c = Σ_j s_j · W_eff(s)[c,j] + b_eff(s)_c    (exact algebraic identity)
```

where `D(s)` is the ReLU activation mask induced by input `s`. Unlike prior work that uses DLA as an approximate component-level decomposition in transformers (Elhage et al., 2021), our DLA is *exact by construction* — the sparse bottleneck followed by a piecewise-linear classifier makes the decomposition an algebraic identity, verified to machine precision on every evaluation sample.

CIS unifies six novel mechanisms into a single hardwired framework:

1. **Exact DLA through ReLU W_eff** — zero-cost, zero-approximation explanations as weighted word lists
2. **Circuit-aware training losses** — completeness, separation, and sharpness as differentiable objectives
3. **DF-FLOPS regularization** — document-frequency-weighted sparsity preserving rare discriminative terms
4. **Vocabulary circuits** — per-class human-readable token subsets with high attribution mass
5. **Gini sharpness as training loss** — first use of the Gini coefficient as a differentiable attribution objective
6. **Mechanistic evaluation as training signal** — replacing post-hoc discover-then-report with joint optimization

All mechanisms are always active during training. The unified loss is:

```
L = L_CE + λ(t)·L_DF-FLOPS + w(t)·(0.1·L_completeness + 0.05·L_separation + 0.01·L_sharpness)
```

We evaluate DLA against seven post-hoc baselines on 12 faithfulness metrics from the ERASER, F-Fidelity, NAOPC, and soft perturbation literatures, across four text classification datasets and three encoder scales.

---

## Method

### Architecture

Input text is encoded by a BERT-family encoder, projected through a vocabulary-dimension MLP with GELU activation and layer normalization, sparsified by DReLU with learnable thresholds, log-saturated, max-pooled over the sequence, and classified by a two-layer ReLU MLP:

```
Encoder → GELU → LayerNorm → Vocab Projection → DReLU → log(1+·) → MaxPool → [ReLU MLP] → logits

s_j = max_i log(1 + max(0, w_{i,j} - θ_j))          (sparse bottleneck)

h = ReLU(W1 · s + b1)                                 (hidden layer, dim=256)
logit_c = W2_c · h + b2_c                             (classification head)
```

The ReLU activation creates a binary mask `D(s) = diag(1[W1·s + b1 > 0])` that is input-dependent but piecewise-constant, yielding an effective weight matrix `W_eff(s) = W2 @ D(s) @ W1` such that:

```
logit_c = Σ_j s_j · W_eff(s)[c,j] + b_eff(s)_c
```

This identity holds exactly — DLA provides zero-cost, zero-approximation explanations without gradient computation, sampling, or surrogate fitting. The piecewise-linear decomposition through ReLU networks is grounded in the theoretical framework of Balestriero & Baraniuk (2018).

### Training

All training hyperparameters are hardwired constants (not configurable via YAML). The CIS loss is always active:

```
L = L_CE + λ(t) · L_DF-FLOPS + w(t) · (0.1 · L_completeness + 0.05 · L_separation + 0.01 · L_sharpness)
```

**Circuit completeness loss.** For each sample, compute DLA for the target class (`attr_j = s_j · W_eff[c,j]`), build a soft mask retaining only the top fraction of dimensions via a differentiable sigmoid with temperature `T`: `mask_j = σ(T · (|attr_j| - threshold))`, forward the masked sparse vector `s ⊙ mask` through the classifier, and compute cross-entropy. This teaches the model to concentrate decision-relevant information in compact circuits.

**Circuit separation loss.** Maintain EMA centroids of per-class mean absolute attribution vectors. The loss is the mean pairwise cosine similarity between class centroids — minimizing it encourages each class to rely on a distinct subset of vocabulary dimensions.

**Attribution sharpness loss.** Compute the Gini coefficient of the per-sample attribution magnitude distribution. The loss is `1 − Gini`, so minimizing it maximizes sharpness. The Gini coefficient has been used to evaluate explanation sparsity (Blum et al., 2024) but never as a differentiable training objective.

| Component | Detail |
|---|---|
| Classifier | ReLU MLP: Linear(V, 256) → ReLU → Linear(256, C) |
| Optimizer | AdamW (fused), weight decay 0.01 |
| Learning rate | Auto-selected via LR range test (Smith, 2017) |
| LR schedule | Linear warmup (6%) + cosine annealing |
| Gradient clipping | Adaptive Gradient Clipping (AGC) with classifier exemption (Brock et al., 2021) |
| Label smoothing | 0.1 |
| Model averaging | EMA (decay 0.999), best checkpoint by validation loss |
| Early stopping | Patience 5 on validation loss |
| Precision | Mixed-precision (bfloat16 autocast) |
| Circuit loss warmup | Quadratic schedule, delayed until 30% of total steps |

### Mechanistic Analysis

The mechanistic suite measures the interpretability properties that CIS training optimizes:

- **DLA verification**: confirms `Σ s_j · W_eff[c,j] + b_eff_c == logit_c` across the evaluation set (mean absolute error reported; verified to machine precision).
- **Vocabulary circuits**: per-class subsets of vocabulary tokens with consistently high attribution, extracted by thresholding mean absolute DLA mass.
- **Circuit completeness**: accuracy retention when all vocabulary dimensions outside the circuit are zero-ablated via activation patching.
- **Semantic fidelity**: within-class consistency (pairwise Jaccard of per-sample top-*k* token sets) and cross-class separation.
- **SAE baseline**: optional overcomplete sparse autoencoder trained on hidden states at the vocabulary projection layer for comparison against DLA.

### Faithfulness Evaluation

Twelve metrics from four evaluation traditions, evaluated per explainer:

| Metric | Protocol | Reference |
|---|---|---|
| Comprehensiveness | Replace top-*k* tokens with `[MASK]`, measure confidence drop | DeYoung et al. (2020) |
| Sufficiency | Keep only top-*k* tokens, mask the rest | DeYoung et al. (2020) |
| Monotonicity | Fraction of sequential removal steps with decreasing confidence | Arya et al. (2019) |
| AOPC | Area over the perturbation curve across *k* values | Samek et al. (2017) |
| NAOPC | Beam-search normalized AOPC with upper/lower bounds | Chrysostomou & Aletras (2024) |
| Filler comprehensiveness | Replace top-*k* with OOD unigrams (frequency-based sampling) | Harbecke & Alt (2020) |
| Soft comprehensiveness | Bernoulli perturbation on the embedding layer | Chrysostomou & Aletras (2023) |
| Soft sufficiency | Bernoulli sufficiency on the embedding layer | Chrysostomou & Aletras (2023) |
| F-Fidelity+ | Surrogate accuracy drop, masking explanation tokens | Agarwal et al. (2024) |
| F-Fidelity− | Surrogate accuracy drop, masking non-explanation tokens | Agarwal et al. (2024) |
| Adversarial sensitivity | Kendall τ stability under character-level perturbations | Alvarez-Melis & Jaakkola (2018) |
| Causal faithfulness | Spearman ρ between DLA scores and MLM counterfactual shifts | Feder et al. (2021) |

### Baseline Explainers

| Explainer | Method | Source |
|---|---|---|
| SPLADE (DLA) | Exact logit decomposition via W_eff | Ours |
| LIME | Local linear surrogate on perturbed text | Ribeiro et al. (2016) |
| Integrated Gradients | Path integral of gradients on the embedding layer | Sundararajan et al. (2017) |
| GradientSHAP | Shapley-value approximation via gradient sampling | Lundberg & Lee (2017) |
| Attention | Mean last-layer attention to CLS token | — |
| Saliency | Gradient magnitude on the embedding layer | Simonyan et al. (2014) |
| DeepLIFT | Reference-based attribution on embeddings | Shrikumar et al. (2017) |
| Random | Uniform random scores (lower-bound baseline) | — |

---

## Getting Started

### Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.1
- CUDA-capable GPU

### Installation

```bash
pip install -e .
```

### Quick Start

Run the full CIS pipeline (train → evaluate → integration analysis):

```bash
python -m splade.scripts.run_experiment --config experiments/main/benchmark_cis.yaml
```

### Reproducing All Results

```bash
bash reproduce_results.sh
```

This runs all configurations (main benchmarks, cross-dataset, scaling, ablation, mechanistic) and writes results to `results/`.

---

## Configuration

Experiments are YAML-driven. All CIS training hyperparameters (DF-FLOPS weights, circuit loss lambdas, warmup schedules, classifier hidden size) are hardwired constants — not configurable via YAML. The configuration specifies only the experimental setup:

```yaml
experiment_name: "sst2_distilbert_cis"
output_dir: "results/main/cis"

data:
  dataset_name: "sst2"            # sst2 | ag_news | imdb | yelp
  train_samples: 2000
  test_samples: 200

model:
  name: "distilbert-base-uncased"  # any BERT-family HuggingFace identifier

evaluation:
  seeds: [42]
  explainers: ["splade", "lime", "ig", "gradient_shap", "attention", "saliency", "deeplift", "random"]

mechanistic:
  circuit_threshold: 0.01         # fraction of attribution mass for circuit extraction
  sae_comparison: false           # train SAE baseline for comparison
```

### Experiment Configurations

| Config | Description |
|---|---|
| `main/benchmark_cis.yaml` | Full CIS pipeline on SST-2 with all 8 explainers |
| `main/benchmark_df_flops.yaml` | DF-FLOPS baseline on SST-2 |
| `datasets/ag_news.yaml` | AG News (4-class topic classification) |
| `datasets/imdb.yaml` | IMDB (binary sentiment) |
| `datasets/yelp.yaml` | Yelp (binary polarity) |
| `scaling/bert_base.yaml` | BERT-base encoder |
| `scaling/bert_large.yaml` | BERT-large encoder |
| `ablation/cis_ablation.yaml` | Full CIS vs DF-FLOPS-only vs Baseline |
| `ablation/df_vs_vanilla.yaml` | DF-FLOPS vs Vanilla FLOPS |
| `mechanistic/base.yaml` | Mechanistic analysis only |
| `mechanistic/sae_comparison.yaml` | SAE comparison baseline |

---

## Pipeline

The experiment pipeline runs three phases per seed:

1. **CIS Training**: Data loading, model training with LR range test, EMA, early stopping, DF-FLOPS, and circuit losses. All CIS mechanisms are always active.
2. **Evaluation**: DLA verification, vocabulary circuit extraction, circuit completeness, semantic fidelity, optional SAE comparison, surrogate fine-tuning for F-Fidelity, and 12-metric benchmark across all configured explainers.
3. **Integration Analysis**: Jaccard overlap between explainer attributions and vocabulary circuits, cross-explainer Spearman correlations.

### Ablation Study

Ablation variants are created by temporarily zeroing CIS constants at runtime, not by config flags:

```bash
python -m splade.scripts.run_ablation --config experiments/ablation/cis_ablation.yaml
```

## Datasets

All datasets are loaded from HuggingFace `datasets`:

| Dataset | Task | Classes | Source |
|---|---|---|---|
| SST-2 | Sentence sentiment | 2 | `glue/sst2` |
| AG News | News topic classification | 4 | `ag_news` |
| IMDB | Movie review sentiment | 2 | `imdb` |
| Yelp | Review polarity | 2 | `yelp_polarity` |

---

## Dependencies

| Category | Packages |
|---|---|
| Core | `torch>=2.1.0`, `transformers>=4.30.0`, `datasets>=2.14.0` |
| Evaluation | `captum>=0.6.0`, `lime>=0.2.0`, `scipy>=1.11.0` |
| Utilities | `numpy>=1.24.0`, `scikit-learn>=1.3.0`, `pyyaml>=6.0`, `tqdm>=4.60.0`, `nltk>=3.8.0` |

---

## Limitations

- **Text classification only.** The architecture is not designed for retrieval, generation, or token-level tasks.
- **English, BERT-family encoders.** Tested with DistilBERT, BERT-base, and BERT-large. Other encoder families may require MLM head path adjustments.
- **Accuracy–interpretability trade-off.** The vocabulary bottleneck constrains model capacity relative to unconstrained dense classifiers; circuit-aware training adds further structural pressure.
- **Vocabulary-level granularity.** DLA identifies contributing vocabulary tokens, not input spans. Subword tokenization may reduce readability for multi-token words.
- **Input-dependent W_eff.** The effective weight matrix varies per input due to the ReLU activation mask, requiring per-sample computation for explanations (though this remains cheaper than gradient-based methods).

---

## References

**Sparse lexical models (architectural basis):**
- Formal, T., Piwowarski, B., & Clinchant, S. (2021). SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking. *SIGIR*. [`arXiv:2107.05720`](https://arxiv.org/abs/2107.05720)
- Formal, T., Lassance, C., Piwowarski, B., & Clinchant, S. (2021). SPLADE v2: Sparse Lexical and Expansion Model for Information Retrieval. [`arXiv:2109.10086`](https://arxiv.org/abs/2109.10086)
- Lassance, C., & Clinchant, S. (2024). SPLADE v3. [`arXiv:2403.06789`](https://arxiv.org/abs/2403.06789)
- Porco, L., Lassance, C., & Clinchant, S. (2025). An Alternative to FLOPS Regularization to Effectively Productionize SPLADE-Doc. *SIGIR*. [`arXiv:2505.15070`](https://arxiv.org/abs/2505.15070) — concurrent DF-FLOPS work for retrieval latency; our formulation targets classification interpretability.

**Piecewise-linear decomposition (W_eff theoretical basis):**
- Balestriero, R., & Baraniuk, R. (2018). A Spline Theory of Deep Networks. *ICML*. [`arXiv:1802.09210`](https://arxiv.org/abs/1802.09210) — establishes that ReLU networks are piecewise-linear mappings with per-region affine parameters; our W_eff is the per-input affine weight matrix.

**Interpretable bottleneck models (differentiated from):**
- Sun, Y., Li, Z., Wang, Y., & Chen, M. (2025). Concept Bottleneck Large Language Models. *ICLR*. [`arXiv:2412.07992`](https://arxiv.org/abs/2412.07992) — concept-level bottleneck; ours operates in vocabulary space without concept annotation.
- Yan, A.N., Gao, Y., Joshi, B., Strubell, E., & Iyyer, M. (2023). Text Bottleneck Models. [`arXiv:2310.19660`](https://arxiv.org/abs/2310.19660)

**Direct Logit Attribution and faithful-by-construction explanations:**
- Elhage, N., et al. (2021). A Mathematical Framework for Transformer Circuits. *Anthropic Transformer Circuits Thread*. — origin of DLA as component-level decomposition; our DLA is exact by architectural design.
- McDougall, C., Conmy, A., Rushing, C., McGrath, T., & Neel, N. (2024). Copy Suppression: Comprehensively Understanding an Attention Head. *BlackboxNLP*. [`arXiv:2310.07325`](https://arxiv.org/abs/2310.07325) — demonstrates DLA limitations in deep transformers due to erasure; our architecture avoids these by design.
- Bohle, M., Fritz, M., & Schiele, B. (2024). CoDA-Nets: Convolutional Dynamic Alignment Networks for Interpretable Classification. — faithful-by-construction for images via dynamic linear decomposition; ours targets text via sparse vocabulary bottleneck.

**Training for interpretability (CIS builds on):**
- Ross, A.S., Hughes, M.C., & Doshi-Velez, F. (2017). Right for the Right Reasons: Training Differentiable Models to Gradient Supervise. *IJCAI*. [`arXiv:1703.03717`](https://arxiv.org/abs/1703.03717) — penalizes input gradients; our losses operate at the circuit/attribution level.
- Liu, Z., Khona, M., Fiete, I.R., & Tegmark, M. (2023). Growing Brains: Co-emergence of Anatomical and Functional Modularity in RNNs. [`arXiv:2305.08746`](https://arxiv.org/abs/2305.08746) — BIMT adds topology-level loss for modularity; CIS adds circuit-level attribution losses.
- Lieberum, T., Rahtz, M., et al. (2025). Open Problems in Mechanistic Interpretability. [`arXiv:2501.16496`](https://arxiv.org/abs/2501.16496) — identifies training for interpretability as an open problem; CIS directly addresses this.

**Circuit discovery and analysis (differentiated from):**
- Conmy, A., Mavor-Parker, A.N., Lynch, A., Heimersheim, S., & Garriga-Alonso, A. (2023). Towards Automated Circuit Discovery for Mechanistic Interpretability. *NeurIPS*. [`arXiv:2304.14997`](https://arxiv.org/abs/2304.14997) — post-hoc discovery; CIS integrates circuit objectives into training.
- Marks, S., Rager, C., Michaud, E.J., Belinkov, Y., Bau, D., & Mueller, A. (2024). Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models. *ICLR*. [`arXiv:2403.19647`](https://arxiv.org/abs/2403.19647) — SAE-feature-level circuits; ours are vocabulary-level.
- Niu, Y., Huang, Q., & Liu, X. (2025). DiscoGP: End-to-end Differentiable Circuit Discovery. *EMNLP*. [`arXiv:2407.03779`](https://arxiv.org/abs/2407.03779) — differentiable masks for post-hoc discovery; our soft masks are training-time losses.
- Chen, J., He, Z., & Gonzalez, J.E. (2025). Rethinking Circuit Completeness. [`arXiv:2505.10039`](https://arxiv.org/abs/2505.10039) — analyzes completeness as measurement; we optimize it as a loss.

**Sparse autoencoders for interpretability (SAE baseline context):**
- Cunningham, H., Ewart, A., Riggs, L., Huben, R., & Sharkey, L. (2023). Sparse Autoencoders Find Highly Interpretable Directions in Language Models. [`arXiv:2309.08600`](https://arxiv.org/abs/2309.08600)
- Anthropic (2024). Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet. [`transformer-circuits.pub`](https://transformer-circuits.pub/2024/scaling-monosemanticity/)
- Gao, L., et al. (2025). Scaling and Evaluating Sparse Autoencoders. *ICLR*. [`arXiv:2406.04093`](https://arxiv.org/abs/2406.04093)
- Gallifant, J., et al. (2025). SAE Features for Classification. *EMNLP*. [`arXiv:2502.11367`](https://arxiv.org/abs/2502.11367) — uses SAE features from LLMs for classification; our architecture makes SAEs unnecessary.

**Attribution sharpness (Gini coefficient context):**
- Blum, L., Beckh, K., Jakobs, M., Hammoudeh, Z., & Schreurs, N. (2024). Sparse Explanations via Pruned Layer-Wise Relevance Propagation. [`arXiv:2404.14271`](https://arxiv.org/abs/2404.14271) — uses Gini for evaluation; we use it as a differentiable training loss.

**Faithfulness evaluation:**
- DeYoung, J., Jain, S., Rajani, N.F., et al. (2020). ERASER: A Benchmark of Rationality. *ACL*. [`arXiv:1911.03429`](https://arxiv.org/abs/1911.03429)
- Chrysostomou, G., & Aletras, N. (2023). Investigating the Faithfulness of Soft Perturbation-based Explanation Methods. [`arXiv:2305.10496`](https://arxiv.org/abs/2305.10496)
- Chrysostomou, G., & Aletras, N. (2024). Normalized AOPC: Fixing Perturbation-Based Faithfulness Metrics. *ACL*. [`arXiv:2408.08137`](https://arxiv.org/abs/2408.08137)
- Agarwal, C., Saxena, E., Krishna, S., et al. (2024). F-Fidelity: A Robust Framework for Faithfulness Evaluation of Explainable AI. *ICLR*. [`arXiv:2410.02970`](https://arxiv.org/abs/2410.02970)

**Training techniques:**
- Brock, A., De, S., Smith, S.L., & Simonyan, K. (2021). High-Performance Large-Scale Image Recognition Without Normalization. *ICML*. [`arXiv:2102.06171`](https://arxiv.org/abs/2102.06171) — Adaptive Gradient Clipping (AGC).

---

## License

See `LICENSE` for details.
