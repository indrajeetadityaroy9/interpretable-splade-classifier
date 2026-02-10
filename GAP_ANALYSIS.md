# Lexical Terminal Transcoder — Gap Analysis & Architecture Upgrade Plan

## Context

The Lexical-SAE architecture (now Lexical Terminal Transcoder) achieves exact DLA attributions via GatedJumpReLU binary gating over vocabulary-sized sparse representations. The README identifies 6 limitations. This plan analyzes each against SOTA literature (2022–2025), quantifies gaps, and proposes concrete modifications ranked by impact.

---

## 1. Literature Review

### Core References (Already Implemented)
- **Rajamanoharan et al. 2024** (arXiv:2407.14435) — JumpReLU SAEs: STE-based binary gating for L0 sparsity. *Already implemented as GatedJumpReLU.*
- **Gao et al. 2024** (arXiv:2406.04093) — Scaling SAEs: TopK activation, scaling laws (tokens ∝ n^0.6), 16M-latent feasibility, 4 evaluation metrics (downstream loss, probe loss, explainability, ablation sparsity).
- **Belrose et al. 2023** (arXiv:2306.03819) — LEACE concept erasure. *Already implemented.*

### New References for Gap Closure
| Paper | Key Contribution | Addresses Limitation |
|-------|-----------------|---------------------|
| **Gated Attention (arXiv:2505.06708, NeurIPS 2025 Best Paper)** | Query-dependent sigmoid gates on attention heads; context-dependent sparsity | L5: Single-sense vocabulary |
| **SAE Polysemy Evaluation (arXiv:2501.06254, 2025)** | Word-in-Context evaluation of SAE polysemy decomposition | L5: Single-sense vocabulary |
| **Unified Attribution (arXiv:2501.18887, 2025)** | Multi-granularity attribution framework (feature/data/component) | L1: Vocabulary-level granularity |
| **NVIDIA UST (2025)** | Universal Sparse Tensor, CSR-based GPU acceleration | L4: NER sparsity gap (memory) |
| **Lei et al. 2025** (arXiv:2512.05865) | Sparse attention post-training; cross-layer transcoders | L2: Input-dependent W_eff |
| **SAE Survey (arXiv:2503.05613, 2025)** | Comprehensive SAE taxonomy, polysemy → monosemantic decomposition | L5: Single-sense vocabulary |

---

## 2. Gap Analysis — Limitations Ranked by Impact

### Priority 1 (HIGH) — Directly impacts core contribution claims

#### L1: Vocabulary-Level Granularity
**Current state:** Attributions map to individual vocab tokens (subwords). `get_clean_vocab_mask()` in `intervene.py` filters continuations/special tokens for display, but the underlying representation is subword-level.
**Gap:** No span-level or word-level aggregation. Users see `["Ġun", "##happy"]` instead of `"unhappy"`. Competing methods (Integrated Gradients, SHAP) operate on input tokens directly.
**SOTA:** Unified Attribution framework (2501.18887) shows multi-granularity attribution is achievable by aggregating feature-level attributions to span/entity level.
**Impact:** HIGH — interpretability is the core selling point. Subword fragmentation undermines human-readable explanations.

#### L4: NER Sparsity Gap (~2K active dims vs ~100-200 for classification)
**Current state:** Per-position `[B, L, V]` representation in `sequence_loop.py` with `_SPARSITY_GAIN = 1.0`. Gate sparsity loss applies uniformly.
**Gap:** 20x more active dims than classification. The per-position nature is inherent, but sparsity could be improved with position-aware or adaptive thresholds.
**SOTA:** Gao et al. show TopK provides exact L0 control without proxy losses. Multi-TopK allows flexible per-token sparsity.
**Impact:** HIGH — undermines the "extreme sparsity" claim for NER. 2K/50K = 96% sparsity vs 99.7% for classification.

### Priority 2 (MEDIUM) — Affects generality and evaluation rigor

#### L3: Encoder Family Scope (Only ModernBERT + DistilBERT)
**Current state:** `AutoModelForMaskedLM` abstraction in `lexical_sae.py` with `_backbone_params` kwarg filtering. Architecture-agnostic in principle.
**Gap:** No benchmarks on RoBERTa, ALBERT, DeBERTa, or decoder models. Claims of generality are unsupported.
**SOTA:** Standard practice is 3+ backbone evaluations.
**Impact:** MEDIUM — reviewers will flag this. Fix is evaluation-only (no code changes needed, just configs).

#### L2: Input-Dependent W_eff
**Current state:** `classifier_forward()` in `lexical_sae.py:148-162` computes `W_eff` per-sample due to classifier ReLU mask. Shape `[B, C, V]`.
**Gap:** No global W_eff summary. Each explanation is per-input. Memory scales as O(B × C × V).
**SOTA:** Lei et al. (2512.05865) show cross-layer transcoders can simplify attribution. Sparse attention reduces effective computation.
**Impact:** MEDIUM — this is an inherent property of piecewise-linear networks. Can mitigate with (a) amortized W_eff statistics, (b) sparse W_eff storage.

### Priority 3 (LOW) — Acknowledged design decisions

#### L5: Single-Sense Vocabulary
**Current state:** `gate_threshold` shape `[vocab_size]` — one scalar per vocab dim. No context dependence.
**Gap:** "bank" (financial vs river) gets one threshold. SAE Polysemy paper (2501.06254) shows SAEs can decompose polysemy when features are hidden-state-dependent.
**SOTA:** Gated Attention (2505.06708) achieves context-dependent gating via query-dependent sigmoid. However, applying this to vocabulary gating would break the backbone-agnostic DLA identity — the gate would depend on hidden states, making `s[j]` a function of `h` rather than a function of MLM logits alone.
**Impact:** LOW — the README correctly identifies this as a fundamental architectural tradeoff. Context-dependent gating breaks DLA. This is a known limitation, not a gap.

#### L6: Log-Compression Removed
**Current state:** Raw MLM logits passed through GatedJumpReLU. `init_threshold=1.0` gates off small values.
**Gap:** Higher magnitude variance than log-compressed representations.
**SOTA:** No specific literature suggests log-compression is needed with binary gating. The gate already handles small-value suppression.
**Impact:** LOW — empirically working. Monitor for training instability on new backbones.

---

## 3. Proposed Modifications

### Mod A: Subword-to-Span Attribution Aggregation (addresses L1)

**What:** Add a post-hoc span aggregation layer that merges subword attributions into word/entity-level attributions using tokenizer offset mappings.

**Where:** New module `splade/attribution/span_aggregation.py`

**Interface:**
```python
def aggregate_subword_attributions(
    token_ids: list[int],
    attribution_scores: torch.Tensor,  # [V] or [L, V]
    tokenizer,
    text: str,
    method: str = "sum",  # "sum" | "max_abs"
) -> list[tuple[str, float]]:
    """Merge subword attributions into word-level scores.

    Methods:
      - "sum": total evidence (risks cancellation on signed scores)
      - "max_abs": peak salience (keeps sign of max-absolute subword)
    """
```

**How:** Use `tokenizer(text, return_offsets_mapping=True)` to get character spans. Group contiguous subwords belonging to the same word. Aggregate scores per method:
```python
if method == "sum":
    score = subword_scores.sum()
elif method == "max_abs":
    idx = subword_scores.abs().argmax()
    score = subword_scores[idx]  # preserves sign
```
Return `[(word, score), ...]`.

**Why both methods:** Sum captures total evidence but risks cancellation when subword scores have mixed signs (e.g., `+0.8 + -0.7 = 0.1` hides salience). Max-abs captures peak salience without cancellation.

**Files modified:**
- `splade/attribution/span_aggregation.py` — **NEW**
- `splade/inference.py` — add `explain_model_spans()` wrapper
- `splade/intervene.py` — update `get_top_tokens()` to optionally return span-level

**DLA preservation:** Yes — aggregation is post-hoc on already-computed DLA attributions. No model changes.

### Mod B: Adaptive Per-Position Sparsity for NER (addresses L4)

**What:** Improve NER sparsity via **inference-only TopK** enforcement. Training continues using GatedJumpReLU + gate sparsity loss (which handles sparsity via gradient). Hard TopK is applied only at inference/evaluation time.

**Why inference-only:** Hard TopK (`scatter_`) is non-differentiable — elements outside top-k get zero gradient, preventing the model from learning to swap features into the active set. The existing `gate_sparsity_loss` already drives training-time sparsity via differentiable L1 on sigmoid gate probabilities.

**Where:** `splade/models/lexical_sae.py`, `splade/config/schema.py`

**Interface:**
```python
# In LexicalSAE:
def _enforce_topk(self, sparse_seq: torch.Tensor, k: int) -> torch.Tensor:
    """Inference-only: keep top-k activations per position."""
    if k <= 0 or k >= self.vocab_size:
        return sparse_seq
    topk_vals, topk_idx = sparse_seq.topk(k, dim=-1)
    result = torch.zeros_like(sparse_seq)
    result.scatter_(-1, topk_idx, topk_vals)
    return result

def _compute_sparse_sequence(self, attention_mask, *, input_ids=None, inputs_embeds=None):
    ...
    sparse_seq = activated.masked_fill(~attention_mask.unsqueeze(-1).bool(), 0.0)
    # TopK only at inference (not training) — preserves gradient flow
    if not self.training and self._topk_per_position > 0:
        sparse_seq = self._enforce_topk(sparse_seq, self._topk_per_position)
    return sparse_seq, gate_probs
```

**Config:** Add `topk_per_position: int = 0` (0 = disabled, e.g. 200 for NER inference).

**DLA preservation:** Yes — TopK is a binary mask (keep/zero), piecewise-linear. Applied post-gating.

**Files modified:**
- `splade/models/lexical_sae.py` — add `_enforce_topk()`, call in `_compute_sparse_sequence()` only when `not self.training`
- `splade/config/schema.py` — add `topk_per_position` field

### Mod C: Multi-Backbone Evaluation Configs (addresses L3)

**What:** Add experiment configs for RoBERTa-base and DeBERTa-v3-base. No code changes — only YAML configs.

**Where:** `experiments/` directory

**Files:**
- `experiments/sst2_roberta.yaml` — **NEW**
- `experiments/sst2_deberta.yaml` — **NEW**
- `experiments/conll_roberta.yaml` — **NEW**

### Mod D: Sparse W_eff Storage and Amortized Statistics (addresses L2)

**What:** Use **"Active Slice"** logic to evaluate only active vocabulary dimensions in W_eff, avoiding complex sparse tensor abstractions. Compute union of active indices per batch, slice W_eff to `[B, C, K_active]` where K_active ≈ 500.

**Why not sparse tensors:** PyTorch sparse tensors are finicky with broadcasting and matmul. Active-slice gives 99% of the speedup with 1% of the complexity.

**Where:** `splade/evaluation/mechanistic.py`

**Interface:**
```python
def compute_active_slice_weff(sparse_vector, W_eff):
    """Slice W_eff to only active vocabulary dimensions.

    Args:
        sparse_vector: [B, V] — pooled sparse representation
        W_eff: [B, C, V] — full effective weight matrix

    Returns:
        W_active: [B, C, K] — sliced to active dims only
        active_ids: [K] — indices into vocab
    """
    active_mask = sparse_vector.abs().sum(0) > 0  # [V] union across batch
    active_ids = active_mask.nonzero(as_tuple=True)[0]  # [K]
    W_active = W_eff[:, :, active_ids]  # [B, C, K]
    return W_active, active_ids
```

**Files modified:**
- `splade/evaluation/mechanistic.py` — add `compute_active_slice_weff()`, use in evaluation loops

### Mod E: Evaluation Metrics from Gao et al. (addresses eval rigor)

**What:** Add downstream loss and probe loss evaluation (from arXiv:2406.04093) to complement existing ERASER metrics.

**Where:** `splade/evaluation/scaling_metrics.py` — **NEW**

**Metrics:**
1. **Downstream loss:** Substitute sparse reconstruction into backbone, measure LM loss delta
2. **Ablation sparsity:** (L1/L2)² of attribution vectors — measures how concentrated attributions are

**Files modified:**
- `splade/evaluation/scaling_metrics.py` — **NEW**
- `splade/scripts/run_experiment.py` — add optional scaling metrics evaluation

---

## 4. Updated System Architecture

```
Input text
    │
    ▼
┌─────────────────────┐
│  HuggingFace MLM    │  AutoModelForMaskedLM (ModernBERT / RoBERTa / DeBERTa)
│  Backbone           │
└────────┬────────────┘
         │ MLM logits [B, L, V]
         ▼
┌─────────────────────┐
│  GatedJumpReLU      │  Binary gate (Heaviside) + sigmoid STE
│  + Optional TopK    │  Per-position TopK cap for NER sparsity control
└────────┬────────────┘
         │ sparse_seq [B, L, V], gate_probs [B, L, V]
         ▼
┌─────────────────────────────────────────────┐
│  Task Head Router                            │
│  ┌──────────────┐    ┌───────────────────┐  │
│  │ classify()   │    │ tag()             │  │
│  │ max-pool→MLP │    │ per-position MLP  │  │
│  └──────┬───────┘    └───────┬───────────┘  │
│         │                    │               │
│    CircuitState         TagCircuitState      │
│  (logits, s, W_eff,   (logits, s_seq,       │
│   b_eff)               W_eff_seq, b_eff)    │
└─────────┬───────────────────┬───────────────┘
          │                   │
          ▼                   ▼
┌─────────────────────────────────────────────┐
│  Attribution Layer                           │
│  ┌────────────┐  ┌────────────────────────┐ │
│  │ DLA        │  │ Span Aggregation (NEW) │ │
│  │ s[j]*W[c,j]│  │ subword → word-level   │ │
│  └────────────┘  └────────────────────────┘ │
└─────────┬───────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────┐
│  Intervention / Evaluation                   │
│  ┌──────────────┐ ┌────────┐ ┌───────────┐ │
│  │ Surgical     │ │ LEACE  │ │ ERASER +  │ │
│  │ Suppression  │ │ Erasure│ │ Scaling   │ │
│  │ (gate→1e6)   │ │        │ │ Metrics   │ │
│  └──────────────┘ └────────┘ └───────────┘ │
└─────────────────────────────────────────────┘
```

### Module Responsibilities

| Module | Responsibility | Interface |
|--------|---------------|-----------|
| `models/lexical_sae.py` | Backbone → sparse → classifier | `forward() → (sparse_seq, gate_probs)` |
| `models/layers/activation.py` | GatedJumpReLU gating | `forward(x) → (out, gate_sigmoid)` |
| `attribution/span_aggregation.py` | **NEW** Subword→word merging | `aggregate(token_ids, scores, tokenizer, text) → [(word, score)]` |
| `circuits/losses.py` | Gate sparsity, completeness, separation | `compute_gate_sparsity_loss(gate_probs) → scalar` |
| `circuits/core.py` | CircuitState + sparse W_eff | `CircuitState(logits, s, W_eff, b_eff)` |
| `evaluation/leace.py` | LEACE concept erasure baseline | `fit_leace_eraser() → eraser` |
| `evaluation/scaling_metrics.py` | **NEW** Downstream loss, ablation sparsity | `compute_downstream_loss(), compute_ablation_sparsity()` |
| `intervene.py` | Surgical suppression, bias eval | `SuppressedModel`, `evaluate_bias()` |
| `training/loop.py` | Classification training with GECO | `train_classification()` |
| `training/sequence_loop.py` | NER training with GECO | `train_sequence()` |

### Data Flow Summary

1. **Forward:** `input_ids → backbone → MLM logits → GatedJumpReLU → [optional TopK] → sparse_seq [B,L,V]`
2. **Classification:** `sparse_seq → max_pool → MLP(ReLU) → logits [B,C]`
3. **NER:** `sparse_seq → per-position MLP(ReLU) → tag_logits [B,L,T]`
4. **Attribution:** `DLA: logit[c] = Σ_j s[j] · W_eff[c,j] + b_eff[c]` → optional span aggregation
5. **Intervention:** Set `gate_threshold[j] = 1e6` (surgical) or apply LEACE projection (erasure)
6. **Loss:** `GECO(CE) + λ_cc·completeness + λ_sep·separation + λ_gate·gate_sparsity`

---

## 5. Implementation Priority & Verification

| Priority | Modification | Effort | Impact |
|----------|-------------|--------|--------|
| 1 | **Mod A:** Span aggregation | Small (1 new file + 2 edits) | HIGH — fixes L1 |
| 2 | **Mod B:** Adaptive TopK for NER | Medium (model + config + loop) | HIGH — fixes L4 |
| 3 | **Mod C:** Multi-backbone configs | Trivial (YAML only) | MEDIUM — fixes L3 |
| 4 | **Mod E:** Scaling evaluation metrics | Small (1 new file) | MEDIUM — eval rigor |
| 5 | **Mod D:** Sparse W_eff | Medium (core datastructure) | LOW — optimization |

### Verification Plan

1. **Mod A:** Run `explain_model_spans()` on SST-2 sample, verify words not subwords in output
2. **Mod B:** Train NER normally, evaluate with `topk_per_position=200` (inference-only), verify sparsity improves from 96% to >99%, F1 within 1% of baseline
3. **Mod C:** Run `run_experiment.py` with RoBERTa and DeBERTa configs, report accuracy
4. **Mod D:** Compare memory usage of sparse vs dense CircuitState on batch of 64
5. **Mod E:** Run scaling metrics on trained SST-2 model, report downstream loss and ablation sparsity
6. **DLA invariant:** For all mods, verify `|logit - Σ s·W_eff - b_eff| < 1e-3` on test samples
