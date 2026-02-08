Grounding the Unified Design: What the Literature Actually Says                        
                                                                                                                                                                                    
  The Core Tension the Papers Reveal                                                                                                                                                

  The literature reveals a clear taxonomy of how interpretability relates to training, and CIS occupies a position that its own codebase contradicts:
  Paradigm: Post-hoc analysis
  Relationship: Train → freeze → explain
  Examples: ACDC, Sparse Feature Circuits, LIME, SHAP
  ────────────────────────────────────────
  Paradigm: Post-training regularization
  Relationship: Train → fine-tune with interpretability constraint
  Examples: Sparse Attention Post-Training (https://arxiv.org/abs/2512.05865)
  ────────────────────────────────────────
  Paradigm: Training-time structural bias
  Relationship: Add topology/modularity loss during training
  Examples: BIMT (https://arxiv.org/abs/2401.03646, https://arxiv.org/abs/2305.08746)
  ────────────────────────────────────────
  Paradigm: Explanation-guided training
  Relationship: Use explanation method output as training signal
  Examples: REGEX (https://arxiv.org/abs/2312.17591), Right for Right Reasons (https://arxiv.org/abs/1703.03717)
  ────────────────────────────────────────
  Paradigm: Faithful-by-construction
  Relationship: Architecture makes forward pass = explanation
  Examples: CoDA-Nets (https://arxiv.org/abs/2104.00032), CIS (this project)
  CIS claims to be in category 5 — the most unified position — where the explanation is not a separate computation but an algebraic identity of the forward pass itself. The README
  states this explicitly: "DLA provides zero-cost, zero-approximation explanations without gradient computation, sampling, or surrogate fitting."

  But the codebase implements category 3 (structural bias) with a category 1 evaluation suite bolted on. The training/ directory adds circuit-shaped losses. The mechanistic/
  directory runs post-hoc analysis. They share no code.

  What Each Paper Teaches About Why This Split Is Wrong

  ---
  1. CoDA-Nets (https://arxiv.org/abs/2104.00032) — The architectural template CIS should follow

  CoDA-Nets prove that when the forward pass IS the explanation, there is no separate "analysis" module. Their core equation:

  y(x) = W_(0→L)(x) · x

  The effective weight matrix W_(0→L) is computed during the forward pass, not extracted after it. The contribution map s_j(x_i) = [W_(0→L)(x_i)]_j ⊙ x_i is an intermediate value
  of the forward computation, not a post-hoc decomposition.

  What this means for CIS: compute_effective_weights() should not be a separate method called by both training/circuit_losses.py and mechanistic/attribution.py. It should be part
  of the forward pass, with W_eff returned alongside logits and sparse_vector as a first-class output. Currently splade.py:64-68 computes classifier_forward() without exposing
  W_eff, then splade.py:70-102 recomputes it from scratch. The forward pass already has all the information — it just throws it away.

  ---
  2. DiscoGP (https://arxiv.org/abs/2407.03779) — Differentiable masking unifies discovery and training

  DiscoGP's key insight is that circuit discovery and circuit optimization use the same primitive: differentiable binary masks. Their loss:

  ℒ_GP = ℒ_fidelity + λ_c·ℒ_complete + λ_s·ℒ_sparse

  This is structurally identical to what CIS does (L_CE + circuit losses), but DiscoGP doesn't have separate "discovery" and "training" modules — the mask parameters ARE the
  circuit, simultaneously optimized for faithfulness and sparsity. The masks are the circuit definition, the training signal, and the explanation, all in one object.

  What this means for CIS: The soft mask in circuit_losses.py:149 (soft_mask = sigmoid(T * (|attr| - threshold))) and the hard circuit extraction in circuits.py:71 (circuit_mask =
  normalized >= threshold) are the same operation at different temperatures. They should be one object — a CircuitMask that can operate in differentiable mode (training) or
  discrete mode (evaluation) via a temperature parameter, exactly like DiscoGP's Gumbel-sigmoid with straight-through estimator.

  ---
  3. Rethinking Circuit Completeness (https://arxiv.org/abs/2505.10039) — The completeness metric must match the completeness loss

  This paper formalizes why CIS's split metrics are a problem. Completeness is defined as:

  D(G\C || G)  — divergence when circuit is ablated from full model

  But CIS measures completeness differently during training vs evaluation:
  - Training (circuit_losses.py:112-155): soft sigmoid masking, top 10% of vocab dimensions by attribution magnitude, cross-entropy loss on masked logits
  - Evaluation (circuits.py:89-117): hard zero-ablation of non-circuit tokens, circuit defined by 1% attribution mass threshold, accuracy retention metric

  These are measuring different properties of different subsets. The paper's AND/OR/ADDER gate framework shows that incomplete circuits are non-deterministic across runs precisely
  because of this kind of inconsistency — the training loss optimizes one circuit boundary while the evaluation metric measures another.

  What this means for CIS: The circuit_threshold (0.01) used in evaluation and the CIRCUIT_FRACTION (0.1) used in training must derive from the same parameterization. A unified
  CircuitExtractor should define the circuit once and expose it for both gradient-based optimization and inference-time measurement.

  ---
  4. BIMT (https://arxiv.org/abs/2401.03646) — Training-time modularity IS the circuit

  BIMT adds a connection-cost penalty during training that causes neurons to self-organize into modular clusters. The key finding: "circuits that automated discovery are trying to
  find might already be segregated by modular training." Post-hoc circuit discovery becomes trivial because the training loss already created the circuits.

  CIS claims the same thing — circuit losses during training should make post-hoc circuit discovery unnecessary. But the codebase contradicts this by having
  mechanistic/circuits.py:21-86 perform circuit extraction from scratch, re-running forward passes and re-computing attributions, completely ignoring the AttributionCentroidTracker
   that accumulated exactly this information during training.

  What this means for CIS: The AttributionCentroidTracker from training should persist as the circuit definition. Post-training "extraction" should just query the tracker's final
  state, not rebuild circuits from scratch.

  ---
  5. Sparse Attention Post-Training (https://arxiv.org/abs/2512.05865) — Constrained loss preserves performance while imposing structure

  Their formulation is clean:

  minimize Σ_l E[|A_l|]  subject to  CE ≤ τ

  Solved via Lagrangian relaxation (GECO). This automatically balances interpretability pressure against performance — when cross-entropy drops below threshold, sparsity
  regularization increases.

  CIS uses fixed λ schedules instead (CIRCUIT_COMPLETENESS_LAMBDA = 0.1, etc.). This is fragile — the right balance depends on the dataset and model, but the constants are
  hardwired. A constrained formulation would let the circuit losses self-regulate.

  ---
  6. Open Problems in Mechanistic Interpretability (https://arxiv.org/abs/2501.16496) — The field's own diagnosis

  The survey identifies training-for-interpretability as an open problem and notes the critical gap: "the current strategy of training a model solely for performance and then
  interpreting it post hoc may not be optimal if our goal is a model that is both interpretable and performant."

  They also warn that attempts to incentivize interpretable activations "have not been competitively performant, and have also allowed superposition to sneak through." CIS's
  DF-FLOPS regularization addresses superposition via sparsity, but the fragmented architecture means the circuit losses can't verify during training whether superposition is
  actually being prevented — that check happens only in the mechanistic/ evaluation phase.

  ---
  7. REGEX (https://arxiv.org/abs/2312.17591) — Explanation-guided training as self-supervision

  REGEX shows that using explanation outputs (Integrated Gradients) as a training signal improves faithfulness:

  ℒ_kl = D_KL(att(x); IG(x)) + D_KL(att(x̃); IG(x))

  This is conceptually what CIS does with circuit losses — using DLA attributions as training signals. But REGEX uses the same IG function for both the training signal and the
  evaluation metric. CIS doesn't — it uses inline DLA (sparse * weights) in training and a separate compute_direct_logit_attribution() function in evaluation.

  ---
  8. Sparse Feature Circuits (https://arxiv.org/abs/2403.19647) — Post-hoc done right, showing what integrated would look like

  This paper is entirely post-hoc but instructive. Their linear approximation for causal attribution:

  IE_atp = ∇_a m|_(a=a_clean) · (a_patch - a_clean)

  operates on SAE-decomposed features. CIS already has a superior decomposition (exact DLA, not approximated), but doesn't exploit it fully because the training-time attribution
  and evaluation-time attribution are separate code paths.

  ---
  The Unified Design These Papers Collectively Mandate

  Every paper points to the same conclusion: the circuit is not a post-hoc discovery — it is a computational object that lives across the entire lifecycle.

  CURRENT ARCHITECTURE (fragmented):

    training/                          mechanistic/
    ┌─────────────────────┐            ┌─────────────────────┐
    │ circuit_losses.py   │            │ attribution.py      │
    │  - inline DLA       │            │  - VocabularyAttrib  │
    │  - soft masking     │  NO SHARED │  - compute_DLA()    │
    │  - centroid tracker │  CODE PATH │                     │
    │  - Gini sharpness   │◄──────────►│ circuits.py         │
    │                     │            │  - VocabularyCircuit │
    │ losses.py           │            │  - extract_circuit() │
    │  - DF-FLOPS         │            │  - measure_complete. │
    └─────────────────────┘            │                     │
                                       │ metrics.py          │
                                       │  - semantic fidelity │
                                       │  - Jaccard (not cos) │
                                       │                     │
                                       │ patching.py         │
                                       │  - hard ablation    │
                                       └─────────────────────┘

  WHAT THE LITERATURE DEMANDS (unified):

    circuits/                    ← single module owns the circuit object
    ┌──────────────────────────────────────────────────────┐
    │                                                      │
    │  core.py                                             │
    │  ┌────────────────────────────────────────────────┐  │
    │  │ CircuitState                                   │  │
    │  │  - W_eff (computed once per forward pass)      │  │
    │  │  - attribution: s ⊙ W_eff[c,:]                │  │
    │  │  - mask(temperature) → soft OR hard            │  │
    │  │  - centroids (EMA, always updated)             │  │
    │  │  - sharpness (Gini, always measured)           │  │
    │  │  - completeness (always measurable)            │  │
    │  └────────────────────────────────────────────────┘  │
    │                                                      │
    │  losses.py                                           │
    │  ┌────────────────────────────────────────────────┐  │
    │  │ Uses CircuitState directly:                    │  │
    │  │  - completeness_loss(state)                    │  │
    │  │  - separation_loss(state)                      │  │
    │  │  - sharpness_loss(state)                       │  │
    │  │  - df_flops_loss(state.sparse_vector)          │  │
    │  └────────────────────────────────────────────────┘  │
    │                                                      │
    │  metrics.py                                          │
    │  ┌────────────────────────────────────────────────┐  │
    │  │ Uses SAME CircuitState:                        │  │
    │  │  - verify_dla(state) → error                   │  │
    │  │  - extract_circuit(state) → VocabularyCircuit  │  │
    │  │  - measure_completeness(state) → float         │  │
    │  │  - measure_separation(state) → float           │  │
    │  │  - measure_sharpness(state) → float            │  │
    │  │  - semantic_fidelity(state) → dict             │  │
    │  └────────────────────────────────────────────────┘  │
    │                                                      │
    └──────────────────────────────────────────────────────┘

  The CircuitState Object — What the Forward Pass Should Return

  Grounded in CoDA-Nets' principle that y(x) = W_(0→L)(x) · x, the forward pass should produce:

  @dataclass
  class CircuitState:
      """Single object computed once, used everywhere."""
      sparse_vector: Tensor    # [B, V] — the bottleneck activations
      logits: Tensor           # [B, C] — classification output
      W_eff: Tensor            # [B, C, V] — effective weight matrix
      b_eff: Tensor            # [B, C] — effective bias
      attribution: Tensor      # [B, C, V] — s_j · W_eff[c,j] per class
      activation_mask: Tensor  # [B, H] — ReLU binary mask D(s)

  This object is the circuit. It is not discovered — it is produced. The same object flows into:

  1. Training losses — completeness_loss(state) uses state.attribution and state.sparse_vector directly, applying sigmoid(T * (|attr| - threshold)) as a differentiable mask
  2. Circuit extraction — extract_circuit(state) uses the same state.attribution with threshold applied as a hard cutoff (T → ∞)
  3. DLA verification — verify_dla(state) checks sum(state.attribution[b,c,:]) + state.b_eff[b,c] ≈ state.logits[b,c]
  4. Semantic fidelity — semantic_fidelity(state) computes Jaccard over state.attribution top-k sets
  5. Inference explanation — explain(state) ranks state.attribution[b,c,:] by magnitude

  Why This Eliminates the Fragmentation
  Current duplication: circuit_losses.py:141 reimplements DLA inline
  Unified resolution: state.attribution computed once in forward pass
  ────────────────────────────────────────
  Current duplication: attribution.py:38 reimplements DLA via function
  Unified resolution: Same state.attribution
  ────────────────────────────────────────
  Current duplication: inference.py:84 reimplements DLA a third time
  Unified resolution: Same state.attribution
  ────────────────────────────────────────
  Current duplication: circuit_losses.py:149 soft mask ≠ circuits.py:71 hard mask
  Unified resolution: state.mask(temperature=T) — one method, two modes
  ────────────────────────────────────────
  Current duplication: circuit_losses.py:175 cosine separation ≠ metrics.py:63 Jaccard separation
  Unified resolution: One metric: cosine on centroids (training) becomes Jaccard on circuits (eval) through the same centroid tracker
  ────────────────────────────────────────
  Current duplication: AttributionCentroidTracker dies after training
  Unified resolution: Persists in CircuitState history, queried by evaluation
  ────────────────────────────────────────
  Current duplication: compute_effective_weights() called separately
  Unified resolution: Part of forward pass, never recomputed
  What the Constrained Formulation (from https://arxiv.org/abs/2512.05865) Would Replace

  Instead of hardwired lambdas:
  # Current: fixed constants
  loss = L_CE + λ(t)·L_DF + w(t)·(0.1·L_comp + 0.05·L_sep + 0.01·L_sharp)

  A GECO-style constrained formulation:
  # Proposed: self-regulating
  # minimize  L_comp + L_sep + L_sharp  (circuit quality)
  # subject to  L_CE ≤ τ_ce            (accuracy floor)
  #             L_DF ≤ τ_df            (sparsity floor)
  #
  # Lagrangian: L_circuit + λ_ce·(L_CE - τ_ce) + λ_df·(L_DF - τ_df)
  # λ_ce, λ_df updated via dual gradient ascent

  This makes the circuit losses the primary objective and accuracy the constraint — which is the correct formulation for a project whose thesis is that circuits are the point, not
  an afterthought.

  ---
  Sources:
  - https://arxiv.org/abs/2104.00032
  - https://arxiv.org/abs/2407.03779
  - https://arxiv.org/abs/2505.10039
  - https://arxiv.org/abs/2401.03646
  - https://arxiv.org/abs/2310.07711
  - https://arxiv.org/abs/2512.05865
  - https://arxiv.org/abs/2501.16496
  - https://arxiv.org/abs/2304.14997
  - https://arxiv.org/abs/2403.19647
  - https://arxiv.org/abs/2407.00886
  - https://arxiv.org/abs/2312.17591
  - https://arxiv.org/abs/1703.03717
  - https://arxiv.org/abs/2502.20268
