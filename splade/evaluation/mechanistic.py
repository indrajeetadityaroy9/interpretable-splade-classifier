from dataclasses import dataclass, field

import torch

from splade.evaluation.compare_explainers import run_explainer_comparison
from splade.evaluation.eraser import run_eraser_evaluation
from splade.mechanistic.attribution import compute_attribution_tensor
from splade.mechanistic.layerwise import run_layerwise_evaluation
from splade.circuits.metrics import (VocabularyCircuit,
                                     extract_vocabulary_circuit,
                                     measure_circuit_completeness,
                                     measure_separation_cosine,
                                     measure_separation_jaccard,
                                     visualize_circuit)
from splade.training.constants import CIRCUIT_FRACTION
from splade.utils.cuda import COMPUTE_DTYPE, DEVICE, unwrap_compiled


@dataclass
class MechanisticResults:
    accuracy: float = 0.0
    circuits: dict[int, VocabularyCircuit] = field(default_factory=dict)
    circuit_completeness: dict[int, float] = field(default_factory=dict)
    semantic_fidelity: dict = field(default_factory=dict)
    dla_verification_error: float = 0.0
    eraser_metrics: dict = field(default_factory=dict)
    explainer_comparison: dict = field(default_factory=dict)
    layerwise_attribution: dict = field(default_factory=dict)
    sae_comparison: dict = field(default_factory=dict)


def _run_sae_comparison(
    model: torch.nn.Module,
    input_ids_list: list[torch.Tensor],
    attention_mask_list: list[torch.Tensor],
    labels: list[int],
    tokenizer,
) -> dict:
    """Train SAE and compare with DLA in terms of active feature counts."""
    from splade.mechanistic.sae import compute_sae_attribution, train_sae_on_splade

    _model = unwrap_compiled(model)

    print("Training SAE on hidden states...")
    sae = train_sae_on_splade(model, input_ids_list, attention_mask_list)

    with torch.inference_mode():
        classifier_weight = _model.classifier_fc2.weight
        vocab_projector_weight = _model.vocab_projector.weight

    dla_active_counts = []
    sae_active_counts = []

    for input_ids, attention_mask, label in zip(input_ids_list, attention_mask_list, labels):
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=COMPUTE_DTYPE):
            _, sparse_vector, _, _ = _model(input_ids, attention_mask)
            bert_output = _model.bert(input_ids=input_ids, attention_mask=attention_mask)
            hidden = bert_output.last_hidden_state
            transformed = _model.vocab_transform(hidden)
            transformed = torch.nn.functional.gelu(transformed)
            transformed = _model.vocab_layer_norm(transformed)
            cls_hidden = transformed[:, 0, :]

        # DLA active count
        dla_active = int((sparse_vector[0] > 0).sum().item())
        dla_active_counts.append(dla_active)

        # SAE active count
        sae_attrib = compute_sae_attribution(
            sae, cls_hidden, classifier_weight, label, vocab_projector_weight,
        )
        sae_active = int((sae_attrib.abs() > 1e-6).sum().item())
        sae_active_counts.append(sae_active)

    # SAE reconstruction error
    all_hidden = []
    for input_ids, attention_mask in zip(input_ids_list, attention_mask_list):
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=COMPUTE_DTYPE):
            bert_output = _model.bert(input_ids=input_ids, attention_mask=attention_mask)
            hidden = bert_output.last_hidden_state
            transformed = _model.vocab_transform(hidden)
            transformed = torch.nn.functional.gelu(transformed)
            transformed = _model.vocab_layer_norm(transformed)
        all_hidden.append(transformed[:, 0, :].detach().float())

    hidden_tensor = torch.cat(all_hidden, dim=0)
    with torch.inference_mode():
        reconstruction, _ = sae(hidden_tensor)
        recon_error = float(torch.nn.functional.mse_loss(
            reconstruction, hidden_tensor,
        ).item())

    return {
        "dla_active_tokens": sum(dla_active_counts) / len(dla_active_counts),
        "sae_active_features": sum(sae_active_counts) / len(sae_active_counts),
        "reconstruction_error": recon_error,
    }


def run_mechanistic_evaluation(
    model: torch.nn.Module,
    input_ids_list: list[torch.Tensor],
    attention_mask_list: list[torch.Tensor],
    labels: list[int],
    tokenizer,
    num_classes: int,
    circuit_fraction: float = CIRCUIT_FRACTION,
    run_sae_comparison: bool = False,
    centroid_tracker=None,
) -> MechanisticResults:
    """Run the full mechanistic interpretability evaluation suite."""
    _model = unwrap_compiled(model)
    results = MechanisticResults()

    # 1. Verify DLA invariant: sum(s_j * W_eff[c,j]) + b_eff_c == logit_c
    total_error = 0.0
    correct = 0
    n = 0
    eval_batch = 32
    for start in range(0, len(input_ids_list), eval_batch):
        end = min(start + eval_batch, len(input_ids_list))
        batch_ids = torch.cat(input_ids_list[start:end], dim=0)
        batch_mask = torch.cat(attention_mask_list[start:end], dim=0)
        batch_labels_t = torch.tensor(labels[start:end], device=DEVICE)

        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=COMPUTE_DTYPE):
            logits, sparse_vector, W_eff, b_eff = _model(batch_ids, batch_mask)

        preds = logits.argmax(dim=-1)
        correct += (preds == batch_labels_t).sum().item()

        attr = compute_attribution_tensor(sparse_vector, W_eff, batch_labels_t)
        actual = logits.gather(1, batch_labels_t.unsqueeze(1)).squeeze(1)
        reconstructed = attr.sum(dim=-1) + b_eff.gather(1, batch_labels_t.unsqueeze(1)).squeeze(1)
        total_error += (actual - reconstructed).abs().sum().item()
        n += end - start

    results.accuracy = correct / n if n > 0 else 0.0
    results.dla_verification_error = total_error / n if n > 0 else 0.0

    # 2. Extract circuits per class
    for class_idx in range(num_classes):
        class_inputs = [
            (ids, mask) for ids, mask, label in zip(input_ids_list, attention_mask_list, labels)
            if label == class_idx
        ]
        if not class_inputs:
            continue

        class_ids, class_masks = zip(*class_inputs)
        precomputed = None
        if centroid_tracker is not None and class_idx < centroid_tracker.num_classes:
            if centroid_tracker._initialized[class_idx]:
                precomputed = centroid_tracker.centroids[class_idx]

        circuit = extract_vocabulary_circuit(
            model, list(class_ids), list(class_masks),
            tokenizer, class_idx, circuit_fraction=circuit_fraction,
            precomputed_attributions=precomputed,
        )
        results.circuits[class_idx] = circuit

    # 3. Measure circuit completeness
    for class_idx, circuit in results.circuits.items():
        if not circuit.token_ids:
            continue
        completeness = measure_circuit_completeness(
            model, input_ids_list, attention_mask_list, labels, circuit,
        )
        circuit.completeness_score = completeness
        results.circuit_completeness[class_idx] = completeness

    # 4. Separation metrics (cosine primary, Jaccard supplementary)
    precomputed_dict = None
    if centroid_tracker is not None:
        precomputed_dict = {
            c: centroid_tracker.centroids[c]
            for c in range(num_classes)
            if centroid_tracker._initialized[c]
        }

    # Only report cosine separation when centroids were actually trained
    # (uninitialized zero centroids trivially give 1.0, which is meaningless)
    has_centroids = (
        centroid_tracker is not None
        and centroid_tracker._initialized.any()
    )
    cosine_sep = measure_separation_cosine(centroid_tracker) if has_centroids else None
    jaccard_result = measure_separation_jaccard(
        model, input_ids_list, attention_mask_list, labels, tokenizer,
        precomputed_attributions=precomputed_dict,
    )
    results.semantic_fidelity = {
        "cosine_separation": cosine_sep,
        **jaccard_result,
    }

    # 5. ERASER faithfulness metrics (FMM bottleneck-level erasure)
    results.eraser_metrics = run_eraser_evaluation(
        model, input_ids_list, attention_mask_list, labels,
    )

    # 6. Baseline explainer comparison (DLA vs gradient vs IG vs attention)
    results.explainer_comparison = run_explainer_comparison(
        model, input_ids_list, attention_mask_list, labels,
    )

    # 7. Layerwise attribution decomposition
    results.layerwise_attribution = run_layerwise_evaluation(
        model, input_ids_list, attention_mask_list, labels,
    )

    # 8. SAE baseline comparison (optional)
    if run_sae_comparison:
        results.sae_comparison = _run_sae_comparison(
            model, input_ids_list, attention_mask_list, labels, tokenizer,
        )

    return results


def print_mechanistic_results(results: MechanisticResults) -> None:
    print("\n" + "=" * 80)
    print("MECHANISTIC INTERPRETABILITY RESULTS")
    print("=" * 80)

    print(f"\n  Accuracy: {results.accuracy:.4f}")
    print(f"  DLA Verification Error: {results.dla_verification_error:.6f}")

    if results.circuits:
        print(f"\n--- VOCABULARY CIRCUITS ---")
        for class_idx, circuit in sorted(results.circuits.items()):
            completeness = results.circuit_completeness.get(class_idx, 0.0)
            print(f"\n  Class {class_idx}: {len(circuit.token_ids)} tokens, "
                  f"completeness={completeness:.4f}")
            print(visualize_circuit(circuit))

    if results.semantic_fidelity:
        sf = results.semantic_fidelity
        print(f"\n--- SEPARATION METRICS ---")
        cos_sep = sf.get('cosine_separation')
        cos_str = f"{cos_sep:.4f}" if cos_sep is not None else "N/A (no trained centroids)"
        print(f"  Cosine separation (primary): {cos_str}")
        print(f"  Within-class consistency (Jaccard): {sf.get('within_class_consistency', 0):.4f}")
        print(f"  Cross-class overlap (Jaccard): {sf.get('cross_class_overlap', 0):.4f}")
        print(f"  Class separation (Jaccard): {sf.get('class_separation', 0):.4f}")

        if "class_top_tokens" in sf:
            for c, tokens in sf["class_top_tokens"].items():
                print(f"  Class {c} top tokens: {', '.join(tokens[:10])}")

    if results.eraser_metrics:
        er = results.eraser_metrics
        print(f"\n--- ERASER FAITHFULNESS (FMM bottleneck-level) ---")
        comp = er.get("comprehensiveness", {})
        suff = er.get("sufficiency", {})
        print(f"  AOPC Comprehensiveness: {er.get('aopc_comprehensiveness', 0):.4f}")
        print(f"  AOPC Sufficiency:       {er.get('aopc_sufficiency', 0):.4f}")
        print(f"  {'k%':<8} {'Comp':>10} {'Suff':>10}")
        for k in sorted(comp.keys()):
            print(f"  {k:<8.0%} {comp[k]:>10.4f} {suff.get(k, 0):>10.4f}")

    if results.explainer_comparison:
        print(f"\n--- EXPLAINER COMPARISON (ERASER metrics) ---")
        header = f"  {'Method':<25} {'AOPC-C':>8} {'AOPC-S':>8} {'Time(s)':>10}"
        print(header)
        print(f"  {'-' * 55}")
        for name, metrics in results.explainer_comparison.items():
            print(
                f"  {name:<25} "
                f"{metrics.get('aopc_comp', 0):>8.4f} "
                f"{metrics.get('aopc_suff', 0):>8.4f} "
                f"{metrics.get('time_seconds', 0):>10.3f}"
            )

    if results.layerwise_attribution:
        lw = results.layerwise_attribution
        print(f"\n--- LAYERWISE ATTRIBUTION ---")
        print(f"  Decomposition error: {lw.get('decomposition_error', 0):.4f}")
        print(f"  Samples: {lw.get('num_samples', 0)}")
        importance = lw.get("layer_importance", [])
        if importance:
            total = sum(importance) or 1.0
            print(f"  {'Layer':<8} {'Importance':>12} {'Fraction':>10}")
            for i, imp in enumerate(importance):
                print(f"  {i:<8} {imp:>12.4f} {imp/total:>10.1%}")

    if results.sae_comparison:
        sae = results.sae_comparison
        print(f"\n--- SAE BASELINE COMPARISON ---")
        print(f"  DLA active tokens (mean): {sae.get('dla_active_tokens', 0):.1f}")
        print(f"  SAE active features (mean): {sae.get('sae_active_features', 0):.1f}")
        print(f"  SAE reconstruction error: {sae.get('reconstruction_error', 0):.4f}")

    print("\n" + "=" * 80)
