from dataclasses import dataclass, field

import torch

from splade.mechanistic.attribution import compute_attribution_tensor
from splade.mechanistic.circuits import (VocabularyCircuit,
                                         extract_vocabulary_circuit,
                                         measure_circuit_completeness,
                                         visualize_circuit)
from splade.mechanistic.metrics import measure_semantic_fidelity
from splade.utils.cuda import COMPUTE_DTYPE, unwrap_compiled


@dataclass
class MechanisticResults:
    accuracy: float = 0.0
    circuits: dict[int, VocabularyCircuit] = field(default_factory=dict)
    circuit_completeness: dict[int, float] = field(default_factory=dict)
    semantic_fidelity: dict = field(default_factory=dict)
    dla_verification_error: float = 0.0
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
    circuit_threshold: float = 0.01,
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
    for input_ids, attention_mask, label in zip(input_ids_list, attention_mask_list, labels):
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=COMPUTE_DTYPE):
            logits, sparse_vector, W_eff, b_eff = _model(input_ids, attention_mask)

        pred = int(logits.argmax(dim=-1).item())
        if pred == label:
            correct += 1

        actual_logit = float(logits[0, label].item())
        attr = compute_attribution_tensor(
            sparse_vector, W_eff, torch.tensor([label], device=sparse_vector.device),
        )
        reconstructed = float(attr[0].sum().item()) + float(b_eff[0, label].item())
        total_error += abs(actual_logit - reconstructed)
        n += 1

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
            tokenizer, class_idx, threshold=circuit_threshold,
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

    # 4. Semantic fidelity
    precomputed_dict = None
    if centroid_tracker is not None:
        precomputed_dict = {
            c: centroid_tracker.centroids[c]
            for c in range(num_classes)
            if centroid_tracker._initialized[c]
        }

    results.semantic_fidelity = measure_semantic_fidelity(
        model, input_ids_list, attention_mask_list, labels, tokenizer,
        precomputed_attributions=precomputed_dict,
    )

    # 5. SAE baseline comparison (optional)
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
            print(visualize_circuit(circuit, max_tokens=10))

    if results.semantic_fidelity:
        sf = results.semantic_fidelity
        print(f"\n--- SEMANTIC FIDELITY ---")
        print(f"  Within-class consistency: {sf.get('within_class_consistency', 0):.4f}")
        print(f"  Cross-class overlap: {sf.get('cross_class_overlap', 0):.4f}")
        print(f"  Class separation: {sf.get('class_separation', 0):.4f}")

        if "class_top_tokens" in sf:
            for c, tokens in sf["class_top_tokens"].items():
                print(f"  Class {c} top tokens: {', '.join(tokens[:10])}")

    if results.sae_comparison:
        sae = results.sae_comparison
        print(f"\n--- SAE BASELINE COMPARISON ---")
        print(f"  DLA active tokens (mean): {sae.get('dla_active_tokens', 0):.1f}")
        print(f"  SAE active features (mean): {sae.get('sae_active_features', 0):.1f}")
        print(f"  SAE reconstruction error: {sae.get('reconstruction_error', 0):.4f}")

    print("\n" + "=" * 80)
