from dataclasses import dataclass, field

import torch

from splade.mechanistic.attribution import (VocabularyAttribution,
                                            compute_direct_logit_attribution)
from splade.mechanistic.circuits import (VocabularyCircuit,
                                         extract_vocabulary_circuit,
                                         measure_circuit_completeness,
                                         visualize_circuit)
from splade.mechanistic.metrics import measure_semantic_fidelity
from splade.mechanistic.patching import ablate_vocabulary_tokens
from splade.utils.cuda import COMPUTE_DTYPE


@dataclass
class MechanisticResults:
    accuracy: float = 0.0
    circuits: dict[int, VocabularyCircuit] = field(default_factory=dict)
    circuit_completeness: dict[int, float] = field(default_factory=dict)
    semantic_fidelity: dict = field(default_factory=dict)
    dla_verification_error: float = 0.0


def run_mechanistic_evaluation(
    model: torch.nn.Module,
    input_ids_list: list[torch.Tensor],
    attention_mask_list: list[torch.Tensor],
    labels: list[int],
    tokenizer,
    num_classes: int,
    circuit_threshold: float = 0.01,
) -> MechanisticResults:
    """Run the full mechanistic interpretability evaluation suite."""
    _model = model._orig_mod if hasattr(model, "_orig_mod") else model
    results = MechanisticResults()

    # 1. Verify DLA invariant: sum(attribution) == logit
    with torch.inference_mode():
        classifier_weight = _model.classifier.weight.cpu().numpy()
        classifier_bias = _model.classifier.bias.cpu().numpy() if _model.classifier.bias is not None else None

    total_error = 0.0
    correct = 0
    n = 0
    for input_ids, attention_mask, label in zip(input_ids_list, attention_mask_list, labels):
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=COMPUTE_DTYPE):
            logits, sparse_vector = _model(input_ids, attention_mask)

        pred = int(logits.argmax(dim=-1).item())
        if pred == label:
            correct += 1

        sparse_np = sparse_vector[0].cpu().numpy()
        actual_logit = float(logits[0, label].item())
        attrib = compute_direct_logit_attribution(sparse_np, classifier_weight, tokenizer, label)
        reconstructed = attrib.logit
        if classifier_bias is not None:
            reconstructed += classifier_bias[label]
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
        circuit = extract_vocabulary_circuit(
            model, list(class_ids), list(class_masks),
            tokenizer, class_idx, threshold=circuit_threshold,
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
    results.semantic_fidelity = measure_semantic_fidelity(
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

    print("\n" + "=" * 80)
