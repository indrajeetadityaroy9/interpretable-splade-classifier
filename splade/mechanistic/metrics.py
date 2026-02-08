import numpy
import torch

from splade.mechanistic.attribution import compute_direct_logit_attribution
from splade.utils.cuda import COMPUTE_DTYPE


def measure_superposition_accuracy_tradeoff(
    models: list[torch.nn.Module],
    tokenizers: list,
    input_ids_list: list[torch.Tensor],
    attention_mask_list: list[torch.Tensor],
    labels: list[int],
    lambda_values: list[float],
) -> list[dict]:
    """Measure sparsity vs accuracy across models trained with different regularization.

    Returns a list of dicts with {lambda, accuracy, mean_sparsity, mean_nonzero_dims}.
    """
    results = []

    for model, tokenizer, lam in zip(models, tokenizers, lambda_values):
        _model = model._orig_mod if hasattr(model, "_orig_mod") else model
        correct = 0
        total_sparsity = 0.0
        total_nonzero = 0.0
        n = 0

        for input_ids, attention_mask, label in zip(input_ids_list, attention_mask_list, labels):
            with torch.inference_mode(), torch.amp.autocast("cuda", dtype=COMPUTE_DTYPE):
                logits, sparse_vector = _model(input_ids, attention_mask)

            pred = int(logits.argmax(dim=-1).item())
            if pred == label:
                correct += 1

            sv = sparse_vector[0].cpu()
            nonzero = (sv > 0).sum().item()
            total_nonzero += nonzero
            total_sparsity += 1.0 - (nonzero / sv.shape[0])
            n += 1

        results.append({
            "lambda": lam,
            "accuracy": correct / n if n > 0 else 0.0,
            "mean_sparsity": total_sparsity / n if n > 0 else 0.0,
            "mean_nonzero_dims": total_nonzero / n if n > 0 else 0.0,
        })

    return results


def measure_semantic_fidelity(
    model: torch.nn.Module,
    input_ids_list: list[torch.Tensor],
    attention_mask_list: list[torch.Tensor],
    labels: list[int],
    tokenizer,
    top_k: int = 10,
) -> dict:
    """Measure whether high-attribution tokens form semantically coherent explanations.

    Computes per-class token frequency overlap and consistency metrics.
    """
    _model = model._orig_mod if hasattr(model, "_orig_mod") else model

    with torch.inference_mode():
        classifier_weight = _model.classifier.weight.cpu().numpy()

    num_classes = classifier_weight.shape[0]
    class_token_counts: dict[int, dict[int, int]] = {c: {} for c in range(num_classes)}

    for input_ids, attention_mask, label in zip(input_ids_list, attention_mask_list, labels):
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=COMPUTE_DTYPE):
            _, sparse_vector = _model(input_ids, attention_mask)

        sparse_np = sparse_vector[0].cpu().numpy()
        attrib = compute_direct_logit_attribution(sparse_np, classifier_weight, tokenizer, label)

        for tid in attrib.token_ids[:top_k]:
            class_token_counts[label][int(tid)] = class_token_counts[label].get(int(tid), 0) + 1

    class_top_tokens = {}
    for c in range(num_classes):
        counts = class_token_counts[c]
        sorted_tokens = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_k]
        class_top_tokens[c] = set(tid for tid, _ in sorted_tokens)

    overlap_matrix = numpy.zeros((num_classes, num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            if class_top_tokens[i] and class_top_tokens[j]:
                intersection = len(class_top_tokens[i] & class_top_tokens[j])
                union = len(class_top_tokens[i] | class_top_tokens[j])
                overlap_matrix[i, j] = intersection / union if union > 0 else 0.0

    within_class = float(numpy.mean([overlap_matrix[i, i] for i in range(num_classes)]))

    cross_class_vals = []
    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            cross_class_vals.append(overlap_matrix[i, j])
    cross_class = float(numpy.mean(cross_class_vals)) if cross_class_vals else 0.0

    separation = within_class - cross_class

    class_top_names = {}
    for c in range(num_classes):
        token_ids = list(class_top_tokens[c])
        class_top_names[c] = tokenizer.convert_ids_to_tokens(token_ids) if token_ids else []

    return {
        "within_class_consistency": within_class,
        "cross_class_overlap": cross_class,
        "class_separation": separation,
        "class_top_tokens": class_top_names,
    }
