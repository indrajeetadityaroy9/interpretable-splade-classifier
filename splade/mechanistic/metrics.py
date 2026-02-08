import torch

from splade.mechanistic.attribution import compute_attribution_tensor
from splade.utils.cuda import COMPUTE_DTYPE, unwrap_compiled


def measure_semantic_fidelity(
    model: torch.nn.Module,
    input_ids_list: list[torch.Tensor],
    attention_mask_list: list[torch.Tensor],
    labels: list[int],
    tokenizer,
    top_k: int = 10,
    precomputed_attributions: dict[int, torch.Tensor] | None = None,
) -> dict:
    """Measure whether high-attribution tokens form semantically coherent explanations.

    Computes per-class token frequency overlap and consistency metrics.

    If precomputed_attributions is provided (e.g. from centroid_tracker), uses
    those to derive class-level top tokens and cross-class overlap, skipping
    per-sample forward passes for those metrics. Per-sample within-class
    consistency still requires forward passes.
    """
    _model = unwrap_compiled(model)
    num_classes = _model.classifier_fc2.weight.shape[0]

    # Per-sample top-k tokens (always computed — needed for within-class Jaccard)
    sample_top_tokens: list[set[int]] = []
    sample_labels: list[int] = []
    class_token_counts: dict[int, dict[int, int]] = {c: {} for c in range(num_classes)}

    for input_ids, attention_mask, label in zip(input_ids_list, attention_mask_list, labels):
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=COMPUTE_DTYPE):
            _, sparse_vector, W_eff, _ = _model(input_ids, attention_mask)

        attr = compute_attribution_tensor(
            sparse_vector, W_eff,
            torch.tensor([label], device=sparse_vector.device),
        )
        top_k_ids = attr[0].abs().topk(top_k).indices.tolist()

        sample_top_tokens.append(set(top_k_ids))
        sample_labels.append(label)

        for tid in top_k_ids:
            class_token_counts[label][tid] = class_token_counts[label].get(tid, 0) + 1

    # Class-level top tokens: use precomputed centroids if available
    class_top_tokens: dict[int, set[int]] = {}
    if precomputed_attributions is not None:
        for c, attr_vec in precomputed_attributions.items():
            class_top_tokens[c] = set(attr_vec.abs().topk(top_k).indices.tolist())
    else:
        for c in range(num_classes):
            counts = class_token_counts[c]
            sorted_tokens = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_k]
            class_top_tokens[c] = set(tid for tid, _ in sorted_tokens)

    # Within-class consistency: average pairwise Jaccard across samples of the same class
    within_class_overlaps = []
    for c in range(num_classes):
        class_indices = [i for i, lbl in enumerate(sample_labels) if lbl == c]
        for i_idx in range(len(class_indices)):
            for j_idx in range(i_idx + 1, len(class_indices)):
                set_i = sample_top_tokens[class_indices[i_idx]]
                set_j = sample_top_tokens[class_indices[j_idx]]
                if set_i and set_j:
                    intersection = len(set_i & set_j)
                    union = len(set_i | set_j)
                    within_class_overlaps.append(intersection / union if union > 0 else 0.0)
    within_class = sum(within_class_overlaps) / len(within_class_overlaps) if within_class_overlaps else 0.0

    # Cross-class overlap: Jaccard between per-class aggregate top-token sets
    cross_class_vals = []
    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            set_i = class_top_tokens.get(i, set())
            set_j = class_top_tokens.get(j, set())
            if set_i and set_j:
                intersection = len(set_i & set_j)
                union = len(set_i | set_j)
                cross_class_vals.append(intersection / union if union > 0 else 0.0)
            else:
                cross_class_vals.append(0.0)
    cross_class = sum(cross_class_vals) / len(cross_class_vals) if cross_class_vals else 0.0

    separation = within_class - cross_class

    class_top_names = {}
    for c in range(num_classes):
        token_ids = list(class_top_tokens.get(c, set()))
        class_top_names[c] = tokenizer.convert_ids_to_tokens(token_ids) if token_ids else []

    return {
        "within_class_consistency": within_class,
        "cross_class_overlap": cross_class,
        "class_separation": separation,
        "class_top_tokens": class_top_names,
    }
