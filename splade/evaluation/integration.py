"""Circuit-faithfulness alignment analysis.

Connects mechanistic circuits to faithfulness metrics by measuring
how well each explainer's attributions align with the vocabulary
circuits extracted from the model's internals.
"""

import numpy
from scipy import stats

from splade.evaluation.benchmark import InterpretabilityResult
from splade.mechanistic.circuits import VocabularyCircuit


def analyze_circuit_faithfulness_alignment(
    circuits: dict[int, VocabularyCircuit],
    explainer_results: list[InterpretabilityResult],
    attributions_per_explainer: dict[str, list[list[tuple[str, float]]]],
    labels: list[int],
    top_k: int = 10,
) -> dict:
    """Measure alignment between explainer attributions and vocabulary circuits.

    For each explainer, computes per-sample Jaccard overlap between the
    top-k attributed tokens and the class-specific circuit token set.
    Reports per-explainer mean circuit overlap and cross-explainer
    correlations between overlap and faithfulness metrics.

    Args:
        circuits: Per-class vocabulary circuits from mechanistic evaluation.
        explainer_results: InterpretabilityResult per explainer (same order
            as attributions_per_explainer keys).
        attributions_per_explainer: Map from explainer name to per-sample
            word-level attributions.
        labels: Per-sample ground-truth labels.
        top_k: Number of top tokens to consider from each attribution.

    Returns:
        Dict with per-explainer overlap scores and cross-explainer correlations.
    """
    # Build circuit token name sets per class
    circuit_token_names: dict[int, set[str]] = {}
    for class_idx, circuit in circuits.items():
        circuit_token_names[class_idx] = {
            name.lower() for name in circuit.token_names
        }

    # Compute per-explainer mean circuit overlap
    explainer_overlaps: dict[str, float] = {}
    for explainer_name, sample_attributions in attributions_per_explainer.items():
        overlaps = []
        for i, attribs in enumerate(sample_attributions):
            label = labels[i]
            circuit_set = circuit_token_names.get(label, set())
            if not circuit_set or not attribs:
                continue
            sorted_attribs = sorted(attribs, key=lambda x: abs(x[1]), reverse=True)
            attrib_set = {tok.lower() for tok, _ in sorted_attribs[:top_k]}
            if not attrib_set:
                continue
            intersection = len(attrib_set & circuit_set)
            union = len(attrib_set | circuit_set)
            jaccard = intersection / union if union > 0 else 0.0
            overlaps.append(jaccard)

        explainer_overlaps[explainer_name] = float(numpy.mean(overlaps)) if overlaps else 0.0

    # Cross-explainer correlations: overlap vs faithfulness metrics
    result_by_name = {r.name: r for r in explainer_results}
    correlations = {}

    explainer_names = list(explainer_overlaps.keys())
    if len(explainer_names) >= 3:
        overlap_vals = [explainer_overlaps[n] for n in explainer_names]

        # Get median-k values for dict metrics
        from splade.evaluation.constants import K_VALUES
        k_mid = K_VALUES[len(K_VALUES) // 2]

        metric_vectors = {
            "comprehensiveness": [
                result_by_name.get(n, InterpretabilityResult(name=n))
                .eraser_comprehensiveness.get(k_mid, 0.0)
                for n in explainer_names
            ],
            "soft_comprehensiveness": [
                result_by_name.get(n, InterpretabilityResult(name=n))
                .soft_comprehensiveness
                for n in explainer_names
            ],
            "filler_comprehensiveness": [
                result_by_name.get(n, InterpretabilityResult(name=n))
                .filler_comprehensiveness.get(k_mid, 0.0)
                for n in explainer_names
            ],
        }

        for metric_name, metric_vals in metric_vectors.items():
            if any(v != 0 for v in metric_vals):
                rho, p_value = stats.spearmanr(overlap_vals, metric_vals)
                correlations[f"overlap_vs_{metric_name}"] = {
                    "spearman_rho": float(rho) if not numpy.isnan(rho) else 0.0,
                    "p_value": float(p_value) if not numpy.isnan(p_value) else 1.0,
                }

    return {
        "explainer_circuit_overlap": explainer_overlaps,
        "correlations": correlations,
    }
