import time
from dataclasses import dataclass, field

import numpy

from splade.evaluation.causal import (MLMCounterfactualGenerator,
                                      compute_causal_faithfulness)
from splade.evaluation.constants import (K_MAX, K_VALUES,
                                         SOFT_METRIC_N_SAMPLES,
                                         MONOTONICITY_MAX_STEPS,
                                         NAOPC_BEAM_SIZE,
                                         ADVERSARIAL_N_PERTURBATIONS)
from splade.evaluation.f_fidelity import compute_f_fidelity
from splade.evaluation.faithfulness import (UnigramSampler,
                                            compute_adversarial_sensitivity,
                                            compute_aopc,
                                            compute_eraser_comprehensiveness,
                                            compute_eraser_sufficiency,
                                            compute_filler_comprehensiveness,
                                            compute_monotonicity,
                                            compute_naopc,
                                            compute_soft_metrics)
from splade.evaluation.token_alignment import normalize_attributions_to_words


@dataclass
class InterpretabilityResult:
    name: str
    filler_comprehensiveness: dict[int, float] = field(default_factory=dict)
    soft_comprehensiveness: float = 0.0
    soft_sufficiency: float = 0.0
    accuracy: float = 0.0
    inference_latency: float = 0.0
    causal_faithfulness: float = 0.0
    eraser_comprehensiveness: dict[int, float] = field(default_factory=dict)
    eraser_sufficiency: dict[int, float] = field(default_factory=dict)
    monotonicity: float = 0.0
    aopc: dict[int, float] = field(default_factory=dict)
    naopc: dict[int, float] = field(default_factory=dict)
    f_fidelity_pos: float = 0.0
    f_fidelity_neg: float = 0.0
    adversarial_sensitivity: float = 0.0


def aggregate_results(results_list: list[list[InterpretabilityResult]]) -> list[dict]:
    if not results_list:
        return []

    methods = [r.name for r in results_list[0]]
    aggregated = []

    for i, name in enumerate(methods):
        method_seeds = [seeds[i] for seeds in results_list]
        stats = {"name": name}

        scalar_metrics = [
            "soft_comprehensiveness", "soft_sufficiency",
            "accuracy", "inference_latency", "causal_faithfulness",
            "monotonicity", "f_fidelity_pos", "f_fidelity_neg",
            "adversarial_sensitivity",
        ]

        for metric in scalar_metrics:
            vals = [getattr(s, metric) for s in method_seeds]
            stats[f"{metric}_mean"] = float(numpy.nanmean(vals))
            stats[f"{metric}_std"] = float(numpy.nanstd(vals))

        k_val = K_VALUES[len(K_VALUES) // 2]
        stats["k"] = k_val
        dict_metrics = [
            "filler_comprehensiveness", "eraser_comprehensiveness",
            "eraser_sufficiency", "aopc", "naopc",
        ]
        for metric in dict_metrics:
            vals = [getattr(s, metric).get(k_val, 0.0) for s in method_seeds]
            stats[f"{metric}_mean"] = float(numpy.mean(vals))
            stats[f"{metric}_std"] = float(numpy.std(vals))

        aggregated.append(stats)
    return aggregated


def print_aggregated_results(aggregated: list[dict]):
    print("\n" + "=" * 130)
    print(f"{'Method':<16} {'Acc':>10} {'Filler':>10} {'ERASER-C':>10} {'ERASER-S':>10} "
          f"{'SoftC':>10} {'Causal':>10} {'Mono':>8} {'AOPC':>10} {'NAOPC':>10} "
          f"{'FFid+':>10} {'AdvSens':>10}")
    print("-" * 130)
    for res in aggregated:
        print(
            f"{res['name']:<16} "
            f"{res['accuracy_mean']:>5.3f}+/-{res['accuracy_std']:>4.3f} "
            f"{res['filler_comprehensiveness_mean']:>5.3f}+/-{res['filler_comprehensiveness_std']:>4.3f} "
            f"{res.get('eraser_comprehensiveness_mean', 0):>5.3f}+/-{res.get('eraser_comprehensiveness_std', 0):>4.3f} "
            f"{res.get('eraser_sufficiency_mean', 0):>5.3f}+/-{res.get('eraser_sufficiency_std', 0):>4.3f} "
            f"{res['soft_comprehensiveness_mean']:>5.3f}+/-{res['soft_comprehensiveness_std']:>4.3f} "
            f"{res['causal_faithfulness_mean']:>5.3f}+/-{res['causal_faithfulness_std']:>4.3f} "
            f"{res.get('monotonicity_mean', 0):>5.3f} "
            f"{res.get('aopc_mean', 0):>5.3f}+/-{res.get('aopc_std', 0):>4.3f} "
            f"{res.get('naopc_mean', 0):>5.3f}+/-{res.get('naopc_std', 0):>4.3f} "
            f"{res.get('f_fidelity_pos_mean', 0):>5.3f}+/-{res.get('f_fidelity_pos_std', 0):>4.3f} "
            f"{res.get('adversarial_sensitivity_mean', 0):>5.3f}+/-{res.get('adversarial_sensitivity_std', 0):>4.3f}"
        )
    print("=" * 130)


def print_interpretability_results(results: list[InterpretabilityResult]) -> None:
    k_display = K_VALUES[len(K_VALUES) // 2]

    print("\n" + "=" * 130)
    print("INTERPRETABILITY BENCHMARK RESULTS")
    print("=" * 130)

    print(f"\n{'Method':<20} {'Filler@' + str(k_display):>8} {'ERASER-C':>8} {'ERASER-S':>8} "
          f"{'SoftC':>8} {'SoftS':>8} {'Causal':>8} {'Mono':>8} "
          f"{'AOPC':>8} {'NAOPC':>8} {'FFid+':>8} {'FFid-':>8} {'AdvSens':>8} "
          f"{'Acc':>8} {'Latency':>10}")
    print("-" * 130)
    for result in results:
        filler = result.filler_comprehensiveness.get(k_display, 0.0)
        eraser_c = result.eraser_comprehensiveness.get(k_display, 0.0)
        eraser_s = result.eraser_sufficiency.get(k_display, 0.0)
        aopc = result.aopc.get(k_display, 0.0)
        naopc = result.naopc.get(k_display, 0.0)
        print(
            f"{result.name:<20} {filler:>8.4f} {eraser_c:>8.4f} {eraser_s:>8.4f} "
            f"{result.soft_comprehensiveness:>8.4f} {result.soft_sufficiency:>8.4f} "
            f"{result.causal_faithfulness:>8.4f} {result.monotonicity:>8.4f} "
            f"{aopc:>8.4f} {naopc:>8.4f} "
            f"{result.f_fidelity_pos:>8.4f} {result.f_fidelity_neg:>8.4f} "
            f"{result.adversarial_sensitivity:>8.4f} "
            f"{result.accuracy:>8.4f} {result.inference_latency*1000:>9.2f}ms"
        )

    print("\n" + "-" * 130)
    print("Interpretation:")
    print("  Filler: Filler-token comprehensiveness (higher = better, OOD-robust)")
    print("  ERASER-C/S: ERASER comprehensiveness/sufficiency (higher comp = better, lower suff = better)")
    print("  Soft Comp/Suff: Probabilistic embedding perturbation (higher = better)")
    print("  Causal: Counterfactual consistency (higher = better)")
    print("  Mono: Monotonicity — fraction of steps with decreasing confidence (higher = better)")
    print("  AOPC: Area Over the Perturbation Curve (higher = better)")
    print("  NAOPC: Normalized AOPC via beam-search bounds (higher = better)")
    print("  FFid+/-: F-Fidelity positive/negative (higher + / lower - = better)")
    print("  AdvSens: Adversarial sensitivity — ranking stability (higher = more stable)")
    print("  Latency: Inference time per sample in ms (lower = better)")
    print("=" * 130)


def benchmark_explainer(
    clf,
    name: str,
    explain_fn,
    batch_explain_fn,
    test_texts: list[str],
    mask_token: str,
    seed: int,
    sampler: UnigramSampler,
    tokenizer,
    max_length: int = 128,
    surrogate_model=None,
    test_labels: list[int] | None = None,
) -> InterpretabilityResult:
    k_values = list(K_VALUES)

    print(f"\nGenerating explanations for {name}...")
    start = time.time()
    raw_attributions = batch_explain_fn(test_texts, K_MAX)
    explanation_time = time.time() - start

    inference_latency = explanation_time / len(test_texts) if test_texts else 0.0

    attributions = [
        normalize_attributions_to_words(text, attrib, tokenizer)
        for text, attrib in zip(test_texts, raw_attributions)
    ]

    print("Computing metrics...")
    result = InterpretabilityResult(
        name=name,
        inference_latency=inference_latency,
    )

    original_probs = clf.predict_proba(test_texts)

    result.filler_comprehensiveness = compute_filler_comprehensiveness(
        clf, test_texts, attributions, k_values, sampler,
        original_probs=original_probs,
    )

    result.soft_comprehensiveness, result.soft_sufficiency = compute_soft_metrics(
        clf, test_texts, attributions, mask_token,
        n_samples=SOFT_METRIC_N_SAMPLES, seed=seed,
        tokenizer=tokenizer, max_length=max_length,
        original_probs=original_probs,
    )

    result.causal_faithfulness = compute_causal_faithfulness(
        clf.model,
        clf.tokenizer,
        test_texts,
        attributions,
        clf.max_length,
        generator=MLMCounterfactualGenerator(clf.tokenizer.name_or_path),
    )

    result.eraser_comprehensiveness = compute_eraser_comprehensiveness(
        clf, test_texts, attributions, k_values, mask_token,
        original_probs=original_probs,
    )

    result.eraser_sufficiency = compute_eraser_sufficiency(
        clf, test_texts, attributions, k_values, mask_token,
        original_probs=original_probs,
    )

    result.monotonicity = compute_monotonicity(
        clf, test_texts, attributions, mask_token,
        original_probs=original_probs,
        max_steps=MONOTONICITY_MAX_STEPS,
    )

    result.aopc = compute_aopc(
        clf, test_texts, attributions, mask_token,
        original_probs=original_probs, k_values=k_values,
    )

    result.naopc = compute_naopc(
        clf, test_texts, attributions, mask_token,
        original_probs=original_probs, k_values=k_values,
        beam_size=NAOPC_BEAM_SIZE, seed=seed,
    )

    if surrogate_model is not None and test_labels is not None:
        result.f_fidelity_pos, result.f_fidelity_neg = compute_f_fidelity(
            surrogate_model, tokenizer, test_texts, test_labels,
            attributions, max_length, seed=seed,
        )

    result.adversarial_sensitivity = compute_adversarial_sensitivity(
        explain_fn, test_texts, attributions, tokenizer,
        n_perturbations=ADVERSARIAL_N_PERTURBATIONS, seed=seed,
    )

    return result, attributions


@dataclass
class ExperimentResults:
    """Unified results from a single-seed experiment run."""
    config_snapshot: dict = field(default_factory=dict)
    seed: int = 0
    accuracy: float = 0.0
    # Mechanistic
    dla_verification_error: float = 0.0
    circuits: dict = field(default_factory=dict)
    circuit_completeness: dict = field(default_factory=dict)
    semantic_fidelity: dict = field(default_factory=dict)
    sae_comparison: dict = field(default_factory=dict)
    # Faithfulness (per-explainer)
    explainer_results: list[InterpretabilityResult] = field(default_factory=list)
    # Integration
    circuit_faithfulness_alignment: dict = field(default_factory=dict)


def _print_integration_results(alignment: dict) -> None:
    """Print circuit-faithfulness alignment table."""
    overlaps = alignment.get("explainer_circuit_overlap", {})
    if not overlaps:
        return

    print(f"\n--- CIRCUIT-FAITHFULNESS ALIGNMENT ---")
    print(f"  {'Explainer':<20} {'Circuit Overlap':>16}")
    print(f"  {'-'*20} {'-'*16}")
    for name, overlap in sorted(overlaps.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name:<20} {overlap:>16.4f}")

    correlations = alignment.get("correlations", {})
    if correlations:
        print(f"\n  Cross-explainer correlations (overlap vs metric):")
        for corr_name, vals in correlations.items():
            rho = vals.get("spearman_rho", 0.0)
            p = vals.get("p_value", 1.0)
            print(f"    {corr_name}: rho={rho:.3f}, p={p:.3f}")


def print_experiment_results(results: ExperimentResults) -> None:
    """Print CIS experiment results across all evaluation phases."""
    print(f"\n{'='*80}")
    print(f"CIS EXPERIMENT RESULTS (seed={results.seed})")
    print(f"{'='*80}")
    print(f"\n  Accuracy: {results.accuracy:.4f}")
    print(f"  DLA Verification Error: {results.dla_verification_error:.6f}")

    print(f"\n--- VOCABULARY CIRCUITS ---")
    for class_idx in sorted(results.circuits.keys()):
        circuit = results.circuits[class_idx]
        completeness = results.circuit_completeness.get(class_idx, 0.0)
        n_tokens = len(circuit.get("token_ids", [])) if isinstance(circuit, dict) else len(circuit.token_ids)
        print(f"  Class {class_idx}: {n_tokens} tokens, completeness={completeness:.4f}")

    sf = results.semantic_fidelity
    print(f"\n--- SEMANTIC FIDELITY ---")
    print(f"  Within-class consistency: {sf.get('within_class_consistency', 0):.4f}")
    print(f"  Cross-class overlap: {sf.get('cross_class_overlap', 0):.4f}")
    print(f"  Class separation: {sf.get('class_separation', 0):.4f}")

    if results.sae_comparison:
        sae = results.sae_comparison
        print(f"\n--- SAE BASELINE COMPARISON ---")
        print(f"  DLA active tokens (mean): {sae.get('dla_active_tokens', 0):.1f}")
        print(f"  SAE active features (mean): {sae.get('sae_active_features', 0):.1f}")
        print(f"  SAE reconstruction error: {sae.get('reconstruction_error', 0):.4f}")

    if results.explainer_results:
        print_interpretability_results(results.explainer_results)

    if results.circuit_faithfulness_alignment:
        _print_integration_results(results.circuit_faithfulness_alignment)

    print(f"\n{'='*80}")
