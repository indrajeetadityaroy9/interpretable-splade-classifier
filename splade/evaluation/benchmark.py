import time
from dataclasses import dataclass, field

import numpy

from splade.evaluation.causal import (MLMCounterfactualGenerator,
                                      compute_causal_faithfulness)
from splade.evaluation.constants import K_MAX, K_VALUES, SOFT_METRIC_N_SAMPLES
from splade.evaluation.faithfulness import (UnigramSampler,
                                            compute_filler_comprehensiveness,
                                            compute_soft_metrics)
from splade.evaluation.token_alignment import normalize_attributions_to_words


@dataclass
class InterpretabilityResult:
    name: str
    filler_comprehensiveness: dict[int, float] = field(default_factory=dict)
    soft_comprehensiveness: float = 0.0
    soft_sufficiency: float = 0.0
    explanation_time: float = 0.0
    accuracy: float = 0.0
    inference_latency: float = 0.0
    causal_faithfulness: float = 0.0


def aggregate_results(results_list: list[list[InterpretabilityResult]]) -> list[dict]:
    if not results_list:
        return []

    methods = [r.name for r in results_list[0]]
    aggregated = []

    for i, name in enumerate(methods):
        method_seeds = [seeds[i] for seeds in results_list]
        stats = {"name": name}

        metrics = [
            "soft_comprehensiveness", "soft_sufficiency",
            "accuracy", "inference_latency", "causal_faithfulness",
        ]

        for metric in metrics:
            vals = [getattr(s, metric) for s in method_seeds]
            stats[f"{metric}_mean"] = float(numpy.nanmean(vals))
            stats[f"{metric}_std"] = float(numpy.nanstd(vals))

        k_val = K_VALUES[len(K_VALUES) // 2]
        stats["k"] = k_val
        for metric in ["filler_comprehensiveness"]:
            vals = [getattr(s, metric).get(k_val, 0.0) for s in method_seeds]
            stats[f"{metric}_mean"] = float(numpy.mean(vals))
            stats[f"{metric}_std"] = float(numpy.std(vals))

        aggregated.append(stats)
    return aggregated


def print_aggregated_results(aggregated: list[dict]):
    print("\n" + "=" * 100)
    print(f"{'Method':<20} {'Accuracy':>12} {'Filler':>12} {'Soft Comp':>12} {'Latency (ms)':>15} {'Causal':>12}")
    print("-" * 100)
    for res in aggregated:
        print(
            f"{res['name']:<20} "
            f"{res['accuracy_mean']:>6.4f}+/-{res['accuracy_std']:>4.4f} "
            f"{res['filler_comprehensiveness_mean']:>6.4f}+/-{res['filler_comprehensiveness_std']:>4.4f} "
            f"{res['soft_comprehensiveness_mean']:>6.4f}+/-{res['soft_comprehensiveness_std']:>4.4f} "
            f"{res['inference_latency_mean']*1000:>8.2f}+/-{res['inference_latency_std']*1000:>4.2f} "
            f"{res['causal_faithfulness_mean']:>6.4f}+/-{res['causal_faithfulness_std']:>4.4f}"
        )
    print("=" * 100)


def print_interpretability_results(results: list[InterpretabilityResult]) -> None:
    k_display = K_VALUES[len(K_VALUES) // 2]

    print("\n" + "=" * 80)
    print("INTERPRETABILITY BENCHMARK RESULTS")
    print("=" * 80)

    print(f"\n{'Method':<30} {'Filler@' + str(k_display):>12} {'Soft Comp':>12} {'Soft Suff':>12} {'Causal':>10} {'Accuracy':>10} {'Latency':>12}")
    print("-" * 100)
    for result in results:
        filler = result.filler_comprehensiveness.get(k_display, 0.0)
        print(
            f"{result.name:<30} {filler:>12.4f} "
            f"{result.soft_comprehensiveness:>12.4f} {result.soft_sufficiency:>12.4f} "
            f"{result.causal_faithfulness:>10.4f} {result.accuracy:>10.4f} "
            f"{result.inference_latency*1000:>11.2f}ms"
        )

    print("\n" + "-" * 80)
    print("Interpretation:")
    print("  Filler: Filler-token comprehensiveness (higher = better, OOD-robust)")
    print("  Soft Comp/Suff: Probabilistic embedding perturbation (higher = better)")
    print("  Causal: Counterfactual consistency (higher = better)")
    print("  Latency: Inference time per sample in ms (lower = better)")
    print("=" * 80)


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
        explanation_time=explanation_time,
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

    return result
