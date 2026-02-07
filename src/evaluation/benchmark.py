"""Benchmark execution and reporting utilities."""

import time
from dataclasses import dataclass, field

import numpy
from tqdm import tqdm

from src.evaluation.adversarial import compute_adversarial_sensitivity
from src.evaluation.faithfulness import (
    UnigramSampler,
    compute_comprehensiveness,
    compute_filler_comprehensiveness,
    compute_monotonicity,
    compute_normalized_aopc,
    compute_soft_comprehensiveness,
    compute_soft_sufficiency,
    compute_sufficiency,
)


@dataclass(frozen=True)
class BenchmarkConfig:
    """Evaluation hyperparameters for the canonical benchmark path."""

    seed: int = 42
    k_values: tuple[int, ...] = (1, 5, 10, 20)
    ffidelity_beta: float = 0.5
    ffidelity_ft_epochs: int = 3
    ffidelity_ft_lr: float = 1e-5
    ffidelity_ft_batch_size: int = 16
    monotonicity_steps: int = 10
    naopc_beam_size: int = 15
    soft_metric_n_samples: int = 20
    adversarial_mcp_threshold: float = 0.7
    adversarial_max_changes: int = 3
    adversarial_test_samples: int = 50
    ig_n_steps: int = 50
    lime_num_samples: int = 500

    @property
    def k_max(self) -> int:
        return max(self.k_values)

    @property
    def k_display(self) -> int:
        return self.k_values[len(self.k_values) // 2]


@dataclass
class InterpretabilityResult:
    """Collected metrics for one explanation method."""

    name: str
    comprehensiveness: dict[int, float] = field(default_factory=dict)
    sufficiency: dict[int, float] = field(default_factory=dict)
    monotonicity: float = 0.0
    aopc: float = 0.0
    naopc: float = 0.0
    naopc_lower: float = 0.0
    naopc_upper: float = 0.0
    f_comprehensiveness: dict[int, float] = field(default_factory=dict)
    f_sufficiency: dict[int, float] = field(default_factory=dict)
    ffidelity_comp: dict[int, float] = field(default_factory=dict)
    ffidelity_suff: dict[int, float] = field(default_factory=dict)
    filler_comprehensiveness: dict[int, float] = field(default_factory=dict)
    soft_comprehensiveness: float = 0.0
    soft_sufficiency: float = 0.0
    adversarial_sensitivity: float = 0.0
    adversarial_mean_tau: float = 0.0
    explanation_time: float = 0.0
    accuracy: float = 0.0


def print_interpretability_results(results: list[InterpretabilityResult], config: BenchmarkConfig) -> None:
    """Print benchmark tables."""
    print("\n" + "=" * 80)
    print("INTERPRETABILITY BENCHMARK RESULTS")
    print("=" * 80)

    k_display = config.k_display

    print(f"\n--- PRIMARY METRICS (robust to OOD artifacts) ---")
    print(f"\n{'Method':<30} {'NAOPC':>10} {'Filler@' + str(k_display):>12} {'Mono':>10} {'Accuracy':>10}")
    print("-" * 74)
    for result in results:
        filler = result.filler_comprehensiveness.get(k_display, 0.0)
        print(
            f"{result.name:<30} {result.naopc:>10.4f} {filler:>12.4f} "
            f"{result.monotonicity:>10.4f} {result.accuracy:>10.4f}"
        )

    print(f"\n{'Method':<30} {'AOPC_lo':>10} {'AOPC_hi':>10} {'Adv Sens':>10} {'Mean Tau':>10}")
    print("-" * 72)
    for result in results:
        print(
            f"{result.name:<30} {result.naopc_lower:>10.4f} "
            f"{result.naopc_upper:>10.4f} {result.adversarial_sensitivity:>10.4f} "
            f"{result.adversarial_mean_tau:>10.4f}"
        )

    print(f"\n--- SUPPLEMENTARY ERASER METRICS ---")
    print("  Note: Comp/Suff can be inflated by OOD artifacts (arXiv:2308.14272).")
    print(
        f"\n{'Method':<30} {'Comp@' + str(k_display):>10} {'Suff@' + str(k_display):>10} "
        f"{'F-Comp@' + str(k_display):>12} {'F-Suff@' + str(k_display):>12} {'Time':>10}"
    )
    print("-" * 86)
    for result in results:
        comp = result.comprehensiveness.get(k_display, 0.0)
        suff = result.sufficiency.get(k_display, 0.0)
        f_comp = result.f_comprehensiveness.get(k_display, 0.0)
        f_suff = result.f_sufficiency.get(k_display, 0.0)
        time_str = f"{result.explanation_time:.1f}s"
        print(
            f"{result.name:<30} {comp:>10.4f} {suff:>10.4f} "
            f"{f_comp:>12.4f} {f_suff:>12.4f} {time_str:>10}"
        )

    print(f"\n--- SOFT PERTURBATION METRICS (arXiv:2305.10496) ---")
    print(f"  Note: Probabilistic masking avoids OOD artifacts of hard masking.")
    print(f"\n{'Method':<30} {'Soft Comp':>12} {'Soft Suff':>12}")
    print("-" * 56)
    for result in results:
        print(
            f"{result.name:<30} {result.soft_comprehensiveness:>12.4f} "
            f"{result.soft_sufficiency:>12.4f}"
        )

    if any(result.ffidelity_comp for result in results):
        print(f"\n--- F-FIDELITY METRICS (fine-tuned model, arXiv:2410.02970) ---")
        print(f"\n{'Method':<30} {'FF-Comp@' + str(k_display):>12} {'FF-Suff@' + str(k_display):>12}")
        print("-" * 56)
        for result in results:
            ff_comp = result.ffidelity_comp.get(k_display, 0.0)
            ff_suff = result.ffidelity_suff.get(k_display, 0.0)
            print(f"{result.name:<30} {ff_comp:>12.4f} {ff_suff:>12.4f}")

    print("\n" + "-" * 82)
    print("Interpretation:")
    print("  NAOPC: Normalized AOPC (0-1, higher = better, per-example normalized)")
    print("  Filler: Filler-token comprehensiveness (higher = better, OOD-robust)")
    print("  Monotonicity: Higher = better (consistent importance ordering)")
    print("  Soft Comp: Soft comprehensiveness (higher = better, probabilistic masking)")
    print("  Soft Suff: Soft sufficiency (lower = better, probabilistic retention)")
    print("  Comp/Suff: ERASER metrics (higher comp / lower suff = better)")
    print("  F-Comp/F-Suff: Beta-bounded variants (reduced OOD effects)")
    print("  FF-Comp/FF-Suff: F-Fidelity with fine-tuned model (proper OOD handling)")
    print("  Adv Sens: Adversarial sensitivity with multi-attack + tau-hat")

    print("\n" + "=" * 80)
    print("WINNER ANALYSIS")
    print("=" * 80)
    best_naopc = max(results, key=lambda result: result.naopc)
    best_filler = max(results, key=lambda result: result.filler_comprehensiveness.get(k_display, 0.0))
    best_mono = max(results, key=lambda result: result.monotonicity)
    best_comp = max(results, key=lambda result: result.comprehensiveness.get(k_display, 0.0))
    best_soft_comp = max(results, key=lambda result: result.soft_comprehensiveness)

    print(f"  Best NAOPC: {best_naopc.name}")
    print(f"  Best Filler Comp@{k_display}: {best_filler.name}")
    print(f"  Best Soft Comp: {best_soft_comp.name}")
    print(f"  Best Monotonicity: {best_mono.name}")
    print(f"  Best Comprehensiveness@{k_display}: {best_comp.name}")


def benchmark_explainer(
    clf,
    name: str,
    explain_fn,
    test_texts: list[str],
    config: BenchmarkConfig,
    mask_token: str,
    attacks: list | None = None,
    sampler: UnigramSampler | None = None,
    ftuned_clf=None,
) -> InterpretabilityResult:
    """Evaluate one explainer on the configured metric suite."""
    k_values = list(config.k_values)

    print(f"\nGenerating explanations for {name}...")
    start = time.time()
    attributions = [explain_fn(text, config.k_max) for text in tqdm(test_texts, desc="Explaining")]
    explanation_time = time.time() - start

    print("Computing metrics...")
    result = InterpretabilityResult(name=name, explanation_time=explanation_time)
    result.comprehensiveness = compute_comprehensiveness(clf, test_texts, attributions, k_values, mask_token)
    result.sufficiency = compute_sufficiency(clf, test_texts, attributions, k_values, mask_token)
    result.monotonicity = compute_monotonicity(clf, test_texts, attributions, config.monotonicity_steps, mask_token)
    aopc_scores = compute_comprehensiveness(
        clf,
        test_texts,
        attributions,
        list(range(1, config.k_max + 1)),
        mask_token,
    )
    result.aopc = float(numpy.mean(list(aopc_scores.values())))

    naopc_result = compute_normalized_aopc(
        clf,
        test_texts,
        attributions,
        k_max=config.k_max,
        beam_size=config.naopc_beam_size,
        mask_token=mask_token,
    )
    result.naopc = naopc_result["naopc"]
    result.naopc_lower = naopc_result["aopc_lower"]
    result.naopc_upper = naopc_result["aopc_upper"]

    result.f_comprehensiveness = compute_comprehensiveness(
        clf,
        test_texts,
        attributions,
        k_values,
        mask_token,
        beta=config.ffidelity_beta,
    )
    result.f_sufficiency = compute_sufficiency(
        clf,
        test_texts,
        attributions,
        k_values,
        mask_token,
        beta=config.ffidelity_beta,
    )

    if ftuned_clf is not None:
        result.ffidelity_comp = compute_comprehensiveness(
            ftuned_clf,
            test_texts,
            attributions,
            k_values,
            mask_token,
            beta=config.ffidelity_beta,
        )
        result.ffidelity_suff = compute_sufficiency(
            ftuned_clf,
            test_texts,
            attributions,
            k_values,
            mask_token,
            beta=config.ffidelity_beta,
        )

    if sampler is not None:
        result.filler_comprehensiveness = compute_filler_comprehensiveness(
            clf,
            test_texts,
            attributions,
            k_values,
            sampler,
        )

    result.soft_comprehensiveness = compute_soft_comprehensiveness(
        clf,
        test_texts,
        attributions,
        mask_token,
        n_samples=config.soft_metric_n_samples,
        seed=config.seed,
    )
    result.soft_sufficiency = compute_soft_sufficiency(
        clf,
        test_texts,
        attributions,
        mask_token,
        n_samples=config.soft_metric_n_samples,
        seed=config.seed,
    )

    adv_result = compute_adversarial_sensitivity(
        clf,
        explain_fn,
        test_texts[:config.adversarial_test_samples],
        attacks=attacks,
        max_changes=config.adversarial_max_changes,
        mcp_threshold=config.adversarial_mcp_threshold,
        top_k=config.k_max,
        seed=config.seed,
    )
    result.adversarial_sensitivity = adv_result["adversarial_sensitivity"]
    result.adversarial_mean_tau = adv_result["mean_tau"]

    return result
