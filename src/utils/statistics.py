"""
Statistical utilities for rigorous ML evaluation.

Provides:
- Bootstrap confidence intervals
- McNemar's test for paired classifier comparison
- Multi-seed result aggregation
- Statistical significance testing

References:
- NeurIPS Paper Checklist: https://neurips.cc/public/guides/PaperChecklist
- "Statistical Significance Tests for Comparing Machine Learning Algorithms"
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from scipy import stats


@dataclass
class StatisticalResult:
    """Container for statistical analysis results."""
    mean: float
    std: float
    ci_lower: float
    ci_upper: float
    n_samples: int
    confidence_level: float = 0.95


@dataclass
class McNemarResult:
    """Container for McNemar's test results."""
    statistic: float
    p_value: float
    n_discordant: int  # Total discordant pairs (b + c)
    model1_better: int  # Cases where model1 correct, model2 wrong
    model2_better: int  # Cases where model2 correct, model1 wrong
    significant: bool
    alpha: float = 0.05


def bootstrap_ci(
    values: np.ndarray,
    confidence_level: float = 0.95,
    n_bootstrap: int = 10000,
    random_state: Optional[int] = None,
) -> StatisticalResult:
    """
    Compute bootstrap confidence interval for a metric.

    Uses the percentile method which is robust and doesn't assume normality.
    Recommended for ML metrics where distributions may be non-normal.

    Args:
        values: Array of metric values from multiple runs
        confidence_level: Confidence level (default 0.95 for 95% CI)
        n_bootstrap: Number of bootstrap resamples
        random_state: Random seed for reproducibility

    Returns:
        StatisticalResult with mean, std, and CI bounds

    Example:
        >>> accuracies = np.array([0.901, 0.894, 0.908, 0.897, 0.903])
        >>> result = bootstrap_ci(accuracies)
        >>> print(result)
        0.9006 ± 0.0051 (95% CI: [0.8948, 0.9060])
    """
    values = np.asarray(values)
    n = len(values)

    if n < 2:
        raise ValueError("Need at least 2 values for bootstrap CI")

    rng = np.random.default_rng(random_state)

    # Generate bootstrap samples
    bootstrap_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(values, size=n, replace=True)
        bootstrap_means[i] = np.mean(sample)

    # Compute percentile CI
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

    return StatisticalResult(
        mean=float(np.mean(values)),
        std=float(np.std(values, ddof=1)),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        n_samples=n,
        confidence_level=confidence_level,
    )


def mcnemar_test(
    y_true: np.ndarray,
    pred1: np.ndarray,
    pred2: np.ndarray,
    alpha: float = 0.05,
    correction: bool = True,
) -> McNemarResult:
    """
    McNemar's test for comparing two classifiers on the same test set.

    Tests whether the disagreements between two classifiers are symmetric.
    Appropriate for paired classifier comparison (same test samples).

    Args:
        y_true: True labels
        pred1: Predictions from classifier 1
        pred2: Predictions from classifier 2
        alpha: Significance level
        correction: Whether to apply continuity correction (Edwards' correction)

    Returns:
        McNemarResult with test statistic and p-value

    Example:
        >>> y_true = np.array([0, 1, 1, 0, 1])
        >>> pred1 = np.array([0, 1, 0, 0, 1])  # 4/5 correct
        >>> pred2 = np.array([0, 0, 1, 0, 1])  # 4/5 correct
        >>> result = mcnemar_test(y_true, pred1, pred2)
        >>> print(result.significant)  # False - same accuracy, symmetric errors
    """
    y_true = np.asarray(y_true)
    pred1 = np.asarray(pred1)
    pred2 = np.asarray(pred2)

    # Compute contingency table
    correct1 = (pred1 == y_true)
    correct2 = (pred2 == y_true)

    # b: model1 correct, model2 wrong
    # c: model1 wrong, model2 correct
    b = np.sum(correct1 & ~correct2)
    c = np.sum(~correct1 & correct2)

    n_discordant = b + c

    if n_discordant == 0:
        # No disagreements - classifiers make identical predictions
        return McNemarResult(
            statistic=0.0,
            p_value=1.0,
            n_discordant=0,
            model1_better=int(b),
            model2_better=int(c),
            significant=False,
            alpha=alpha,
        )

    # McNemar's chi-squared statistic
    if correction:
        # Edwards' continuity correction
        statistic = (abs(b - c) - 1) ** 2 / (b + c)
    else:
        statistic = (b - c) ** 2 / (b + c)

    # Chi-squared distribution with 1 degree of freedom
    p_value = 1 - stats.chi2.cdf(statistic, df=1)

    return McNemarResult(
        statistic=float(statistic),
        p_value=float(p_value),
        n_discordant=int(n_discordant),
        model1_better=int(b),
        model2_better=int(c),
        significant=p_value < alpha,
        alpha=alpha,
    )


def paired_t_test(
    values1: np.ndarray,
    values2: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[float, float, bool]:
    """
    Paired t-test for comparing two models across multiple runs.

    Appropriate when comparing mean performance across seeds.
    Each pair (values1[i], values2[i]) should be from the same seed.

    Args:
        values1: Metric values from model 1 across seeds
        values2: Metric values from model 2 across seeds
        alpha: Significance level

    Returns:
        Tuple of (t_statistic, p_value, is_significant)
    """
    values1 = np.asarray(values1)
    values2 = np.asarray(values2)

    if len(values1) != len(values2):
        raise ValueError("Both arrays must have the same length")

    t_stat, p_value = stats.ttest_rel(values1, values2)

    return float(t_stat), float(p_value), p_value < alpha


def aggregate_results(
    results: List[Dict[str, float]],
    metrics: Optional[List[str]] = None,
    confidence_level: float = 0.95,
) -> Dict[str, StatisticalResult]:
    """
    Aggregate results from multiple runs with statistical analysis.

    Args:
        results: List of dicts, each containing metric values from one run
        metrics: List of metric names to aggregate (default: all keys in first result)
        confidence_level: Confidence level for CIs

    Returns:
        Dict mapping metric name to StatisticalResult

    Example:
        >>> results = [
        ...     {"accuracy": 0.901, "f1": 0.899},
        ...     {"accuracy": 0.894, "f1": 0.891},
        ...     {"accuracy": 0.908, "f1": 0.906},
        ... ]
        >>> stats = aggregate_results(results)
        >>> print(stats["accuracy"])
    """
    if not results:
        raise ValueError("Results list cannot be empty")

    if metrics is None:
        metrics = list(results[0].keys())

    aggregated = {}
    for metric in metrics:
        values = np.array([r[metric] for r in results if metric in r])
        if len(values) >= 2:
            aggregated[metric] = bootstrap_ci(values, confidence_level=confidence_level)
        elif len(values) == 1:
            # Single run - no CI possible
            aggregated[metric] = StatisticalResult(
                mean=float(values[0]),
                std=0.0,
                ci_lower=float(values[0]),
                ci_upper=float(values[0]),
                n_samples=1,
                confidence_level=confidence_level,
            )

    return aggregated


def effect_size_cohens_d(
    values1: np.ndarray,
    values2: np.ndarray,
) -> float:
    """
    Compute Cohen's d effect size for comparing two groups.

    Interpretation:
    - |d| < 0.2: negligible
    - 0.2 <= |d| < 0.5: small
    - 0.5 <= |d| < 0.8: medium
    - |d| >= 0.8: large

    Args:
        values1: Values from group 1
        values2: Values from group 2

    Returns:
        Cohen's d effect size
    """
    values1 = np.asarray(values1)
    values2 = np.asarray(values2)

    n1, n2 = len(values1), len(values2)
    var1, var2 = np.var(values1, ddof=1), np.var(values2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return float((np.mean(values1) - np.mean(values2)) / pooled_std)
