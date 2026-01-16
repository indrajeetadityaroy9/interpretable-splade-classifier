"""
Benchmark comparing sklearn TF-IDF vs Neural SPLADE classifier.

Supports multi-seed experiments with statistical significance testing.

Usage:
    # Single run (legacy)
    python -m src.benchmark --dataset ag_news --epochs 3

    # Multi-seed with statistical analysis (recommended)
    python -m src.benchmark --dataset ag_news --epochs 3 --seeds 5

    # Save results to JSON
    python -m src.benchmark --dataset ag_news --seeds 5 --output results/benchmark.json

Both implementations use sklearn-compatible APIs:

    # Sklearn
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(texts)
    clf = LogisticRegression().fit(X_train, y_train)
    preds = clf.predict(X_test)

    # SPLADE (equally simple!)
    clf = SPLADEClassifier()
    clf.fit(train_texts, train_labels)
    preds = clf.predict(test_texts)
"""

import time
import argparse
import json
import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from src.models import SPLADEClassifier
from src.data import load_classification_data
from src.utils import (
    set_seed,
    load_stopwords,
    simple_tokenizer,
    validate_data_sources,
    bootstrap_ci,
    mcnemar_test,
    paired_t_test,
    aggregate_results,
    effect_size_cohens_d,
)


# Default seeds for reproducible multi-run experiments
DEFAULT_SEEDS = [42, 123, 456, 789, 999]


@dataclass
class SingleRunResult:
    """Results from a single training run."""
    model: str
    seed: int
    accuracy: float
    f1: float
    sparsity: float
    train_time_s: float
    inference_ms: float
    predictions: Optional[List[int]] = None  # For McNemar's test


def train_sklearn_baseline(
    train_texts: List[str],
    train_labels: List[int],
    test_texts: List[str],
    test_labels: List[int],
    num_classes: int = 2,
    seed: int = 42,
    verbose: bool = True,
) -> SingleRunResult:
    """Train and evaluate sklearn TF-IDF + Logistic Regression baseline."""
    set_seed(seed)

    if verbose:
        print(f"\n[Sklearn TF-IDF] Seed={seed}")

    start_time = time.time()

    # 1. Vectorize
    stopwords = load_stopwords()
    tokenizer_func = lambda text: simple_tokenizer(text, stopwords)
    vectorizer = TfidfVectorizer(
        max_features=5000,
        tokenizer=tokenizer_func,
        token_pattern=None
    )

    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    # 2. Train Classifier
    if num_classes > 2:
        clf = LogisticRegression(max_iter=1000, multi_class='multinomial', random_state=seed)
    else:
        clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(X_train, train_labels)

    train_time = time.time() - start_time

    # 3. Evaluate
    start_inf = time.time()
    preds = clf.predict(X_test)
    inf_time = time.time() - start_inf

    acc = accuracy_score(test_labels, preds)
    if num_classes > 2:
        f1 = f1_score(test_labels, preds, average='macro')
    else:
        f1 = f1_score(test_labels, preds)
    sparsity = 100.0 * (1.0 - X_test.nnz / (X_test.shape[0] * X_test.shape[1]))

    if verbose:
        print(f"  Accuracy: {acc:.4f}, F1: {f1:.4f}, Sparsity: {sparsity:.2f}%")

    return SingleRunResult(
        model="Sklearn TF-IDF",
        seed=seed,
        accuracy=acc,
        f1=f1,
        sparsity=sparsity,
        train_time_s=train_time,
        inference_ms=inf_time * 1000,
        predictions=preds.tolist(),
    )


def train_splade_classifier(
    train_texts: List[str],
    train_labels: List[int],
    test_texts: List[str],
    test_labels: List[int],
    num_labels: int = 1,
    class_names: Optional[List[str]] = None,
    epochs: int = 5,
    seed: int = 42,
    verbose: bool = True,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    flops_lambda: float = 1e-4,
) -> SingleRunResult:
    """Train and evaluate Neural SPLADE classifier."""
    set_seed(seed)

    if verbose:
        print(f"\n[Neural SPLADE] Seed={seed}")

    # Create classifier with sklearn-like API
    clf = SPLADEClassifier(
        num_labels=num_labels,
        class_names=class_names,
        batch_size=batch_size,
        learning_rate=learning_rate,
        flops_lambda=flops_lambda,
        random_state=seed,
        verbose=verbose,
    )

    start_time = time.time()
    clf.fit(train_texts, train_labels, epochs=epochs)
    train_time = time.time() - start_time

    start_inf = time.time()
    preds = clf.predict(test_texts)
    inf_time = time.time() - start_inf

    acc = accuracy_score(test_labels, preds)
    if num_labels > 1:
        f1 = f1_score(test_labels, preds, average='macro')
    else:
        f1 = f1_score(test_labels, preds)
    sparsity = clf.get_sparsity(test_texts)

    if verbose:
        print(f"  Accuracy: {acc:.4f}, F1: {f1:.4f}, Sparsity: {sparsity:.2f}%")

    return SingleRunResult(
        model="Neural SPLADE",
        seed=seed,
        accuracy=acc,
        f1=f1,
        sparsity=sparsity,
        train_time_s=train_time,
        inference_ms=inf_time * 1000,
        predictions=preds if isinstance(preds, list) else preds.tolist(),
    )


def run_multi_seed_benchmark(
    train_texts: List[str],
    train_labels: List[int],
    test_texts: List[str],
    test_labels: List[int],
    num_classes: int,
    num_labels: int,
    class_names: Optional[List[str]],
    seeds: List[int],
    epochs: int = 5,
    skip_sklearn: bool = False,
    skip_splade: bool = False,
    verbose: bool = True,
    **splade_kwargs,
) -> Dict:
    """
    Run benchmark with multiple seeds and compute statistical analysis.

    Returns:
        Dict containing:
        - 'sklearn_results': List of SingleRunResult
        - 'splade_results': List of SingleRunResult
        - 'sklearn_stats': Aggregated statistics with CIs
        - 'splade_stats': Aggregated statistics with CIs
        - 'comparison': Statistical comparison (McNemar's, paired t-test)
    """
    sklearn_results = []
    splade_results = []

    print(f"\n{'='*60}")
    print(f"MULTI-SEED BENCHMARK ({len(seeds)} seeds)")
    print(f"Seeds: {seeds}")
    print(f"{'='*60}")

    for i, seed in enumerate(seeds):
        print(f"\n--- Run {i+1}/{len(seeds)} (seed={seed}) ---")

        if not skip_sklearn:
            result = train_sklearn_baseline(
                train_texts, train_labels,
                test_texts, test_labels,
                num_classes=num_classes,
                seed=seed,
                verbose=verbose,
            )
            sklearn_results.append(result)

        if not skip_splade:
            result = train_splade_classifier(
                train_texts, train_labels,
                test_texts, test_labels,
                num_labels=num_labels,
                class_names=class_names,
                epochs=epochs,
                seed=seed,
                verbose=verbose,
                **splade_kwargs,
            )
            splade_results.append(result)

    output = {
        "seeds": seeds,
        "n_seeds": len(seeds),
        "sklearn_results": [asdict(r) for r in sklearn_results],
        "splade_results": [asdict(r) for r in splade_results],
    }

    metrics = ["accuracy", "f1", "sparsity", "train_time_s", "inference_ms"]

    if sklearn_results:
        sklearn_dicts = [asdict(r) for r in sklearn_results]
        # Remove predictions from stats (large)
        for d in sklearn_dicts:
            d.pop("predictions", None)
        output["sklearn_stats"] = {
            k: asdict(v) for k, v in
            aggregate_results(sklearn_dicts, metrics=metrics).items()
        }

    if splade_results:
        splade_dicts = [asdict(r) for r in splade_results]
        for d in splade_dicts:
            d.pop("predictions", None)
        output["splade_stats"] = {
            k: asdict(v) for k, v in
            aggregate_results(splade_dicts, metrics=metrics).items()
        }

    # Statistical comparison
    if sklearn_results and splade_results and len(seeds) >= 2:
        sklearn_accs = np.array([r.accuracy for r in sklearn_results])
        splade_accs = np.array([r.accuracy for r in splade_results])

        # Paired t-test across seeds
        t_stat, p_value, significant = paired_t_test(splade_accs, sklearn_accs)

        # Effect size
        cohens_d = effect_size_cohens_d(splade_accs, sklearn_accs)

        output["comparison"] = {
            "paired_t_test": {
                "t_statistic": t_stat,
                "p_value": p_value,
                "significant_at_0.05": significant,
            },
            "effect_size": {
                "cohens_d": cohens_d,
                "interpretation": (
                    "large" if abs(cohens_d) >= 0.8 else
                    "medium" if abs(cohens_d) >= 0.5 else
                    "small" if abs(cohens_d) >= 0.2 else
                    "negligible"
                ),
            },
            "accuracy_difference": {
                "mean": float(np.mean(splade_accs) - np.mean(sklearn_accs)),
                "favors": "SPLADE" if np.mean(splade_accs) > np.mean(sklearn_accs) else "Sklearn",
            },
        }

        # McNemar's test on final seed (using saved predictions)
        if sklearn_results[-1].predictions and splade_results[-1].predictions:
            mcnemar_result = mcnemar_test(
                np.array(test_labels),
                np.array(sklearn_results[-1].predictions),
                np.array(splade_results[-1].predictions),
            )
            output["comparison"]["mcnemar_test"] = {
                "statistic": mcnemar_result.statistic,
                "p_value": mcnemar_result.p_value,
                "significant_at_0.05": mcnemar_result.significant,
                "sklearn_better_cases": mcnemar_result.model1_better,
                "splade_better_cases": mcnemar_result.model2_better,
            }

    return output


def print_statistical_summary(results: Dict):
    """Print formatted statistical summary."""
    print("\n" + "="*60)
    print("STATISTICAL SUMMARY")
    print("="*60)

    n_seeds = results.get("n_seeds", 1)
    print(f"\nNumber of runs: {n_seeds}")

    # Format results table
    table_data = []

    if "sklearn_stats" in results:
        stats = results["sklearn_stats"]
        table_data.append({
            "Model": "Sklearn TF-IDF",
            "Accuracy": f"{stats['accuracy']['mean']:.4f} ± {stats['accuracy']['std']:.4f}",
            "F1": f"{stats['f1']['mean']:.4f} ± {stats['f1']['std']:.4f}",
            "95% CI (Acc)": f"[{stats['accuracy']['ci_lower']:.4f}, {stats['accuracy']['ci_upper']:.4f}]",
        })

    if "splade_stats" in results:
        stats = results["splade_stats"]
        table_data.append({
            "Model": "Neural SPLADE",
            "Accuracy": f"{stats['accuracy']['mean']:.4f} ± {stats['accuracy']['std']:.4f}",
            "F1": f"{stats['f1']['mean']:.4f} ± {stats['f1']['std']:.4f}",
            "95% CI (Acc)": f"[{stats['accuracy']['ci_lower']:.4f}, {stats['accuracy']['ci_upper']:.4f}]",
        })

    if table_data:
        df = pd.DataFrame(table_data)
        print("\n" + df.to_string(index=False))

    # Statistical comparison
    if "comparison" in results:
        comp = results["comparison"]
        print("\n--- Statistical Comparison ---")

        if "accuracy_difference" in comp:
            diff = comp["accuracy_difference"]
            print(f"Mean accuracy difference: {diff['mean']:+.4f} (favors {diff['favors']})")

        if "paired_t_test" in comp:
            tt = comp["paired_t_test"]
            sig = "YES" if tt["significant_at_0.05"] else "NO"
            print(f"Paired t-test: t={tt['t_statistic']:.3f}, p={tt['p_value']:.4f} (significant: {sig})")

        if "effect_size" in comp:
            es = comp["effect_size"]
            print(f"Effect size (Cohen's d): {es['cohens_d']:.3f} ({es['interpretation']})")

        if "mcnemar_test" in comp:
            mc = comp["mcnemar_test"]
            sig = "YES" if mc["significant_at_0.05"] else "NO"
            print(f"McNemar's test: χ²={mc['statistic']:.3f}, p={mc['p_value']:.4f} (significant: {sig})")
            print(f"  Sklearn better on {mc['sklearn_better_cases']} cases, SPLADE better on {mc['splade_better_cases']} cases")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark sklearn TF-IDF vs Neural SPLADE classifier with statistical analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single run benchmark
  python -m src.benchmark --dataset ag_news --epochs 3

  # Multi-seed benchmark with 5 runs (recommended)
  python -m src.benchmark --dataset ag_news --epochs 3 --seeds 5

  # Custom seeds
  python -m src.benchmark --dataset ag_news --seeds 42,123,456

  # Save results to JSON
  python -m src.benchmark --dataset ag_news --seeds 5 --output results/benchmark.json
        """
    )

    # Data source options
    data_group = parser.add_argument_group('Data Source (choose one)')
    data_group.add_argument('--train_path', type=str, default=None,
                           help='Path to training CSV/TSV file')
    data_group.add_argument('--test_path', type=str, default=None,
                           help='Path to test CSV/TSV file')
    data_group.add_argument('--dataset', type=str, default=None,
                           help='HuggingFace dataset name (e.g., imdb, ag_news)')

    # Statistical options
    stat_group = parser.add_argument_group('Statistical Analysis')
    stat_group.add_argument('--seeds', type=str, default="1",
                           help='Number of seeds (int) or comma-separated seed list (e.g., "5" or "42,123,456")')
    stat_group.add_argument('--output', type=str, default=None,
                           help='Path to save results as JSON')

    # Training options
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Limit samples per split')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to save/load SPLADE model (disabled in multi-seed mode)')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Training epochs for SPLADE')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for SPLADE')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Learning rate for SPLADE')
    parser.add_argument('--flops_lambda', type=float, default=1e-4,
                       help='FLOPS regularization weight')
    parser.add_argument('--skip_sklearn', action='store_true',
                       help='Skip sklearn baseline')
    parser.add_argument('--skip_splade', action='store_true',
                       help='Skip SPLADE classifier')
    parser.add_argument('--quiet', action='store_true',
                       help='Reduce output verbosity')

    args = parser.parse_args()

    # Parse seeds
    if ',' in args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(',')]
    else:
        n_seeds = int(args.seeds)
        seeds = DEFAULT_SEEDS[:n_seeds] if n_seeds <= len(DEFAULT_SEEDS) else DEFAULT_SEEDS

    # Validate inputs
    has_local, has_hf = validate_data_sources(
        args.train_path, args.test_path, args.dataset,
        raise_on_error=False, print_error=True
    )
    if not has_local and not has_hf:
        return

    # Load data
    print("Loading data...")
    if has_local:
        print(f"  Source: Local files")
        print(f"  Train: {args.train_path}")
        print(f"  Test: {args.test_path}")
        train_texts, train_labels, train_meta = load_classification_data(
            file_path=args.train_path,
            max_samples=args.max_samples,
        )
        test_texts, test_labels, test_meta = load_classification_data(
            file_path=args.test_path,
            max_samples=args.max_samples,
        )
        num_classes = max(train_meta['num_labels'], test_meta['num_labels'])
        class_names = None
    else:
        print(f"  Source: HuggingFace dataset '{args.dataset}'")
        train_texts, train_labels, train_meta = load_classification_data(
            dataset=args.dataset,
            split="train",
            max_samples=args.max_samples,
        )
        test_texts, test_labels, test_meta = load_classification_data(
            dataset=args.dataset,
            split="test",
            max_samples=args.max_samples,
        )
        num_classes = train_meta['num_labels']
        class_names = train_meta.get('class_names')

    # For SPLADE: use num_labels=1 for binary (BCE), num_labels=N for multi-class
    num_labels = 1 if num_classes == 2 else num_classes

    print(f"  Train samples: {len(train_texts)}")
    print(f"  Test samples: {len(test_texts)}")
    print(f"  Num classes: {num_classes}")
    if class_names:
        print(f"  Classes: {class_names}")

    # Run benchmark
    results = run_multi_seed_benchmark(
        train_texts=train_texts,
        train_labels=train_labels,
        test_texts=test_texts,
        test_labels=test_labels,
        num_classes=num_classes,
        num_labels=num_labels,
        class_names=class_names,
        seeds=seeds,
        epochs=args.epochs,
        skip_sklearn=args.skip_sklearn,
        skip_splade=args.skip_splade,
        verbose=not args.quiet,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        flops_lambda=args.flops_lambda,
    )

    results["config"] = {
        "dataset": args.dataset or "local",
        "train_samples": len(train_texts),
        "test_samples": len(test_texts),
        "num_classes": num_classes,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "flops_lambda": args.flops_lambda,
    }

    print_statistical_summary(results)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        # Remove predictions from saved results (too large)
        save_results = results.copy()
        for key in ["sklearn_results", "splade_results"]:
            if key in save_results:
                for r in save_results[key]:
                    r.pop("predictions", None)
        with open(args.output, 'w') as f:
            json.dump(save_results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
