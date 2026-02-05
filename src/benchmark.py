"""SPLADE classifier performance benchmark."""

import argparse
import time
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)

from src.data import load_benchmark_data
from src.models import SPLADEClassifier


@dataclass
class BenchmarkMetrics:
    """Classification benchmark results."""
    accuracy: float = 0.0
    precision_macro: float = 0.0
    recall_macro: float = 0.0
    f1_macro: float = 0.0
    f1_weighted: float = 0.0
    train_time: float = 0.0
    inference_time: float = 0.0
    samples_per_second: float = 0.0
    f1_per_class: list[float] = field(default_factory=list)
    confusion_matrix: list[list[int]] = field(default_factory=list)
    roc_auc: float = 0.0


def compute_classification_metrics(
    y_true: list[int], y_pred: list[int],
    y_proba: list[list[float]], num_labels: int,
) -> BenchmarkMetrics:
    """Compute standard classification metrics."""
    metrics = BenchmarkMetrics()
    metrics.accuracy = accuracy_score(y_true, y_pred)
    metrics.precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    metrics.recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    metrics.f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    metrics.f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    metrics.f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()
    metrics.confusion_matrix = confusion_matrix(y_true, y_pred).tolist()

    if num_labels == 2:
        metrics.roc_auc = roc_auc_score(y_true, [p[1] for p in y_proba])
    else:
        metrics.roc_auc = roc_auc_score(y_true, np.array(y_proba), multi_class="ovr")

    return metrics


def print_benchmark_metrics(metrics: BenchmarkMetrics, name: str) -> None:
    """Display benchmark results in a formatted table."""
    print(f"\n{name} Results:")
    print("-" * 40)
    print(f"  Accuracy:           {metrics.accuracy * 100:.2f}%")
    print(f"  F1 (macro):         {metrics.f1_macro * 100:.2f}%")
    print(f"  F1 (weighted):      {metrics.f1_weighted * 100:.2f}%")
    print(f"  Precision (macro):  {metrics.precision_macro * 100:.2f}%")
    print(f"  Recall (macro):     {metrics.recall_macro * 100:.2f}%")
    print(f"  ROC-AUC:            {metrics.roc_auc * 100:.2f}%")
    print(f"\n  Train time:         {metrics.train_time:.1f}s")
    print(f"  Inference time:     {metrics.inference_time:.2f}s")
    print(f"  Throughput:         {metrics.samples_per_second:.0f} samples/s")
    print(f"\n  Per-class F1:       {[f'{f * 100:.1f}%' for f in metrics.f1_per_class]}")
    print("  Confusion Matrix:")
    for row in metrics.confusion_matrix:
        print(f"    {row}")


def run_benchmarks(
    datasets: list[str],
    train_samples: int, test_samples: int, epochs: int, batch_size: int,
) -> dict[str, BenchmarkMetrics]:
    """Execute benchmarks across specified datasets."""
    all_results = {}

    for dataset in datasets:
        print(f"\n{'=' * 60}\nSPLADEClassifier - {dataset.upper()}\n{'=' * 60}")
        train_texts, train_labels, test_texts, test_labels, num_labels = load_benchmark_data(dataset, train_samples, test_samples)
        
        clf = SPLADEClassifier(num_labels=num_labels, batch_size=batch_size)
        
        train_start = time.time()
        clf.fit(train_texts, train_labels, epochs=epochs)
        train_time = time.time() - train_start

        for _ in range(3): # Warmup
            clf.predict(test_texts[: min(100, len(test_texts))])

        infer_start = time.time()
        y_proba = clf.predict_proba(test_texts)
        y_pred = [np.argmax(p) for p in y_proba]
        infer_time = time.time() - infer_start

        metrics = compute_classification_metrics(test_labels, y_pred, y_proba, num_labels)
        metrics.train_time, metrics.inference_time = train_time, infer_time
        metrics.samples_per_second = len(test_texts) / infer_time

        print_benchmark_metrics(metrics, "SPLADEClassifier")
        all_results[dataset] = metrics

    if len(datasets) > 1:
        print(f"\n{'=' * 60}\nFINAL SUMMARY\n{'=' * 60}")
        print(f"\n{'Dataset':<12} {'Accuracy':>12} {'F1 (macro)':>12} {'Throughput':>12}")
        print("-" * 50)
        for dataset in datasets:
            m = all_results[dataset]
            print(f"{dataset:<12} {m.accuracy * 100:>11.1f}% {m.f1_macro * 100:>11.1f}% {m.samples_per_second:>10.0f}/s")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Benchmark SPLADE classifier")
    parser.add_argument("--dataset", type=str, default="all", choices=["ag_news", "sst2", "imdb", "all"])
    parser.add_argument("--train-samples", type=int, default=10000)
    parser.add_argument("--test-samples", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    run_benchmarks(
        datasets=["ag_news", "sst2", "imdb"] if args.dataset == "all" else [args.dataset],
        train_samples=args.train_samples, test_samples=args.test_samples,
        epochs=args.epochs, batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
