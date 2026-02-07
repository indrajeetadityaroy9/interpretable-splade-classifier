"""CLI entry point for the canonical interpretability benchmark."""

import argparse

from src.baselines.splade_adapters import (
    SPLADEAttentionExplainer,
    SPLADEIntegratedGradientsExplainer,
    SPLADELIMEExplainer,
)
from src.data.loader import load_sst2_data
from src.evaluation.adversarial import CharacterAttack, TextFoolerAttack, WordNetAttack
from src.evaluation.benchmark import BenchmarkConfig, benchmark_explainer, print_interpretability_results
from src.evaluation.faithfulness import UnigramSampler
from src.models.classifier import SPLADEClassifier
from src.training.finetune import finetune_splade_for_ffidelity
from src.utils.cuda import set_seed


def run_benchmark(
    train_samples: int,
    test_samples: int,
    epochs: int,
    batch_size: int,
    seed: int,
) -> list:
    """Run single-seed benchmark on SST-2."""
    config = BenchmarkConfig(seed=seed)
    set_seed(config.seed)

    train_texts, train_labels, test_texts, test_labels, num_labels = load_sst2_data(
        train_samples,
        test_samples,
        seed=config.seed,
    )

    clf = SPLADEClassifier(num_labels=num_labels, batch_size=batch_size)
    clf.fit(train_texts, train_labels, epochs=epochs)
    accuracy = clf.score(test_texts, test_labels)
    mask_token = clf.tokenizer.mask_token

    sampler = UnigramSampler(test_texts, seed=config.seed)
    print("\nFine-tuning model copy for F-Fidelity...")
    fine_tuned_clf = finetune_splade_for_ffidelity(
        clf,
        train_texts,
        train_labels,
        beta=config.ffidelity_beta,
        ft_epochs=config.ffidelity_ft_epochs,
        ft_lr=config.ffidelity_ft_lr,
        batch_size=config.ffidelity_ft_batch_size,
        mask_token=mask_token,
        seed=config.seed,
    )

    attacks = [
        WordNetAttack(max_changes=config.adversarial_max_changes),
        TextFoolerAttack(clf, max_changes=config.adversarial_max_changes),
        CharacterAttack(max_changes=config.adversarial_max_changes),
    ]

    results = [
        benchmark_explainer(
            clf,
            "SPLADE",
            lambda text, top_k: clf.explain(text, top_k=top_k),
            test_texts,
            config,
            mask_token,
            attacks=attacks,
            sampler=sampler,
            ftuned_clf=fine_tuned_clf,
        )
    ]
    results[0].accuracy = accuracy

    baselines = [
        (SPLADEAttentionExplainer(clf), "Attention"),
        (SPLADELIMEExplainer(clf, num_samples=config.lime_num_samples), "LIME"),
        (SPLADEIntegratedGradientsExplainer(clf, n_steps=config.ig_n_steps), "IntGrad"),
    ]
    for explainer, name in baselines:
        result = benchmark_explainer(
            clf,
            name,
            lambda text, top_k, baseline=explainer: baseline.explain(text, top_k=top_k),
            test_texts,
            config,
            mask_token,
            attacks=attacks,
            sampler=sampler,
            ftuned_clf=fine_tuned_clf,
        )
        result.accuracy = accuracy
        results.append(result)

    print_interpretability_results(results, config)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run canonical interpretability benchmark on SST-2")
    parser.add_argument("--train-samples", type=int, default=2000)
    parser.add_argument("--test-samples", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_benchmark(
        train_samples=args.train_samples,
        test_samples=args.test_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
