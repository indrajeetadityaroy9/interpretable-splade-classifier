"""B-cos ablation: compare ReLU classifier vs B-cos classifier.

Demonstrates CIS generalizes beyond 2-layer ReLU. B-cos provides
exact W_eff (DLA) for arbitrary depth classifiers.

Variants:
  1. ReLU 2-layer (default SpladeModel)
  2. B-cos 2-layer
  3. B-cos 3-layer
"""

import argparse
import json
import os

from transformers import AutoTokenizer

from splade.config.load import load_config
from splade.config.schema import Config
from splade.data.loader import infer_max_length, load_dataset_by_name
from splade.evaluation.mechanistic import run_mechanistic_evaluation
from splade.inference import score_model
from splade.models.splade import SpladeModel
from splade.models.splade_bcos import SpladeBcosModel
from splade.pipelines import prepare_mechanistic_inputs
from splade.training.loop import train_model
from splade.training.optim import _infer_batch_size
from splade.utils.cuda import DEVICE, set_seed


def _train_and_evaluate(config: Config, seed: int, model, tokenizer,
                        train_texts, train_labels, test_texts, test_labels,
                        num_labels, max_length, batch_size) -> dict:
    """Train a model and run mechanistic evaluation."""
    val_size = min(200, len(train_texts) // 5)
    val_texts = train_texts[-val_size:]
    val_labels = train_labels[-val_size:]
    train_texts_split = train_texts[:-val_size]
    train_labels_split = train_labels[:-val_size]

    centroid_tracker = train_model(
        model, tokenizer, train_texts_split, train_labels_split,
        model_name=config.model.name, num_labels=num_labels,
        val_texts=val_texts, val_labels=val_labels,
        max_length=max_length, batch_size=batch_size,
    )

    accuracy = score_model(
        model, tokenizer, test_texts, test_labels,
        max_length, batch_size, num_labels,
    )

    input_ids_list, attention_mask_list = prepare_mechanistic_inputs(
        tokenizer, test_texts, max_length,
    )

    results = run_mechanistic_evaluation(
        model, input_ids_list, attention_mask_list,
        test_labels, tokenizer, num_classes=num_labels,
        centroid_tracker=centroid_tracker,
    )

    completeness_vals = list(results.circuit_completeness.values())
    sf = results.semantic_fidelity

    return {
        "accuracy": accuracy,
        "dla_error": results.dla_verification_error,
        "completeness_mean": (
            sum(completeness_vals) / len(completeness_vals)
            if completeness_vals else 0.0
        ),
        "cosine_separation": sf.get("cosine_separation"),
        "jaccard_separation": sf.get("class_separation", 0.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="B-cos vs ReLU ablation")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config.evaluation.seeds[0]

    train_texts, train_labels, test_texts, test_labels, num_labels = load_dataset_by_name(
        config.data.dataset_name,
        config.data.train_samples,
        config.data.test_samples,
        seed=seed,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model.name)
    max_length = infer_max_length(train_texts, tokenizer)
    batch_size = _infer_batch_size(config.model.name, max_length)

    variants = {
        "ReLU 2-layer": lambda: SpladeModel(config.model.name, num_labels).to(DEVICE),
        "B-cos 2-layer": lambda: SpladeBcosModel(
            config.model.name, num_labels, num_bcos_layers=2,
        ).to(DEVICE),
        "B-cos 3-layer": lambda: SpladeBcosModel(
            config.model.name, num_labels, num_bcos_layers=3,
        ).to(DEVICE),
    }

    all_results = {}
    for name, model_fn in variants.items():
        print(f"\n{'='*60}")
        print(f"Variant: {name}")
        print(f"{'='*60}")

        set_seed(seed)
        model = model_fn()

        metrics = _train_and_evaluate(
            config, seed, model, tokenizer,
            train_texts, train_labels, test_texts, test_labels,
            num_labels, max_length, batch_size,
        )
        all_results[name] = metrics

        print(f"\n  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  DLA Error: {metrics['dla_error']:.6f}")
        print(f"  Completeness: {metrics['completeness_mean']:.4f}")
        print(f"  Jaccard Sep: {metrics['jaccard_separation']:.4f}")

    # Comparison table
    print(f"\n{'='*80}")
    print("B-COS ABLATION COMPARISON")
    print(f"{'='*80}")
    header = f"{'Variant':<20} {'Accuracy':>10} {'DLA Err':>10} {'Complete':>10} {'Cos Sep':>10} {'Jac Sep':>10}"
    print(header)
    print("-" * 80)
    for name, m in all_results.items():
        cos_str = f"{m['cosine_separation']:>10.4f}" if m["cosine_separation"] is not None else f"{'N/A':>10}"
        print(
            f"{name:<20} {m['accuracy']:>10.4f} {m['dla_error']:>10.6f} "
            f"{m['completeness_mean']:>10.4f} {cos_str} {m['jaccard_separation']:>10.4f}"
        )
    print(f"{'='*80}")

    os.makedirs(config.output_dir, exist_ok=True)
    output_path = os.path.join(config.output_dir, "bcos_ablation_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
