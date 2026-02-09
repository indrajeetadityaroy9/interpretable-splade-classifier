"""Experiment A: Faithfulness Stress Test.

Trains Lexical-SAE and compares removability of DLA vs baseline explainers
(Gradient x Input, Integrated Gradients, Attention). DLA should achieve
near-100% flip rate due to exact algebraic attribution.
"""

import argparse
import json
import os
from dataclasses import asdict

import yaml

from splade.config.load import load_config
from splade.evaluation.faithfulness import compare_with_baseline_explainer
from splade.pipelines import setup_and_train


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment A: Faithfulness Stress Test")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    config = load_config(args.config)
    os.makedirs(config.output_dir, exist_ok=True)

    with open(os.path.join(config.output_dir, "resolved_config.yaml"), "w") as f:
        yaml.dump(asdict(config), f)

    all_results = []
    for seed in config.evaluation.seeds:
        print(f"\n{'=' * 60}")
        print(f"FAITHFULNESS EXPERIMENT — SEED {seed}")
        print(f"{'=' * 60}")

        # Train
        print("\n--- Training Lexical-SAE ---")
        exp = setup_and_train(config, seed)
        print(f"Test accuracy: {exp.accuracy:.4f}")

        # Removability comparison
        print("\n--- Removability Comparison ---")
        comparison = compare_with_baseline_explainer(
            exp.model, exp.tokenizer,
            exp.test_texts, exp.test_labels,
            exp.max_length, top_k=5, batch_size=exp.batch_size,
        )

        # Print table
        print(f"\n{'Explainer':<25} {'Flip Rate':>10} {'Prob Drop':>10} {'Time (s)':>10}")
        print("-" * 55)
        for name, metrics in comparison.items():
            print(f"{name:<25} {metrics['flip_rate']:>10.3f} {metrics['mean_prob_drop']:>10.3f} {metrics['time_seconds']:>10.2f}")

        all_results.append({
            "seed": seed,
            "accuracy": exp.accuracy,
            "comparison": comparison,
        })

    output_path = os.path.join(config.output_dir, "faithfulness_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
