"""CIS experiment pipeline: Train -> Mechanistic Evaluation.

Two-phase pipeline:
  Phase 1: CIS Training (DF-FLOPS + circuit losses + GECO)
  Phase 2: Mechanistic Evaluation (DLA verification, circuit extraction, completeness, separation)
"""

import argparse
import json
import os
from dataclasses import asdict

import yaml

from splade.config.load import load_config
from splade.config.schema import Config
from splade.evaluation.mechanistic import (print_mechanistic_results,
                                           run_mechanistic_evaluation)
from splade.pipelines import prepare_mechanistic_inputs, setup_and_train


def _print_cis_config(config: Config) -> None:
    """Print CIS experiment configuration."""
    train_str = "full" if config.data.train_samples <= 0 else str(config.data.train_samples)
    test_str = "full" if config.data.test_samples <= 0 else str(config.data.test_samples)
    print("\n--- Circuit-Integrated SPLADE (CIS) ---")
    print(f"  Model:       {config.model.name}")
    print(f"  Dataset:     {config.data.dataset_name} (train={train_str}, test={test_str})")
    print(f"  Seeds:       {config.evaluation.seeds}")
    print(f"  SAE compare: {'yes' if config.mechanistic.sae_comparison else 'no'}")
    print()


def run_experiment(config: Config) -> list[dict]:
    """Run the full CIS experiment pipeline across all seeds."""
    all_results = []

    _print_cis_config(config)

    os.makedirs(config.output_dir, exist_ok=True)
    with open(os.path.join(config.output_dir, "resolved_config.yaml"), "w") as f:
        yaml.dump(asdict(config), f)

    for seed in config.evaluation.seeds:
        print(f"\n{'#' * 60}")
        print(f"EXPERIMENT SEED {seed}")
        print(f"{'#' * 60}")

        # Phase 1: CIS Training
        print("\n--- PHASE 1: CIS TRAINING ---")
        exp = setup_and_train(config, seed)
        print(f"Accuracy: {exp.accuracy:.4f}")

        # Phase 2: Mechanistic Evaluation
        print("\n--- PHASE 2: MECHANISTIC EVALUATION ---")

        input_ids_list, attention_mask_list = prepare_mechanistic_inputs(
            exp.tokenizer, exp.test_texts, exp.max_length,
        )

        mechanistic_results = run_mechanistic_evaluation(
            exp.model, input_ids_list, attention_mask_list,
            exp.test_labels, exp.tokenizer, num_classes=exp.num_labels,
            circuit_fraction=config.mechanistic.circuit_fraction,
            run_sae_comparison=config.mechanistic.sae_comparison,
            centroid_tracker=exp.centroid_tracker,
        )

        print_mechanistic_results(mechanistic_results)

        result = {
            "seed": seed,
            "accuracy": exp.accuracy,
            "dla_verification_error": mechanistic_results.dla_verification_error,
            "semantic_fidelity": mechanistic_results.semantic_fidelity,
            "eraser_metrics": mechanistic_results.eraser_metrics,
            "explainer_comparison": mechanistic_results.explainer_comparison,
            "layerwise_attribution": mechanistic_results.layerwise_attribution,
            "sae_comparison": mechanistic_results.sae_comparison,
            "circuit_completeness": {
                str(k): v for k, v in mechanistic_results.circuit_completeness.items()
            },
            "circuits": {},
        }

        for class_idx, circuit in mechanistic_results.circuits.items():
            result["circuits"][str(class_idx)] = {
                "token_ids": circuit.token_ids,
                "token_names": circuit.token_names,
                "attribution_scores": circuit.attribution_scores,
                "completeness_score": circuit.completeness_score,
            }

        all_results.append(result)

    _save_results(config, all_results)

    return all_results


def _save_results(config: Config, results: list[dict]) -> None:
    """Save experiment results to JSON."""
    output_path = os.path.join(config.output_dir, "experiment_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run CIS experiment: Train -> Mechanistic Evaluation",
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    config = load_config(args.config)
    run_experiment(config)


if __name__ == "__main__":
    main()
