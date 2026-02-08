"""CIS ablation study: Full CIS vs DF-FLOPS only vs Baseline.

Ablation variants are created by temporarily zeroing CIS constants,
not by config flags. All three variants train from scratch.
"""

import argparse
from unittest.mock import patch

from splade.config.load import load_config
from splade.evaluation.mechanistic import (print_mechanistic_results,
                                           run_mechanistic_evaluation)
from splade.pipelines import prepare_mechanistic_inputs, setup_and_train
from splade.training import constants


def _run_variant(config, seed, name):
    print(f"\n{'='*60}")
    print(f"Ablation: {name}")
    print(f"{'='*60}")

    exp = setup_and_train(config, seed)

    input_ids_list, attention_mask_list = prepare_mechanistic_inputs(
        exp.tokenizer, exp.test_texts, exp.max_length,
    )

    results = run_mechanistic_evaluation(
        exp.model, input_ids_list, attention_mask_list,
        exp.test_labels, exp.tokenizer, num_classes=exp.num_labels,
    )
    print(f"\n--- {name} Results ---")
    print(f"  Accuracy: {exp.accuracy:.4f}")
    print_mechanistic_results(results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CIS ablation study")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config.evaluation.seeds[0]

    # Variant 1: Baseline — zero all circuit lambdas
    # (DF-FLOPS is always active via SatLambdaSchedule, not a patchable constant,
    #  so "Baseline" and "DF-FLOPS only" are equivalent)
    with patch.object(constants, "CIRCUIT_COMPLETENESS_LAMBDA", 0.0), \
         patch.object(constants, "CIRCUIT_SEPARATION_LAMBDA", 0.0), \
         patch.object(constants, "CIRCUIT_SHARPNESS_LAMBDA", 0.0):
        _run_variant(config, seed, "Baseline (no circuit losses)")

    # Variant 2: Full CIS — default constants
    _run_variant(config, seed, "Full CIS")


if __name__ == "__main__":
    main()
