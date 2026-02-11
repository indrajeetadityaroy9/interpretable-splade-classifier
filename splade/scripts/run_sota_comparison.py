"""Experiment: SOTA Comparison.

Trains Lexical-SAE and a dense ModernBERT baseline on the same data,
runs word-order sensitivity tests, and optionally compares against
a post-hoc SAE. Outputs a unified comparison table.
"""

import argparse
import json
import os
from dataclasses import asdict

import yaml

from splade.config.load import load_config
from splade.evaluation.dense_baseline import (
    score_dense_baseline,
    train_dense_baseline,
)
from splade.evaluation.mechanistic import (
    print_mechanistic_results,
    run_mechanistic_evaluation,
)
from splade.evaluation.word_order import evaluate_word_order_sensitivity
from splade.pipelines import prepare_mechanistic_inputs, setup_and_train


def main() -> None:
    parser = argparse.ArgumentParser(description="SOTA Comparison Experiment")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config.seed
    os.makedirs(config.output_dir, exist_ok=True)

    with open(os.path.join(config.output_dir, "resolved_config.yaml"), "w") as f:
        yaml.dump(asdict(config), f)

    print(f"\n{'=' * 60}")
    print(f"SOTA COMPARISON EXPERIMENT")
    print(f"{'=' * 60}")

    # 1. Train Lexical-SAE
    print("\n--- Training Lexical-SAE ---")
    exp = setup_and_train(config, seed=seed)
    lexical_acc = exp.accuracy
    print(f"Lexical-SAE test accuracy: {lexical_acc:.4f}")

    # 2. Train dense baseline (same data, same seed)
    print("\n--- Training Dense Baseline ---")
    dense_model, dense_tokenizer, dense_val_acc = train_dense_baseline(
        model_name=config.model.name,
        train_texts=exp.train_texts,
        train_labels=exp.train_labels,
        val_texts=exp.val_texts,
        val_labels=exp.val_labels,
        num_labels=exp.num_labels,
        max_length=exp.max_length,
        seed=seed,
    )
    dense_test_acc = score_dense_baseline(
        dense_model, dense_tokenizer, exp.test_texts, exp.test_labels, exp.max_length,
    )
    print(f"Dense baseline test accuracy: {dense_test_acc:.4f}")

    # 3. Word-order sensitivity
    print("\n--- Word-Order Sensitivity ---")
    word_order = evaluate_word_order_sensitivity(
        exp.model, exp.tokenizer, exp.max_length,
    )
    print(f"Mean L2 distance: {word_order['mean_l2_distance']:.4f}")
    print(f"Mean cosine similarity: {word_order['mean_cosine_similarity']:.4f}")
    print(f"Prediction agreement: {word_order['prediction_agreement_rate']:.2%}")
    for pair in word_order["per_pair"]:
        print(f"  '{pair['sent_a'][:40]}...' vs '{pair['sent_b'][:40]}...'")
        print(f"    L2={pair['l2']:.4f}  cos={pair['cosine']:.4f}  "
              f"pred_a={pair['pred_a']}  pred_b={pair['pred_b']}")

    # 4. Mechanistic evaluation (with SAE comparison if enabled)
    print("\n--- Mechanistic Evaluation ---")
    input_ids_list, attention_mask_list = prepare_mechanistic_inputs(
        exp.tokenizer, exp.test_texts, exp.max_length,
    )
    mech_results = run_mechanistic_evaluation(
        exp.model, input_ids_list, attention_mask_list,
        exp.test_labels, exp.tokenizer, num_classes=exp.num_labels,
        run_sae_comparison=True,
        centroid_tracker=exp.centroid_tracker,
        texts=exp.test_texts,
        max_length=exp.max_length,
    )
    print_mechanistic_results(mech_results)

    # 5. Unified comparison table
    print(f"\n{'=' * 75}")
    print(f"COMPARISON TABLE")
    print(f"{'=' * 75}")
    print(f"{'Method':<25} {'Accuracy':>10} {'Recon Error':>12} {'Active Dims':>12} {'Order-Sens':>12}")
    print("-" * 71)
    print(f"{'Dense ' + config.model.name.split('/')[-1]:<25} {dense_test_acc:>10.4f} {'N/A':>12} {'N/A':>12} {'N/A':>12}")
    print(
        f"{'Lexical-SAE':<25} {lexical_acc:>10.4f} {'0.000':>12} "
        f"{mech_results.mean_active_dims:>12.0f} "
        f"{'Yes' if word_order['prediction_agreement_rate'] < 0.8 else 'Partial':>12}"
    )
    if mech_results.sae_comparison:
        sae = mech_results.sae_comparison
        dead_str = f"{sae.get('sae_dead_feature_fraction', 0):.1%} dead" if 'sae_dead_feature_fraction' in sae else "N/A"
        print(
            f"{'Post-hoc SAE':<25} {'N/A':>10} {sae['reconstruction_error']:>12.4f} "
            f"{sae['sae_active_features']:>12.0f} {'N/A':>12}"
        )
        print(f"  SAE details: hidden_dim={sae.get('sae_hidden_dim', 'N/A')}, {dead_str}")
    print(f"{'=' * 75}")

    # Save results
    results = {
        "seed": seed,
        "lexical_sae_accuracy": lexical_acc,
        "dense_baseline_accuracy": dense_test_acc,
        "dense_val_accuracy": dense_val_acc,
        "word_order": word_order,
        "mean_active_dims": mech_results.mean_active_dims,
        "dla_verification_error": mech_results.dla_verification_error,
        "sae_comparison": mech_results.sae_comparison,
        "polysemy_scores": mech_results.polysemy_scores,
    }

    output_path = os.path.join(config.output_dir, "sota_comparison_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
