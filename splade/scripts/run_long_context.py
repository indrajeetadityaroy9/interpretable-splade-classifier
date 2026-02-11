"""Experiment C: Long-Context Needle in Haystack.

Tests whether Lexical-SAE's sparse bottleneck naturally localizes signal
in long documents. Trains on short texts, then evaluates on progressively
longer inputs (padding with irrelevant text). ModernBERT supports up to 8192 tokens.

Key insight: The max-pool sparse bottleneck should preserve signal regardless
of document length, since the informative tokens still produce high sparse
activations. The attribution should point to the same tokens.
"""

import argparse
import json
import os
import random
from dataclasses import asdict

import yaml

from splade.config.load import load_config
from splade.inference import score_model
from splade.pipelines import setup_and_train


def _pad_texts_to_length(
    texts: list[str],
    target_word_count: int,
    filler_texts: list[str],
    seed: int = 42,
) -> list[str]:
    """Pad each text to approximately target_word_count by appending filler."""
    rng = random.Random(seed)
    padded = []
    for text in texts:
        words = text.split()
        while len(words) < target_word_count and filler_texts:
            filler = rng.choice(filler_texts)
            words.extend(filler.split())
        padded.append(" ".join(words[:target_word_count]))
    return padded


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment C: Long-Context Needle in Haystack")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    config = load_config(args.config)
    os.makedirs(config.output_dir, exist_ok=True)

    with open(os.path.join(config.output_dir, "resolved_config.yaml"), "w") as f:
        yaml.dump(asdict(config), f)

    print(f"\n{'=' * 60}")
    print("LONG-CONTEXT NEEDLE IN HAYSTACK EXPERIMENT")
    print(f"{'=' * 60}")

    seed = config.seed

    # Train on standard-length texts
    print("\n--- Training Lexical-SAE ---")
    exp = setup_and_train(config, seed)
    print(f"Baseline accuracy (original length): {exp.accuracy:.4f}")

    # Load filler texts for padding (use train texts as filler)
    filler_texts = exp.train_texts[:500]

    # Test at increasing lengths
    target_lengths = [50, 100, 200, 500, 1000, 2000]
    max_token_lengths = [64, 128, 256, 512, 1024, 2048]

    results = {
        "seed": seed,
        "baseline_accuracy": exp.accuracy,
        "length_results": [],
    }

    print(f"\n{'Word Count':<15} {'Max Tokens':<15} {'Accuracy':>10}")
    print("-" * 40)

    for word_count, max_tokens in zip(target_lengths, max_token_lengths):
        padded_texts = _pad_texts_to_length(
            exp.test_texts, word_count, filler_texts, seed=seed,
        )
        acc = score_model(
            exp.model, exp.tokenizer, padded_texts, exp.test_labels,
            max_length=max_tokens, batch_size=exp.batch_size,
            num_labels=exp.num_labels,
        )
        print(f"{word_count:<15} {max_tokens:<15} {acc:>10.4f}")
        results["length_results"].append({
            "word_count": word_count,
            "max_tokens": max_tokens,
            "accuracy": acc,
        })

    output_path = os.path.join(config.output_dir, "long_context_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
