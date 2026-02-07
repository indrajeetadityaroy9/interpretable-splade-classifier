"""CLI entry point for canonical SPLADE training on SST-2."""

import argparse
import os

import torch

from src.data.loader import load_sst2_data
from src.models.classifier import SPLADEClassifier
from src.utils.cuda import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SPLADE on SST-2")
    parser.add_argument("--train-samples", type=int, default=2000)
    parser.add_argument("--test-samples", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=None, help="Fixed epochs (default: early stopping)")
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    set_seed(args.seed)
    train_texts, train_labels, test_texts, test_labels, num_labels = load_sst2_data(
        args.train_samples,
        args.test_samples,
        seed=args.seed,
    )

    clf = SPLADEClassifier(num_labels=num_labels, batch_size=args.batch_size)
    clf.fit(train_texts, train_labels, epochs=args.epochs, max_epochs=args.max_epochs)

    accuracy = clf.score(test_texts, test_labels)
    print(f"\nTest accuracy: {accuracy:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.output_dir, f"splade_sst2_seed{args.seed}.pt")
    torch.save(clf.model.state_dict(), checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
