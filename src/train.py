"""
SPLADE classifier training script.

Supports two data sources:
    Option A - Local files:
        python -m src.train --train_path data/train.csv --test_path data/test.csv --epochs 5

    Option B - HuggingFace datasets:
        python -m src.train --dataset imdb --epochs 5
        python -m src.train --dataset ag_news --epochs 5
"""

import argparse
import os
import time

from src.models import SPLADEClassifier
from src.data import load_classification_data
from src.utils import validate_data_sources


def train(args):
    """Train SPLADE classifier using the sklearn-style API."""
    print("Loading data...")

    has_local, has_hf = validate_data_sources(
        args.train_path, args.test_path, args.dataset,
        raise_on_error=False, print_error=True
    )
    if not has_local and not has_hf:
        return

    if has_local:
        print(f"  Source: Local files")
        print(f"  Train: {args.train_path}")
        print(f"  Test: {args.test_path}")
        train_texts, train_labels, train_meta = load_classification_data(
            file_path=args.train_path,
            max_samples=args.max_train_samples
        )
        test_texts, test_labels, test_meta = load_classification_data(
            file_path=args.test_path
        )
        num_classes = train_meta.get('num_labels', max(train_labels) + 1)
        class_names = None
    else:
        print(f"  Source: HuggingFace dataset '{args.dataset}'")
        train_texts, train_labels, train_meta = load_classification_data(
            dataset=args.dataset,
            split="train",
            max_samples=args.max_train_samples
        )
        test_texts, test_labels, test_meta = load_classification_data(
            dataset=args.dataset,
            split="test"
        )
        num_classes = train_meta['num_labels']
        class_names = train_meta.get('class_names')

    # For binary classification (2 classes), use num_labels=1 (BCE loss)
    # For multi-class, use num_labels=num_classes (CrossEntropy loss)
    num_labels = 1 if num_classes == 2 else num_classes

    print(f"  Train samples: {len(train_texts)}")
    print(f"  Test samples: {len(test_texts)}")
    print(f"  Num classes: {num_classes}")
    if class_names:
        print(f"  Classes: {class_names}")

    clf = SPLADEClassifier(
        num_labels=num_labels,
        class_names=class_names,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        flops_lambda=args.flops_lambda,
        verbose=True,
    )

    print(f"\nStarting training for {args.epochs} epochs...")
    start_time = time.time()

    clf.fit(train_texts, train_labels, epochs=args.epochs)

    print(f"\nTraining finished in {time.time() - start_time:.2f}s")

    accuracy = clf.score(test_texts, test_labels)
    sparsity = clf.get_sparsity(test_texts)
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Sparsity: {sparsity:.2f}%")

    model_path = os.path.join(args.output_dir, 'model.pth')
    os.makedirs(args.output_dir, exist_ok=True)
    clf.save(model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train SPLADE classifier on text classification data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train on local CSV/TSV files
  python -m src.train --train_path data/train.csv --test_path data/test.csv --epochs 5

  # Train on HuggingFace dataset
  python -m src.train --dataset imdb --epochs 5
  python -m src.train --dataset ag_news --epochs 3

  # Quick test with limited samples
  python -m src.train --dataset ag_news --epochs 1 --max_train_samples 1000
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

    # Training options
    parser.add_argument('--max_train_samples', type=int, default=None,
                       help='Limit training samples (for debugging)')
    parser.add_argument('--output_dir', type=str, default='models',
                       help='Directory to save trained model')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--flops_lambda', type=float, default=1e-4,
                       help='FLOPS regularization strength')

    args = parser.parse_args()
    train(args)
