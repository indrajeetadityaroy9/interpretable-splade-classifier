"""
Data loading utilities for SPLADE classifier.

Provides a unified interface for loading text classification data from:
- Local CSV/TSV files (custom datasets)
- HuggingFace datasets (standard benchmarks)

Example usage:
    # Option A: Local files
    texts, labels, meta = load_classification_data(file_path="data/train.csv")

    # Option B: HuggingFace datasets
    texts, labels, meta = load_classification_data(dataset="ag_news", split="train")
"""

from typing import Tuple, Optional, List, Dict, Any
import os
import random

import pandas as pd


# =============================================================================
# Supported HuggingFace Datasets Registry
# =============================================================================

SUPPORTED_DATASETS: Dict[str, Dict[str, Any]] = {
    # Binary classification
    "imdb": {
        "hf_name": "imdb",
        "text_column": "text",
        "label_column": "label",
        "num_labels": 2,
        "class_names": ["Negative", "Positive"],
    },
    "sst2": {
        "hf_name": "glue",
        "subset": "sst2",
        "text_column": "sentence",
        "label_column": "label",
        "num_labels": 2,
        "class_names": ["Negative", "Positive"],
    },
    "yelp_polarity": {
        "hf_name": "yelp_polarity",
        "text_column": "text",
        "label_column": "label",
        "num_labels": 2,
        "class_names": ["Negative", "Positive"],
    },
    "rotten_tomatoes": {
        "hf_name": "rotten_tomatoes",
        "text_column": "text",
        "label_column": "label",
        "num_labels": 2,
        "class_names": ["Negative", "Positive"],
    },

    # Multi-class classification
    "ag_news": {
        "hf_name": "ag_news",
        "text_column": "text",
        "label_column": "label",
        "num_labels": 4,
        "class_names": ["World", "Sports", "Business", "Sci/Tech"],
    },
    "yelp_review_full": {
        "hf_name": "yelp_review_full",
        "text_column": "text",
        "label_column": "label",
        "num_labels": 5,
        "class_names": ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"],
    },
    "dbpedia_14": {
        "hf_name": "fancyzhx/dbpedia_14",
        "text_column": "content",
        "label_column": "label",
        "num_labels": 14,
        "class_names": None,
    },
    "amazon_polarity": {
        "hf_name": "amazon_polarity",
        "text_column": "content",
        "label_column": "label",
        "num_labels": 2,
        "class_names": ["Negative", "Positive"],
    },
}


# =============================================================================
# Unified Data Loading Functions
# =============================================================================

def load_classification_data(
    # Option A: Local file
    file_path: Optional[str] = None,
    text_column: str = "text",
    label_column: str = "label",
    # Option B: HuggingFace dataset
    dataset: Optional[str] = None,
    split: str = "train",
    # Common options
    max_samples: Optional[int] = None,
    seed: Optional[int] = None,
    # Label consistency (pass train's mapping to test to ensure consistency)
    label_mapping: Optional[Dict[int, int]] = None,
) -> Tuple[List[str], List[int], Dict[str, Any]]:
    """
    Load text classification data from local files OR HuggingFace datasets.

    Args:
        file_path: Path to local CSV/TSV file (Option A)
        text_column: Column name containing text (for local files)
        label_column: Column name containing labels (for local files)
        dataset: HuggingFace dataset name (Option B)
        split: Dataset split ("train", "test", "validation")
        max_samples: Limit number of samples (for debugging)
        seed: Random seed for sampling (default: 42 if not provided)
        label_mapping: Pre-existing label mapping from training set. When provided,
            this mapping is applied to labels instead of computing a new one.
            Use this to ensure train and test splits use consistent label encoding.

    Returns:
        Tuple of (texts, labels, metadata)
        - texts: List[str] of input texts
        - labels: List[int] of integer labels
        - metadata: Dict with num_labels, class_names, dataset info, label_mapping

    Examples:
        # Option A: Local CSV/TSV file
        texts, labels, meta = load_classification_data(file_path="data/train.csv")

        # Option B: HuggingFace dataset
        texts, labels, meta = load_classification_data(dataset="ag_news", split="train")

        # Load test set with train's label mapping for consistency
        train_texts, train_labels, train_meta = load_classification_data(file_path="train.csv")
        test_texts, test_labels, test_meta = load_classification_data(
            file_path="test.csv",
            label_mapping=train_meta['label_mapping']
        )

    Raises:
        ValueError: If neither file_path nor dataset is specified
        ValueError: If label_mapping is provided but labels contain unknown values
    """
    if file_path is not None:
        # Option A: Load from local file
        texts, labels, metadata = _load_from_file(
            file_path, text_column, label_column, label_mapping=label_mapping
        )
    elif dataset is not None:
        # Option B: Load from HuggingFace
        texts, labels, metadata = _load_from_huggingface(
            dataset, split, label_mapping=label_mapping
        )
    else:
        raise ValueError(
            "Must specify either 'file_path' (for local files) or "
            "'dataset' (for HuggingFace datasets)"
        )

    # Sample if requested
    if max_samples is not None and len(texts) > max_samples:
        # Use provided seed or default to 42 for reproducibility
        sample_seed = seed if seed is not None else 42
        random.seed(sample_seed)
        indices = random.sample(range(len(texts)), max_samples)
        texts = [texts[i] for i in indices]
        labels = [labels[i] for i in indices]
        metadata["num_samples"] = max_samples
        metadata["sample_seed"] = sample_seed
    else:
        metadata["num_samples"] = len(texts)

    return texts, labels, metadata


def _normalize_labels(
    labels: List[int],
) -> Tuple[List[int], Dict[int, int], int]:
    """
    Normalize labels to be 0-indexed and contiguous.

    Handles cases where labels are:
    - Non-contiguous: [0, 2, 5] -> [0, 1, 2]
    - Not 0-indexed: [1, 2, 3] -> [0, 1, 2]

    Args:
        labels: List of integer labels (possibly non-contiguous)

    Returns:
        Tuple of:
        - normalized_labels: List[int] with values in [0, num_classes-1]
        - label_mapping: Dict mapping original labels to normalized labels
        - num_labels: Number of unique classes

    Example:
        >>> _normalize_labels([5, 2, 5, 0, 2])
        ([2, 1, 2, 0, 1], {0: 0, 2: 1, 5: 2}, 3)
    """
    unique_labels = sorted(set(labels))
    num_labels = len(unique_labels)

    # Check if already normalized (0-indexed and contiguous)
    expected = list(range(num_labels))
    if unique_labels == expected:
        # Already normalized, no mapping needed
        return labels, {i: i for i in unique_labels}, num_labels

    # Create mapping from original to normalized labels
    label_mapping = {orig: norm for norm, orig in enumerate(unique_labels)}

    # Apply mapping
    normalized_labels = [label_mapping[l] for l in labels]

    # Log warning about remapping
    import warnings
    warnings.warn(
        f"Labels were not 0-indexed and contiguous. "
        f"Remapped {unique_labels} -> {list(range(num_labels))}. "
        f"Original label mapping: {label_mapping}",
        UserWarning
    )

    return normalized_labels, label_mapping, num_labels


def _load_from_file(
    file_path: str,
    text_column: str,
    label_column: str,
    label_mapping: Optional[Dict[int, int]] = None,
) -> Tuple[List[str], List[int], Dict[str, Any]]:
    """Load data from a local CSV/TSV file.

    Args:
        file_path: Path to the CSV/TSV file
        text_column: Name of the text column
        label_column: Name of the label column
        label_mapping: Optional pre-existing label mapping to apply.
            If provided, this mapping is used instead of computing a new one.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext in (".tsv", ".txt"):
        sep = "\t"
    else:
        sep = ","

    try:
        df = pd.read_csv(file_path, sep=sep)
        if text_column not in df.columns:
            # Try without header (legacy TSV format: id, text, label)
            df = pd.read_csv(
                file_path, sep=sep, header=None,
                names=["id", text_column, label_column]
            )
    except (FileNotFoundError, pd.errors.ParserError, pd.errors.EmptyDataError) as e:
        raise ValueError(f"Failed to load {file_path}: {e}")

    texts = df[text_column].astype(str).tolist()
    raw_labels = df[label_column].astype(int).tolist()

    if label_mapping is not None:
        # Validate that all labels are in the mapping
        unknown_labels = set(raw_labels) - set(label_mapping.keys())
        if unknown_labels:
            raise ValueError(
                f"Labels {unknown_labels} in {file_path} not found in provided label_mapping. "
                f"Known labels: {set(label_mapping.keys())}. "
                f"Ensure train and test sets have the same label space."
            )
        labels = [label_mapping[l] for l in raw_labels]
        num_labels = len(set(label_mapping.values()))
    else:
        # Validate and normalize labels to be 0-indexed and contiguous
        labels, label_mapping, num_labels = _normalize_labels(raw_labels)

    metadata = {
        "source": "file",
        "file_path": file_path,
        "num_labels": num_labels,
        "class_names": None,
        "label_mapping": label_mapping,  # Original label -> normalized label
    }

    return texts, labels, metadata


def _load_from_huggingface(
    dataset_name: str,
    split: str,
    label_mapping: Optional[Dict[int, int]] = None,
) -> Tuple[List[str], List[int], Dict[str, Any]]:
    """Load data from HuggingFace datasets.

    Args:
        dataset_name: Name of the HuggingFace dataset
        split: Dataset split (train, test, validation)
        label_mapping: Optional pre-existing label mapping to apply.
    """
    try:
        from datasets import load_dataset as hf_load_dataset
    except ImportError:
        raise ImportError(
            "HuggingFace datasets library is required for loading HuggingFace datasets. "
            "Install with: pip install datasets"
        )

    # Get config from registry or use defaults
    if dataset_name in SUPPORTED_DATASETS:
        config = SUPPORTED_DATASETS[dataset_name]
        hf_name = config["hf_name"]
        subset = config.get("subset")
        text_column = config["text_column"]
        label_column = config["label_column"]
        num_labels = config["num_labels"]
        class_names = config.get("class_names")
    else:
        # Assume it's a direct HuggingFace dataset name
        hf_name = dataset_name
        subset = None
        text_column = "text"
        label_column = "label"
        num_labels = None
        class_names = None

    # Load dataset
    if subset:
        ds = hf_load_dataset(hf_name, subset, split=split)
    else:
        ds = hf_load_dataset(hf_name, split=split)

    # Extract texts and labels
    texts = list(ds[text_column])
    raw_labels = [int(l) for l in ds[label_column]]

    # Apply existing mapping or use registry/compute new one
    if label_mapping is not None:
        # Validate that all labels are in the mapping
        unknown_labels = set(raw_labels) - set(label_mapping.keys())
        if unknown_labels:
            raise ValueError(
                f"Labels {unknown_labels} in {dataset_name}/{split} not found in provided label_mapping. "
                f"Known labels: {set(label_mapping.keys())}. "
                f"Ensure train and test sets have the same label space."
            )
        labels = [label_mapping[l] for l in raw_labels]
        num_labels = len(set(label_mapping.values()))
    elif num_labels is None:
        labels, label_mapping, num_labels = _normalize_labels(raw_labels)
    else:
        labels = raw_labels
        label_mapping = {i: i for i in range(num_labels)}

    metadata = {
        "source": "huggingface",
        "dataset": dataset_name,
        "split": split,
        "num_labels": num_labels,
        "class_names": class_names,
        "label_mapping": label_mapping,
    }

    return texts, labels, metadata


def list_supported_datasets() -> List[str]:
    """Return list of supported HuggingFace dataset names."""
    return list(SUPPORTED_DATASETS.keys())
