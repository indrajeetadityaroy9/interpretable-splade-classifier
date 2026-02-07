"""SST-2 data loading."""

import random

from datasets import load_dataset


def _load_sst2_split(split: str, max_samples: int, seed: int) -> tuple[list[str], list[int]]:
    dataset = load_dataset("glue", "sst2", split=split)
    texts = list(dataset["sentence"])
    labels = [int(label) for label in dataset["label"]]
    combined = list(zip(texts, labels))
    random.Random(seed).shuffle(combined)
    shuffled_texts, shuffled_labels = zip(*combined)
    return list(shuffled_texts)[:max_samples], list(shuffled_labels)[:max_samples]


def load_sst2_data(
    train_samples: int,
    test_samples: int,
    seed: int = 42,
) -> tuple[list[str], list[int], list[str], list[int], int]:
    """Return shuffled SST-2 train/validation slices and number of labels."""
    train_texts, train_labels = _load_sst2_split("train", train_samples, seed)
    test_texts, test_labels = _load_sst2_split("validation", test_samples, seed)
    return train_texts, train_labels, test_texts, test_labels, 2
