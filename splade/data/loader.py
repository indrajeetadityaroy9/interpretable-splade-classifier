import random

import numpy
from datasets import load_dataset


_DATASETS = {
    "sst2": {"path": "glue", "name": "sst2", "text_col": "sentence", "test_split": "validation", "num_labels": 2},
    "ag_news": {"path": "ag_news", "name": None, "text_col": "text", "test_split": "test", "num_labels": 4},
    "imdb": {"path": "imdb", "name": None, "text_col": "text", "test_split": "test", "num_labels": 2},
    "yelp": {"path": "yelp_polarity", "name": None, "text_col": "text", "test_split": "test", "num_labels": 2},
}


def _shuffle_and_truncate(
    texts: list[str], labels: list[int], max_samples: int, seed: int
) -> tuple[list[str], list[int]]:
    combined = list(zip(texts, labels))
    random.Random(seed).shuffle(combined)
    shuffled_texts, shuffled_labels = zip(*combined)
    return list(shuffled_texts)[:max_samples], list(shuffled_labels)[:max_samples]


def infer_max_length(texts: list[str], tokenizer) -> int:
    sample = texts[:500]
    lengths = [len(tokenizer.encode(t, add_special_tokens=True)) for t in sample]
    p99 = int(numpy.percentile(lengths, 99))
    aligned = ((p99 + 7) // 8) * 8
    return max(64, min(512, aligned))


def _load_split(cfg: dict, split: str, max_samples: int, seed: int) -> tuple[list[str], list[int]]:
    dataset = load_dataset(cfg["path"], cfg["name"], split=split)
    texts = list(dataset[cfg["text_col"]])
    labels = [int(label) for label in dataset["label"]]
    return _shuffle_and_truncate(texts, labels, max_samples, seed)


def load_dataset_by_name(
    name: str,
    train_samples: int,
    test_samples: int,
    seed: int = 42,
) -> tuple[list[str], list[int], list[str], list[int], int]:
    cfg = _DATASETS[name]
    train_texts, train_labels = _load_split(cfg, "train", train_samples, seed)
    test_texts, test_labels = _load_split(cfg, cfg["test_split"], test_samples, seed)
    return train_texts, train_labels, test_texts, test_labels, cfg["num_labels"]
