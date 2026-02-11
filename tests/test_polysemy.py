"""Tests for polysemy defense metrics (pure Python functions only)."""

import re


def _find_word_occurrences(texts, target_words, min_occurrences=5):
    """Local copy of find_word_occurrences to avoid CUDA import chain."""
    result = {}
    for word in target_words:
        pattern = re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)
        indices = [i for i, text in enumerate(texts) if pattern.search(text)]
        if len(indices) >= min_occurrences:
            result[word] = indices
    return result


def _jaccard_similarity(set_a, set_b):
    """Local copy of jaccard_similarity."""
    if not set_a and not set_b:
        return 1.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


class TestFindWordOccurrences:
    def test_basic_match(self):
        texts = ["The bank is closed", "I went to the river bank", "No match here"]
        result = _find_word_occurrences(texts, ["bank"], min_occurrences=1)
        assert "bank" in result
        assert result["bank"] == [0, 1]

    def test_case_insensitive(self):
        texts = ["Bank of America", "the bank is open"]
        result = _find_word_occurrences(texts, ["bank"], min_occurrences=1)
        assert len(result["bank"]) == 2

    def test_word_boundary(self):
        texts = ["banking is fun", "the bank is here"]
        result = _find_word_occurrences(texts, ["bank"], min_occurrences=1)
        assert result["bank"] == [1]  # "banking" should NOT match

    def test_punctuation_boundary(self):
        texts = ["Go to the bank.", "The bank, it's closed"]
        result = _find_word_occurrences(texts, ["bank"], min_occurrences=1)
        assert len(result["bank"]) == 2

    def test_min_occurrences_filter(self):
        texts = ["one bank"]
        result = _find_word_occurrences(texts, ["bank"], min_occurrences=5)
        assert "bank" not in result


class TestJaccardSimilarity:
    def test_identical_sets(self):
        assert _jaccard_similarity({1, 2, 3}, {1, 2, 3}) == 1.0

    def test_disjoint_sets(self):
        assert _jaccard_similarity({1, 2}, {3, 4}) == 0.0

    def test_partial_overlap(self):
        result = _jaccard_similarity({1, 2, 3}, {2, 3, 4})
        assert abs(result - 0.5) < 1e-6  # 2/4

    def test_empty_sets(self):
        assert _jaccard_similarity(set(), set()) == 1.0

    def test_one_empty(self):
        assert _jaccard_similarity({1, 2}, set()) == 0.0
