"""Tests for ERASER faithfulness metrics."""

import torch
import pytest

from splade.evaluation.eraser import (
    _get_topk_mask,
    compute_aopc,
    compute_comprehensiveness,
    compute_sufficiency,
)


@pytest.fixture
def simple_model():
    """Minimal model with classifier_logits_only for testing."""

    class FakeModel:
        def classifier_logits_only(self, sparse_vector: torch.Tensor) -> torch.Tensor:
            # Linear classifier: logit_0 = sum of first half, logit_1 = sum of second half
            V = sparse_vector.shape[1]
            mid = V // 2
            logit_0 = sparse_vector[:, :mid].sum(dim=1)
            logit_1 = sparse_vector[:, mid:].sum(dim=1)
            return torch.stack([logit_0, logit_1], dim=1)

    return FakeModel()


class TestGetTopkMask:
    def test_correct_count(self):
        # 10 active (non-zero) dims, top 30% of 10 = 3
        attr = torch.tensor([[0.1, 0.5, 0.3, 0.9, 0.2, 0.05, 0.7, 0.4, 0.6, 0.8]])
        mask = _get_topk_mask(attr, 0.3)
        assert mask.sum().item() == 3

    def test_selects_highest(self):
        attr = torch.tensor([[0.1, 0.9, 0.2, 0.8, 0.3]])
        mask = _get_topk_mask(attr, 0.4)  # top 40% = 2 dims
        assert mask[0, 1].item() is True  # 0.9
        assert mask[0, 3].item() is True  # 0.8

    def test_minimum_one(self):
        attr = torch.zeros(1, 100)
        attr[0, 50] = 1.0
        mask = _get_topk_mask(attr, 0.001)  # 0.1% of 100 = 0, clamped to 1
        assert mask.sum().item() == 1


class TestComprehensiveness:
    def test_removing_important_dims_hurts(self, simple_model):
        V = 20
        sparse = torch.zeros(4, V)
        sparse[:, :10] = 1.0  # class 0 signal in first half
        labels = torch.zeros(4, dtype=torch.long)

        # Attribution points to first half (correctly)
        attr = torch.zeros(4, V)
        attr[:, :10] = 1.0

        comp = compute_comprehensiveness(simple_model, sparse, attr, labels)
        # Removing top dims should drop probability significantly
        assert comp[0.50] > 0.0

    def test_wrong_attribution_lower(self, simple_model):
        V = 20
        sparse = torch.zeros(4, V)
        sparse[:, :10] = 1.0
        labels = torch.zeros(4, dtype=torch.long)

        # Perfect attribution: points to important dims (first half)
        attr_perfect = torch.zeros(4, V)
        attr_perfect[:, :10] = 1.0
        comp_perfect = compute_comprehensiveness(simple_model, sparse, attr_perfect, labels)

        # Wrong attribution: points to unimportant dims (second half)
        # Since sparse[:, 10:] = 0, removing these has zero effect → comp ≈ 0
        attr_wrong = torch.zeros(4, V)
        attr_wrong[:, 10:] = 1.0
        comp_wrong = compute_comprehensiveness(simple_model, sparse, attr_wrong, labels)

        # Perfect should have higher comprehensiveness
        assert comp_perfect[0.50] > comp_wrong[0.50]


class TestSufficiency:
    def test_keeping_important_dims_preserves(self, simple_model):
        V = 20
        sparse = torch.zeros(4, V)
        sparse[:, :10] = 1.0
        labels = torch.zeros(4, dtype=torch.long)

        attr = torch.zeros(4, V)
        attr[:, :10] = 1.0

        suff = compute_sufficiency(simple_model, sparse, attr, labels)
        # Keeping top 50% (which is the important half) should preserve prediction
        # Sufficiency = original_prob - patched_prob, lower = better
        assert suff[0.50] < 0.5


class TestAOPC:
    def test_bounds(self):
        scores = {0.01: 0.1, 0.05: 0.2, 0.10: 0.3, 0.20: 0.5, 0.50: 0.8}
        aopc = compute_aopc(scores)
        assert 0.0 <= aopc <= 1.0

    def test_mean(self):
        scores = {0.1: 0.2, 0.2: 0.4}
        assert abs(compute_aopc(scores) - 0.3) < 1e-6

    def test_empty(self):
        assert compute_aopc({}) == 0.0
