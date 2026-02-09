"""Tests for data loader, including e-SNLI multi-column support."""

import pytest

from splade.data.loader import _DATASETS, _load_split


class TestDatasetRegistry:
    def test_all_required_datasets_present(self):
        for name in ("sst2", "ag_news", "esnli"):
            assert name in _DATASETS, f"Missing required dataset: {name}"

    def test_esnli_has_text_cols(self):
        cfg = _DATASETS["esnli"]
        assert "text_cols" in cfg
        assert cfg["text_cols"] == ["premise", "hypothesis"]
        assert cfg["num_labels"] == 3

    @pytest.mark.parametrize("name", list(_DATASETS.keys()))
    def test_dataset_has_required_fields(self, name):
        cfg = _DATASETS[name]
        assert "path" in cfg
        assert "test_split" in cfg
        assert "num_labels" in cfg
        assert "text_col" in cfg or "text_cols" in cfg


class TestEsnliLoading:
    @pytest.mark.slow
    def test_esnli_loads_and_returns_3_classes(self):
        """Verify e-SNLI loads correctly (requires network access)."""
        cfg = _DATASETS["esnli"]
        texts, labels = _load_split(cfg, "test", 50, seed=42)
        assert len(texts) == 50
        assert len(labels) == 50
        assert all(lbl in (0, 1, 2) for lbl in labels)

    @pytest.mark.slow
    def test_esnli_text_format(self):
        """Verify premise/hypothesis concatenation."""
        cfg = _DATASETS["esnli"]
        texts, _ = _load_split(cfg, "test", 10, seed=42)
        for text in texts:
            assert "[SEP]" in text, f"Expected [SEP] separator in: {text[:80]}"
