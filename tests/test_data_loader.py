"""Tests for data loader, including Banking77 and BeaverTails support."""

import pytest

from splade.data.loader import _DATASETS, _load_split


class TestDatasetRegistry:
    def test_all_required_datasets_present(self):
        for name in ("banking77", "beavertails"):
            assert name in _DATASETS, f"Missing required dataset: {name}"

    def test_banking77_config(self):
        cfg = _DATASETS["banking77"]
        assert cfg["num_labels"] == 77
        assert cfg["text_col"] == "text"

    def test_beavertails_config(self):
        cfg = _DATASETS["beavertails"]
        assert cfg["num_labels"] == 2
        assert "text_cols" in cfg
        assert cfg["text_cols"] == ["prompt", "response"]
        assert cfg.get("label_invert") is True

    @pytest.mark.parametrize("name", list(_DATASETS.keys()))
    def test_dataset_has_required_fields(self, name):
        cfg = _DATASETS[name]
        assert "path" in cfg
        assert "test_split" in cfg
        assert "num_labels" in cfg
        assert "text_col" in cfg or "text_cols" in cfg


class TestBeavertailsLoading:
    @pytest.mark.slow
    def test_beavertails_loads_binary(self):
        """Verify BeaverTails loads with inverted binary labels."""
        cfg = _DATASETS["beavertails"]
        texts, labels = _load_split(cfg, "test", 50, seed=42)
        assert len(texts) == 50
        assert all(lbl in (0, 1) for lbl in labels)

    @pytest.mark.slow
    def test_beavertails_text_format(self):
        """Verify prompt/response concatenation."""
        cfg = _DATASETS["beavertails"]
        texts, _ = _load_split(cfg, "test", 10, seed=42)
        for text in texts:
            assert "[SEP]" in text, f"Expected [SEP] separator in: {text[:80]}"
