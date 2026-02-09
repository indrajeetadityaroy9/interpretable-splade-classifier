"""Tests for B-cos layers and classifier."""

import torch

from splade.models.bcos import BcosLinear, BcosClassifier


class TestBcosLinear:
    def test_output_shape(self):
        layer = BcosLinear(64, 32, B=2)
        x = torch.randn(4, 64)
        out = layer(x)
        assert out.shape == (4, 32)

    def test_dynamic_weight_shape(self):
        layer = BcosLinear(64, 32, B=2)
        x = torch.randn(4, 64)
        W_dyn = layer.get_dynamic_weight(x)
        assert W_dyn.shape == (4, 32, 64)

    def test_dynamic_weight_exact(self):
        """W_dyn @ x should exactly reproduce forward(x)."""
        torch.manual_seed(42)
        layer = BcosLinear(64, 32, B=2)
        x = torch.randn(4, 64)

        out = layer(x)
        W_dyn = layer.get_dynamic_weight(x)
        reconstructed = torch.bmm(W_dyn, x.unsqueeze(-1)).squeeze(-1)

        assert torch.allclose(out, reconstructed, atol=1e-5)

    def test_dynamic_weight_exact_b3(self):
        """W_dyn @ x should exactly reproduce forward(x) for B=3."""
        torch.manual_seed(42)
        layer = BcosLinear(64, 32, B=3)
        x = torch.randn(4, 64)

        out = layer(x)
        W_dyn = layer.get_dynamic_weight(x)
        reconstructed = torch.bmm(W_dyn, x.unsqueeze(-1)).squeeze(-1)

        assert torch.allclose(out, reconstructed, atol=1e-5)

    def test_no_bias(self):
        layer = BcosLinear(64, 32)
        param_names = [name for name, _ in layer.named_parameters()]
        assert "bias" not in param_names
        assert "weight" in param_names


class TestBcosClassifier:
    def test_forward_shape(self):
        clf = BcosClassifier(100, 32, 2, num_layers=2)
        x = torch.randn(4, 100)
        logits = clf(x)
        assert logits.shape == (4, 2)

    def test_classifier_forward_shape(self):
        clf = BcosClassifier(100, 32, 2, num_layers=2)
        x = torch.randn(4, 100)
        logits, W_eff, b_eff = clf.classifier_forward(x)
        assert logits.shape == (4, 2)
        assert W_eff.shape == (4, 2, 100)
        assert b_eff.shape == (4, 2)

    def test_w_eff_exact_2layer(self):
        """DLA identity: logits == W_eff @ sparse_vector (no bias in B-cos)."""
        torch.manual_seed(42)
        clf = BcosClassifier(100, 32, 2, num_layers=2)
        x = torch.randn(4, 100)

        logits, W_eff, b_eff = clf.classifier_forward(x)
        reconstructed = torch.bmm(W_eff, x.unsqueeze(-1)).squeeze(-1) + b_eff

        assert torch.allclose(logits, reconstructed, atol=1e-4), \
            f"Max error: {(logits - reconstructed).abs().max().item()}"

    def test_w_eff_exact_3layer(self):
        """DLA identity holds for 3-layer B-cos too."""
        torch.manual_seed(42)
        clf = BcosClassifier(100, 32, 2, num_layers=3)
        x = torch.randn(4, 100)

        logits, W_eff, b_eff = clf.classifier_forward(x)
        reconstructed = torch.bmm(W_eff, x.unsqueeze(-1)).squeeze(-1) + b_eff

        assert torch.allclose(logits, reconstructed, atol=1e-4), \
            f"Max error: {(logits - reconstructed).abs().max().item()}"

    def test_b_eff_is_zero(self):
        clf = BcosClassifier(100, 32, 2, num_layers=2)
        x = torch.randn(4, 100)
        _, _, b_eff = clf.classifier_forward(x)
        assert (b_eff == 0).all()

    def test_3layer_shape(self):
        clf = BcosClassifier(100, 32, 2, num_layers=3)
        x = torch.randn(4, 100)
        logits, W_eff, b_eff = clf.classifier_forward(x)
        assert logits.shape == (4, 2)
        assert W_eff.shape == (4, 2, 100)

    def test_forward_matches_classifier_forward(self):
        """forward() should produce same logits as classifier_forward()."""
        torch.manual_seed(42)
        clf = BcosClassifier(100, 32, 2, num_layers=2)
        x = torch.randn(4, 100)

        logits_fwd = clf(x)
        logits_clf, _, _ = clf.classifier_forward(x)
        assert torch.allclose(logits_fwd, logits_clf, atol=1e-6)


class TestDLAWithBcos:
    """Test that DLA works end-to-end with B-cos classifier."""

    def test_dla_attribution(self):
        """compute_attribution_tensor works with B-cos W_eff."""
        from splade.mechanistic.attribution import compute_attribution_tensor

        torch.manual_seed(42)
        clf = BcosClassifier(100, 32, 2, num_layers=2)
        sparse_vector = torch.randn(4, 100).abs()  # Sparse vectors are non-negative
        target_classes = torch.zeros(4, dtype=torch.long)

        logits, W_eff, b_eff = clf.classifier_forward(sparse_vector)
        attr = compute_attribution_tensor(sparse_vector, W_eff, target_classes)

        # DLA identity: sum(attr) + b_eff[target] == logit[target]
        target_logits = logits.gather(1, target_classes.unsqueeze(1)).squeeze(1)
        target_bias = b_eff.gather(1, target_classes.unsqueeze(1)).squeeze(1)
        reconstructed = attr.sum(dim=-1) + target_bias

        assert torch.allclose(target_logits, reconstructed, atol=1e-4), \
            f"DLA error: {(target_logits - reconstructed).abs().max().item()}"
