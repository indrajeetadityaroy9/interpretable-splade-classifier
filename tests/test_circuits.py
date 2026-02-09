"""Tests for splade.circuits module — core, geco, losses."""

import math

import pytest
import torch

from splade.circuits.core import CircuitState, circuit_mask
from splade.circuits.geco import GECOController


class TestCircuitState:
    def test_unpack_as_4_tuple(self):
        logits = torch.randn(2, 3)
        sparse = torch.randn(2, 10)
        w_eff = torch.randn(2, 3, 10)
        b_eff = torch.randn(2, 3)
        state = CircuitState(logits, sparse, w_eff, b_eff)

        l, s, w, b = state
        assert torch.equal(l, logits)
        assert torch.equal(s, sparse)
        assert torch.equal(w, w_eff)
        assert torch.equal(b, b_eff)

    def test_named_access(self):
        state = CircuitState(
            logits=torch.randn(1, 2),
            sparse_vector=torch.randn(1, 5),
            W_eff=torch.randn(1, 2, 5),
            b_eff=torch.randn(1, 2),
        )
        assert state.logits.shape == (1, 2)
        assert state.sparse_vector.shape == (1, 5)
        assert state.W_eff.shape == (1, 2, 5)

    def test_is_tuple_subclass(self):
        state = CircuitState(
            torch.randn(1, 2), torch.randn(1, 5),
            torch.randn(1, 2, 5), torch.randn(1, 2),
        )
        assert isinstance(state, tuple)
        assert len(state) == 4


class TestCircuitMask:
    def test_hard_mask_is_nearly_binary(self):
        attr = torch.rand(4, 100)
        mask = circuit_mask(attr, circuit_fraction=0.1, temperature=1e6)
        # At temperature 1e6, mask values should be 0, 0.5 (at threshold), or 1
        assert ((mask < 0.01) | ((mask > 0.49) & (mask < 0.51)) | (mask > 0.99)).all()

    def test_soft_mask_is_smooth(self):
        attr = torch.rand(4, 100)
        mask = circuit_mask(attr, circuit_fraction=0.1, temperature=10.0)
        # At temperature 10, there should be some intermediate values
        intermediate = (mask > 0.01) & (mask < 0.99)
        assert intermediate.any()

    def test_correct_fraction_retained(self):
        attr = torch.rand(4, 1000)
        mask = circuit_mask(attr, circuit_fraction=0.1, temperature=1e6)
        # ~10% of dims should be active (allowing some tolerance)
        active_frac = (mask > 0.5).float().mean(dim=-1)
        assert (active_frac > 0.08).all() and (active_frac < 0.12).all()

    def test_output_shape_matches_input(self):
        attr = torch.rand(8, 200)
        mask = circuit_mask(attr, circuit_fraction=0.2, temperature=50.0)
        assert mask.shape == attr.shape

    def test_fraction_one_retains_all(self):
        attr = torch.rand(2, 50)
        mask = circuit_mask(attr, circuit_fraction=1.0, temperature=1e6)
        # With fraction=1.0, all values should be >= 0.5 (min element sits at threshold)
        assert (mask >= 0.5).all()


class TestGECOController:
    def test_warmup_records_ce(self):
        geco = GECOController()
        for i in range(10):
            geco.record_warmup_ce(1.0 - i * 0.05)
        assert len(geco._warmup_ces) == 10

    def test_finalize_warmup_sets_tau_from_percentile(self):
        """tau_ce should be set from 25th percentile of warmup CE values."""
        geco = GECOController()
        # Feed 100 values; finalize uses last 50: [0.55, 0.559, ..., 0.991]
        for i in range(100):
            geco.record_warmup_ce(0.1 + 0.009 * i)
        tau = geco.finalize_warmup()
        # 25th percentile of last 50 values ≈ 0.66
        assert 0.5 < tau < 0.8

    def test_finalize_warmup_constant_input(self):
        """When all warmup CE values are identical, tau = that value."""
        geco = GECOController()
        for _ in range(60):
            geco.record_warmup_ce(0.5)
        tau = geco.finalize_warmup()
        assert abs(tau - 0.5) < 1e-6

    def test_lambda_increases_when_ce_above_tau(self):
        geco = GECOController()
        for _ in range(10):
            geco.record_warmup_ce(0.5)
        geco.finalize_warmup()

        initial_lambda = geco.lambda_ce
        # CE = 2.0 >> tau, so lambda should increase
        ce = torch.tensor(2.0)
        obj = torch.tensor(0.1)
        geco.compute_loss(ce, obj)
        assert geco.lambda_ce > initial_lambda

    def test_lambda_decreases_when_ce_below_tau(self):
        geco = GECOController()
        for _ in range(10):
            geco.record_warmup_ce(0.5)
        geco.finalize_warmup()

        initial_lambda = geco.lambda_ce
        # CE = 0.01 << tau, so lambda should decrease
        ce = torch.tensor(0.01)
        obj = torch.tensor(0.1)
        geco.compute_loss(ce, obj)
        assert geco.lambda_ce < initial_lambda

    def test_compute_loss_returns_lagrangian(self):
        geco = GECOController()
        for _ in range(10):
            geco.record_warmup_ce(0.5)
        geco.finalize_warmup()

        ce = torch.tensor(2.0, requires_grad=True)
        obj = torch.tensor(3.0, requires_grad=True)
        loss = geco.compute_loss(ce, obj)
        # Loss = obj + lambda_updated * ce. Lambda increases because CE > tau.
        # The returned value should be > obj (3.0) because lambda * ce > 0
        assert loss.item() > 3.0
        # And the loss should have gradients flowing through both inputs
        loss.backward()
        assert ce.grad is not None
        assert obj.grad is not None

    def test_lambda_always_positive(self):
        geco = GECOController()
        for _ in range(10):
            geco.record_warmup_ce(0.5)
        geco.finalize_warmup()

        # Drive lambda down aggressively
        for _ in range(100):
            geco.compute_loss(torch.tensor(0.01), torch.tensor(0.1))
        assert geco.lambda_ce > 0

    def test_no_constructor_args_needed(self):
        """GECOController should work with zero arguments (fully adaptive)."""
        geco = GECOController()
        assert geco.lambda_ce == 1.0
        assert geco.tau_ce is None

    def test_lambda_bounded_by_clamp(self):
        """log_lambda should be clamped to [-5, 5], bounding lambda."""
        from splade.circuits.geco import _LOG_LAMBDA_CLAMP
        geco = GECOController()
        for _ in range(10):
            geco.record_warmup_ce(0.5)
        geco.finalize_warmup()

        # Drive lambda up with many high-CE steps
        for _ in range(1000):
            geco.compute_loss(torch.tensor(10.0), torch.tensor(0.1))
        # Lambda should be bounded by exp(5) ≈ 148.4
        assert geco.lambda_ce <= math.exp(_LOG_LAMBDA_CLAMP) + 1e-6
        assert geco._log_lambda <= _LOG_LAMBDA_CLAMP + 1e-6

        # Drive lambda down with many low-CE steps
        for _ in range(1000):
            geco.compute_loss(torch.tensor(0.001), torch.tensor(0.1))
        # Lambda should be bounded by exp(-5) ≈ 0.007
        assert geco.lambda_ce >= math.exp(-_LOG_LAMBDA_CLAMP) - 1e-6
        assert geco._log_lambda >= -_LOG_LAMBDA_CLAMP - 1e-6

    def test_lambda_stable_moderate_violation(self):
        """With small positive constraint, lambda should grow moderately."""
        geco = GECOController()
        for _ in range(10):
            geco.record_warmup_ce(0.5)
        geco.finalize_warmup()

        # CE slightly above tau (constraint ≈ 0.1)
        for _ in range(50):
            geco.compute_loss(torch.tensor(0.6), torch.tensor(0.1))
        # Lambda should increase but stay reasonable (not millions)
        assert geco.lambda_ce > 1.0  # increased
        assert geco.lambda_ce < 200  # bounded


class TestHoyerSparsity:
    def test_one_hot_is_maximally_sparse(self):
        """Hoyer sparsity of a one-hot vector should be close to 1.0."""
        from splade.circuits.losses import compute_sharpness_loss

        n = 100
        sparse = torch.zeros(1, n)
        sparse[0, 0] = 1.0
        w_eff = torch.eye(n).unsqueeze(0)  # identity, so attr = sparse
        labels = torch.tensor([0])

        loss = compute_sharpness_loss(sparse, w_eff, labels)
        # Hoyer of one-hot ≈ 1.0, so loss = 1 - 1.0 ≈ 0.0
        assert loss.item() < 0.05

    def test_uniform_is_not_sparse(self):
        """Hoyer sparsity of a uniform vector should be close to 0.0."""
        from splade.circuits.losses import compute_sharpness_loss

        n = 100
        sparse = torch.ones(1, n)
        # W_eff[b, c, j]: need W_eff[:, 0, :] = ones so attr = sparse * W_eff = ones
        w_eff = torch.ones(1, 1, n)
        labels = torch.tensor([0])

        loss = compute_sharpness_loss(sparse, w_eff, labels)
        # Hoyer of uniform ≈ 0.0, so loss = 1 - 0.0 ≈ 1.0
        assert loss.item() > 0.9


class TestGradientCentralization:
    def test_centralizes_weight_gradients(self):
        from splade.training.optim import _gradient_centralization

        model = torch.nn.Linear(10, 5)
        # Create a dummy loss and backward
        x = torch.randn(3, 10)
        loss = model(x).sum()
        loss.backward()

        _gradient_centralization(model)

        # After centralization, gradient mean across non-output dims should be ~0
        grad = model.weight.grad
        mean_per_output = grad.mean(dim=1)
        assert torch.allclose(mean_per_output, torch.zeros_like(mean_per_output), atol=1e-6)

    def test_skips_1d_params(self):
        from splade.training.optim import _gradient_centralization

        model = torch.nn.Linear(10, 5)
        x = torch.randn(3, 10)
        loss = model(x).sum()
        loss.backward()

        bias_grad_before = model.bias.grad.clone()
        _gradient_centralization(model)
        # Bias (1D) should be unchanged
        assert torch.equal(model.bias.grad, bias_grad_before)


class TestRemovedConstants:
    """Verify that manually-tuned constants have been removed from constants.py."""

    def test_no_agc_constants(self):
        from splade.training import constants
        assert not hasattr(constants, "AGC_CLIP_FACTOR")
        assert not hasattr(constants, "AGC_EPS")

    def test_no_df_alpha_beta(self):
        from splade.training import constants
        assert not hasattr(constants, "DF_ALPHA")
        assert not hasattr(constants, "DF_BETA")

    def test_no_circuit_loss_weights(self):
        from splade.training import constants
        assert not hasattr(constants, "CC_WEIGHT")
        assert not hasattr(constants, "SEP_WEIGHT")
        assert not hasattr(constants, "SHARP_WEIGHT")

    def test_no_geco_tunable_params(self):
        from splade.training import constants
        assert not hasattr(constants, "GECO_ETA")
        assert not hasattr(constants, "GECO_EMA_DECAY")
        assert not hasattr(constants, "GECO_TAU_MARGIN")

    def test_no_centroid_momentum(self):
        from splade.training import constants
        assert not hasattr(constants, "CENTROID_MOMENTUM")

    def test_no_df_momentum(self):
        from splade.training import constants
        assert not hasattr(constants, "DF_MOMENTUM")

    def test_no_circuit_fraction_in_constants(self):
        from splade.training import constants
        assert not hasattr(constants, "CIRCUIT_FRACTION")
        assert not hasattr(constants, "CIRCUIT_WARMUP_FRACTION")
