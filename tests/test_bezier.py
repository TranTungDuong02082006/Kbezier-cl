"""
Tests for Bézier curve and path loss.

Verifies:
1. φ(0) = w₀, φ(1) = w₁ (endpoint correctness)
2. φ̇(θ) matches numerical derivative
3. path_loss gradient w.r.t. w_v matches finite difference — THE critical test
4. functional_call gradient flow to w_v is intact
"""

import torch
import torch.nn as nn
import pytest
from torch.func import functional_call

from kbezier.connectivity.bezier import BezierCurve
from kbezier.connectivity.path_loss import PathLoss


def _make_tiny_model_and_params():
    """Create a tiny model and two random parameter snapshots."""
    model = nn.Sequential(
        nn.Linear(4, 3, bias=False),
        nn.ReLU(),
        nn.Linear(3, 2, bias=False),
    )

    w0 = {name: torch.randn_like(p) for name, p in model.named_parameters()}
    w1 = {name: torch.randn_like(p) for name, p in model.named_parameters()}

    return model, w0, w1


class TestEndpoints:
    """φ(0) = w₀, φ(1) = w₁."""

    def test_phi_0_equals_w0(self):
        """φ(0) should exactly equal the start anchor."""
        torch.manual_seed(42)
        _, w0, w1 = _make_tiny_model_and_params()
        curve = BezierCurve(w0, w1)

        phi_0 = curve.interpolate(0.0)
        for name in w0:
            torch.testing.assert_close(
                phi_0[name], w0[name],
                msg=f"φ(0) != w₀ for parameter {name}"
            )

    def test_phi_1_equals_w1(self):
        """φ(1) should exactly equal the end anchor."""
        torch.manual_seed(42)
        _, w0, w1 = _make_tiny_model_and_params()
        curve = BezierCurve(w0, w1)

        phi_1 = curve.interpolate(1.0)
        for name in w1:
            torch.testing.assert_close(
                phi_1[name], w1[name],
                msg=f"φ(1) != w₁ for parameter {name}"
            )


class TestVelocity:
    """φ̇(θ) matches numerical derivative."""

    def test_velocity_numerical(self):
        """Analytical velocity should match (φ(θ+δ) - φ(θ-δ)) / (2δ)."""
        torch.manual_seed(42)
        _, w0, w1 = _make_tiny_model_and_params()
        curve = BezierCurve(w0, w1)

        theta = 0.4
        delta = 1e-4

        # Analytical velocity
        v_analytical = curve.velocity(theta)

        # Numerical velocity
        phi_plus = curve.interpolate(theta + delta)
        phi_minus = curve.interpolate(theta - delta)

        for name in v_analytical:
            v_numerical = (phi_plus[name] - phi_minus[name]) / (2 * delta)
            torch.testing.assert_close(
                v_analytical[name], v_numerical,
                atol=1e-3, rtol=1e-3,
                msg=f"Velocity mismatch at θ={theta} for {name}"
            )


class TestPathLossGradient:
    """
    THE CRITICAL TEST: path_loss gradient w.r.t. w_v matches finite difference.

    This catches broken gradient flow through functional_call that
    endpoint tests DO NOT catch.
    """

    def test_path_loss_gradient_vs_finite_diff(self):
        """
        Verify ∂L_path/∂w_v via autograd matches finite difference.

        Perturb each component of w_v by ±δ, measure loss change.
        If functional_call gradient flow is broken, this FAILS.
        """
        torch.manual_seed(42)
        model, w0, w1 = _make_tiny_model_and_params()

        # Put actual weights in model (for functional_call to use as architecture)
        for name, param in model.named_parameters():
            param.data.copy_(w0[name])

        curve = BezierCurve(w0, w1)

        # Create replay data
        x = torch.randn(20, 4)
        y = torch.randint(0, 2, (20,))

        # Use a fixed theta for deterministic testing
        theta = 0.5

        # Compute analytical gradient
        curve.zero_grad()
        phi_theta = curve.interpolate(theta)
        logits = functional_call(model, phi_theta, (x,))
        loss = nn.CrossEntropyLoss()(logits, y)
        loss.backward()

        # Check that gradients exist for w_v parameters
        for safe_name, param in curve.w_v.items():
            assert param.grad is not None, (
                f"Gradient is None for w_v[{safe_name}] — "
                "functional_call gradient flow is broken!"
            )

        # Finite difference check on a subset of parameters
        delta = 1e-4
        for safe_name, param in curve.w_v.items():
            analytical_grad = param.grad.data.clone()

            # Only check first few elements to save time
            n_check = min(3, param.numel())
            flat_param = param.data.flatten()
            flat_grad = analytical_grad.flatten()

            for idx in range(n_check):
                # +δ
                flat_param[idx] += delta
                phi_plus = curve.interpolate(theta)
                logits_plus = functional_call(model, phi_plus, (x,))
                loss_plus = nn.CrossEntropyLoss()(logits_plus, y).item()

                # -2δ (back to -δ from original)
                flat_param[idx] -= 2 * delta
                phi_minus = curve.interpolate(theta)
                logits_minus = functional_call(model, phi_minus, (x,))
                loss_minus = nn.CrossEntropyLoss()(logits_minus, y).item()

                # Restore
                flat_param[idx] += delta

                fd_grad = (loss_plus - loss_minus) / (2 * delta)
                an_grad = flat_grad[idx].item()

                # Check relative error
                if abs(fd_grad) > 1e-6:
                    rel_error = abs(an_grad - fd_grad) / abs(fd_grad)
                    assert rel_error < 0.1, (
                        f"Gradient mismatch for w_v[{safe_name}][{idx}]: "
                        f"analytical={an_grad:.6f}, fd={fd_grad:.6f}, "
                        f"rel_error={rel_error:.4f}"
                    )


class TestFunctionalCallGradientFlow:
    """Verify gradient flow through torch.func.functional_call."""

    def test_gradients_not_none(self):
        """functional_call should produce non-None gradients for interpolated params."""
        torch.manual_seed(42)
        model, w0, w1 = _make_tiny_model_and_params()

        for name, param in model.named_parameters():
            param.data.copy_(w0[name])

        curve = BezierCurve(w0, w1)

        x = torch.randn(5, 4)
        y = torch.randint(0, 2, (5,))

        curve.zero_grad()
        phi = curve.interpolate(0.5)
        logits = functional_call(model, phi, (x,))
        loss = nn.CrossEntropyLoss()(logits, y)
        loss.backward()

        n_with_grad = sum(
            1 for p in curve.w_v.parameters() if p.grad is not None
        )
        n_total = sum(1 for _ in curve.w_v.parameters())

        assert n_with_grad == n_total, (
            f"Only {n_with_grad}/{n_total} w_v parameters have gradients. "
            "Gradient flow through functional_call is broken."
        )
