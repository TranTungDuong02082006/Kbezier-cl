"""
Tests for K-FAC layer hooks.

Verifies:
1. Linear hook: A shape = (d_in+1)×(d_in+1), B shape = d_out×d_out (with bias)
2. Conv hook: A shape = (C_in·k²+1)×(C_in·k²+1) after unfold
3. Hook cleanup: no hooks remain after remove()
"""

import torch
import torch.nn as nn
import pytest

from kbezier.metrics.layer_hooks import KFACHookManager


class TestLinearHookShapes:
    """Test that hooks produce correct shapes for Linear layers."""

    def test_linear_with_bias(self):
        """A should be (d_in+1)×(d_in+1), B should be d_out×d_out with bias absorption."""
        model = nn.Sequential(
            nn.Linear(5, 3, bias=True),
            nn.ReLU(),
            nn.Linear(3, 2, bias=True),
        )

        manager = KFACHookManager(model, absorb_bias=True)
        manager.register_hooks()

        x = torch.randn(10, 5)
        y = torch.randint(0, 2, (10,))

        try:
            model.zero_grad()
            output = model(x)
            loss = nn.CrossEntropyLoss()(output, y)
            loss.backward()
            manager.accumulate_step()

            factors = manager.finalize()

            # Layer 0: Linear(5, 3) → A: (5+1, 5+1), B: (3, 3)
            layer_names = list(factors.keys())
            assert len(layer_names) == 2, f"Expected 2 layers, got {len(layer_names)}"

            A0, B0 = factors[layer_names[0]]
            assert A0.shape == (6, 6), f"A0 shape {A0.shape}, expected (6,6)"
            assert B0.shape == (3, 3), f"B0 shape {B0.shape}, expected (3,3)"

            # Layer 1: Linear(3, 2) → A: (3+1, 3+1), B: (2, 2)
            A1, B1 = factors[layer_names[1]]
            assert A1.shape == (4, 4), f"A1 shape {A1.shape}, expected (4,4)"
            assert B1.shape == (2, 2), f"B1 shape {B1.shape}, expected (2,2)"
        finally:
            manager.remove_hooks()

    def test_linear_no_bias(self):
        """Without bias, A should be (d_in, d_in)."""
        model = nn.Sequential(
            nn.Linear(5, 3, bias=False),
            nn.ReLU(),
            nn.Linear(3, 2, bias=False),
        )

        manager = KFACHookManager(model, absorb_bias=True)
        manager.register_hooks()

        x = torch.randn(10, 5)
        y = torch.randint(0, 2, (10,))

        try:
            model.zero_grad()
            output = model(x)
            loss = nn.CrossEntropyLoss()(output, y)
            loss.backward()
            manager.accumulate_step()

            factors = manager.finalize()
            layer_names = list(factors.keys())

            A0, B0 = factors[layer_names[0]]
            assert A0.shape == (5, 5), f"A0 shape {A0.shape}, expected (5,5)"
        finally:
            manager.remove_hooks()


class TestConvHookUnfold:
    """Test that Conv2d hooks correctly unfold inputs."""

    def test_conv_shapes(self):
        """A should be (C_in·k²+1)×(C_in·k²+1) for Conv2d with bias."""
        model = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(8, 4, bias=True),
        )

        manager = KFACHookManager(model, absorb_bias=True)
        manager.register_hooks()

        x = torch.randn(10, 3, 8, 8)
        y = torch.randint(0, 4, (10,))

        try:
            model.zero_grad()
            output = model(x)
            loss = nn.CrossEntropyLoss()(output, y)
            loss.backward()
            manager.accumulate_step()

            factors = manager.finalize()
            layer_names = list(factors.keys())

            # Conv2d(3, 8, 3): C_in*k² = 3*9 = 27, +1 for bias = 28
            conv_name = [n for n in layer_names if "0" in n][0]
            A_conv, B_conv = factors[conv_name]
            assert A_conv.shape == (28, 28), f"A_conv shape {A_conv.shape}, expected (28,28)"
            assert B_conv.shape == (8, 8), f"B_conv shape {B_conv.shape}, expected (8,8)"
        finally:
            manager.remove_hooks()


class TestHookCleanup:
    """Test that hooks are properly removed."""

    def test_no_hooks_after_remove(self):
        """After remove_hooks(), no hooks should remain on the model."""
        model = nn.Linear(5, 3)
        manager = KFACHookManager(model)
        manager.register_hooks()

        # Count hooks
        n_hooks_before = len(model._forward_hooks) + len(model._backward_hooks)
        assert n_hooks_before > 0, "Hooks should be registered"

        manager.remove_hooks()

        n_hooks_after = len(model._forward_hooks) + len(model._backward_hooks)
        assert n_hooks_after == 0, f"Hooks not cleaned up: {n_hooks_after} remaining"

    def test_repeated_register_remove(self):
        """Multiple register/remove cycles should not leak."""
        model = nn.Linear(5, 3)
        manager = KFACHookManager(model)

        for _ in range(5):
            manager.register_hooks()
            manager.remove_hooks()

        n_hooks = len(model._forward_hooks) + len(model._backward_hooks)
        assert n_hooks == 0, f"Hook leak after repeated cycles: {n_hooks}"
