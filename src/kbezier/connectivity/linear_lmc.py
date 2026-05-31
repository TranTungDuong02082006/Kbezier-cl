"""
Linear mode connectivity baseline: φ(θ) = (1-θ) w₀ + θ w₁.

Used to demonstrate that linear paths collapse as T increases,
while Bézier curves maintain low loss barriers.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class LinearPath(nn.Module):
    """Linear interpolation between two parameter snapshots (no control point)."""

    def __init__(
        self,
        w0: Dict[str, torch.Tensor],
        w1: Dict[str, torch.Tensor],
    ):
        super().__init__()
        self._keys = sorted(w0.keys())
        for name in self._keys:
            buf_0 = "w0_" + name.replace(".", "__")
            buf_1 = "w1_" + name.replace(".", "__")
            self.register_buffer(buf_0, w0[name].detach().clone())
            self.register_buffer(buf_1, w1[name].detach().clone())

    def interpolate(self, theta: float) -> Dict[str, torch.Tensor]:
        """φ(θ) = (1-θ) w₀ + θ w₁."""
        result = {}
        for name in self._keys:
            w0 = getattr(self, f"w0_{name.replace('.', '__')}")
            w1 = getattr(self, f"w1_{name.replace('.', '__')}")
            result[name] = (1 - theta) * w0 + theta * w1
        return result

    def velocity(self, theta: float) -> Dict[str, torch.Tensor]:
        """φ̇(θ) = w₁ - w₀ (constant)."""
        result = {}
        for name in self._keys:
            w0 = getattr(self, f"w0_{name.replace('.', '__')}")
            w1 = getattr(self, f"w1_{name.replace('.', '__')}")
            result[name] = w1 - w0
        return result
