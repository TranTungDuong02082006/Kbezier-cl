"""
Quadratic Bézier curve in parameter space.

φ(θ) = (1-θ)² w₀ + 2θ(1-θ) wᵥ + θ² w₁,  θ ∈ [0,1]
φ̇(θ) = 2(1-θ)(wᵥ - w₀) + 2θ(w₁ - wᵥ)

- w₀, w₁: anchors (detached, fixed endpoints)
- wᵥ: control point (optimizable nn.ParameterDict)
- Anchor management: save_anchors/load_anchors for disk offload

CRITICAL: interpolate(θ) returns a parameter dict that is differentiable
w.r.t. wᵥ. This is essential for path_loss gradient flow via
torch.func.functional_call.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn


class BezierCurve(nn.Module):
    """
    Quadratic Bézier curve connecting two parameter snapshots.

    Endpoints are fixed (detached). Control point wᵥ is optimizable.
    Only the CURRENT pair's anchors live in GPU memory; previous
    pairs are disk-offloaded.
    """

    def __init__(
        self,
        w0: Dict[str, torch.Tensor],
        w1: Dict[str, torch.Tensor],
        init: str = "midpoint",
    ):
        """
        Args:
            w0: Start anchor parameters {name: tensor} (detached copy).
            w1: End anchor parameters {name: tensor} (detached copy).
            init: Control point initialization. "midpoint" or "random".
        """
        super().__init__()

        # Store anchors as buffers (not optimized)
        self._w0_keys = sorted(w0.keys())
        for name in self._w0_keys:
            # Register as buffer with sanitized name
            buf_name_0 = "w0_" + name.replace(".", "__")
            buf_name_1 = "w1_" + name.replace(".", "__")
            self.register_buffer(buf_name_0, w0[name].detach().clone())
            self.register_buffer(buf_name_1, w1[name].detach().clone())

        # Control point wᵥ as ParameterDict (optimized)
        self.w_v = nn.ParameterDict()
        for name in self._w0_keys:
            safe_name = name.replace(".", "__")
            if init == "midpoint":
                init_val = (w0[name].detach() + w1[name].detach()) / 2.0
            elif init == "random":
                init_val = w0[name].detach() + torch.randn_like(w0[name]) * 0.01
            else:
                raise ValueError(f"Unknown init: {init}")
            self.w_v[safe_name] = nn.Parameter(init_val)

    def _get_anchor(self, prefix: str, name: str) -> torch.Tensor:
        """Get anchor tensor by original parameter name."""
        buf_name = f"{prefix}_{name.replace('.', '__')}"
        return getattr(self, buf_name)

    def interpolate(self, theta: float) -> Dict[str, torch.Tensor]:
        """
        Compute φ(θ) = (1-θ)² w₀ + 2θ(1-θ) wᵥ + θ² w₁.

        Returns a parameter dict DIFFERENTIABLE w.r.t. self.w_v.
        Use with torch.func.functional_call for gradient flow.

        Args:
            theta: Interpolation parameter in [0, 1].

        Returns:
            Dict mapping original param names → interpolated tensors.
        """
        result = {}
        t = theta
        c0 = (1 - t) ** 2
        c_v = 2 * t * (1 - t)
        c1 = t ** 2

        for name in self._w0_keys:
            safe_name = name.replace(".", "__")
            w0 = self._get_anchor("w0", name)
            w1 = self._get_anchor("w1", name)
            wv = self.w_v[safe_name]

            # This expression IS differentiable w.r.t. wv (nn.Parameter)
            result[name] = c0 * w0 + c_v * wv + c1 * w1

        return result

    def velocity(self, theta: float) -> Dict[str, torch.Tensor]:
        """
        Compute φ̇(θ) = 2(1-θ)(wᵥ - w₀) + 2θ(w₁ - wᵥ).

        Differentiable w.r.t. self.w_v.
        Used for Fisher-geodesic energy term.

        Args:
            theta: Interpolation parameter in [0, 1].

        Returns:
            Dict mapping original param names → velocity tensors.
        """
        result = {}
        t = theta

        for name in self._w0_keys:
            safe_name = name.replace(".", "__")
            w0 = self._get_anchor("w0", name)
            w1 = self._get_anchor("w1", name)
            wv = self.w_v[safe_name]

            result[name] = 2 * (1 - t) * (wv - w0) + 2 * t * (w1 - wv)

        return result

    def save_anchors(self, path: str | Path) -> None:
        """Save anchors and control point to disk for memory offloading."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "w0": {name: self._get_anchor("w0", name).cpu() for name in self._w0_keys},
            "w1": {name: self._get_anchor("w1", name).cpu() for name in self._w0_keys},
            "w_v": {name: self.w_v[name.replace(".", "__")].data.cpu() for name in self._w0_keys},
            "keys": self._w0_keys,
        }
        torch.save(state, path)

    @classmethod
    def load_anchors(cls, path: str | Path, device: str = "cpu") -> "BezierCurve":
        """Load a BezierCurve from disk."""
        state = torch.load(path, map_location=device, weights_only=False)
        curve = cls(state["w0"], state["w1"], init="midpoint")
        # Restore control point values
        for name in state["keys"]:
            safe_name = name.replace(".", "__")
            curve.w_v[safe_name].data.copy_(state["w_v"][name])
        return curve

    def update_endpoint(self, w1_new: Dict[str, torch.Tensor]) -> None:
        """
        Update the end anchor w₁ (e.g., after training completes on a task).

        This is called when training finishes and we know the final weights.
        """
        for name in self._w0_keys:
            if name in w1_new:
                buf_name = f"w1_{name.replace('.', '__')}"
                getattr(self, buf_name).copy_(w1_new[name].detach())
