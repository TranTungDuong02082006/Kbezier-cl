"""
Diagonal Fisher approximation (EWC-style).

Cheap fallback and ablation baseline. Stores per-parameter diagonal
of the Fisher Information Matrix.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from kbezier.metrics.fisher_base import FisherMetric


class DiagonalFisher(FisherMetric):
    """
    Diagonal Fisher: F ≈ diag(E[g² per parameter]).

    Used by EWC and F-SAM baselines.
    """

    def __init__(self, damping: float = 1e-3):
        self.damping = damping
        # {param_name: diagonal_fisher_values}
        self.diag: Dict[str, torch.Tensor] = {}

    def accumulate(
        self,
        model: nn.Module,
        data_loader,
        criterion=None,
        n_samples: Optional[int] = None,
    ) -> None:
        """Estimate diagonal Fisher from data."""
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        device = next(model.parameters()).device
        model.eval()

        # Initialize accumulators
        fisher_sum = {
            name: torch.zeros_like(param)
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        count = 0
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            model.zero_grad()

            output = model(x)
            loss = criterion(output, y)
            loss.backward()

            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_sum[name] += param.grad.data ** 2 * x.size(0)

            count += x.size(0)
            if n_samples is not None and count >= n_samples:
                break

        # Average
        self.diag = {name: f / max(count, 1) for name, f in fisher_sum.items()}

    def inv_mv(self, grad_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """F⁻¹ @ grad = grad / (diag_F + λ) per element."""
        result = {}
        for name, grad in grad_dict.items():
            if name in self.diag:
                result[name] = grad / (self.diag[name] + self.damping)
            else:
                result[name] = grad
        return result

    def quad(self, vec_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """vᵀ F v = Σ diag_F * v²."""
        total = torch.tensor(0.0)
        for name, vec in vec_dict.items():
            if name in self.diag:
                val = (self.diag[name] * vec ** 2).sum()
                if total.device != val.device:
                    total = total.to(val.device)
                total = total + val
        return total

    def top_eigs(self, k: int = 1) -> torch.Tensor:
        """Top-k values from the diagonal."""
        all_vals = torch.cat([d.flatten() for d in self.diag.values()])
        k = min(k, len(all_vals))
        return torch.topk(all_vals, k).values

    def state_dict(self) -> Dict[str, Any]:
        return {
            "diag": {k: v.cpu() for k, v in self.diag.items()},
            "damping": self.damping,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.damping = state["damping"]
        self.diag = state["diag"]

    def to(self, device: str) -> "DiagonalFisher":
        self.diag = {k: v.to(device) for k, v in self.diag.items()}
        return self
