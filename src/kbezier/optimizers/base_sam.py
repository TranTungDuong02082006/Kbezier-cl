"""
SAM (Sharpness-Aware Minimization) — Euclidean baseline.

Two-step process:
1. Compute perturbation ε = ρ g/‖g‖₂ (worst-case in Euclidean ball)
2. Compute gradient at perturbed point w+ε, update w

Subclasses (R-SAM) override only _compute_perturbation() to change
the geometry from Euclidean ball to Riemannian ellipsoid.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn


class SAMOptimizer:
    """
    Sharpness-Aware Minimization with pluggable geometry.

    The base class implements Euclidean SAM. R-SAM overrides
    _compute_perturbation to use Riemannian ellipsoid.
    """

    def __init__(
        self,
        model: nn.Module,
        base_optimizer: torch.optim.Optimizer,
        rho: float = 0.05,
    ):
        """
        Args:
            model: Network to optimize.
            base_optimizer: Underlying optimizer (e.g., SGD, Adam).
            rho: Perturbation radius.
        """
        self.model = model
        self.base_optimizer = base_optimizer
        self.rho = rho

        # Compute tracking
        self.forward_count = 0
        self.backward_count = 0

    def step(self, loss_fn: Callable[[], torch.Tensor]) -> float:
        """
        Perform one SAM step.

        Args:
            loss_fn: Callable that returns the loss (must support calling twice).
                     Typically: lambda: criterion(model(x), y)

        Returns:
            Loss value (float).
        """
        # ── Step 1: Compute gradient at current point ──
        self.base_optimizer.zero_grad()
        loss = loss_fn()
        loss.backward()
        self.backward_count += 1
        loss_val = loss.item()

        # Collect gradients
        grad_dict = self._collect_gradients()

        # Compute perturbation
        perturbation = self._compute_perturbation(grad_dict)

        # Apply perturbation: w → w + ε
        self._apply_perturbation(perturbation, sign=+1)

        # ── Step 2: Compute gradient at perturbed point ──
        self.base_optimizer.zero_grad()
        loss_perturbed = loss_fn()
        loss_perturbed.backward()
        self.backward_count += 1
        self.forward_count += 1  # second forward pass

        # Revert perturbation: w + ε → w
        self._apply_perturbation(perturbation, sign=-1)

        # ── Step 3: Update with gradient at perturbed point ──
        self.base_optimizer.step()

        return loss_val

    def _collect_gradients(self) -> Dict[str, torch.Tensor]:
        """Collect current gradients from model parameters."""
        grad_dict = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_dict[name] = param.grad.data.clone()
        return grad_dict

    def _compute_perturbation(
        self, grad_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute perturbation direction.

        Base class: Euclidean SAM
            ε = ρ · g / ‖g‖₂

        Override in R-SAM for Riemannian geometry.
        """
        # Compute global L2 norm
        grad_norm = torch.sqrt(
            sum(g.norm() ** 2 for g in grad_dict.values())
        ).clamp(min=1e-12)

        # Scale each gradient component
        perturbation = {
            name: self.rho * grad / grad_norm
            for name, grad in grad_dict.items()
        }
        return perturbation

    def _apply_perturbation(
        self, perturbation: Dict[str, torch.Tensor], sign: int
    ) -> None:
        """Add or subtract perturbation from model parameters."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in perturbation:
                    param.add_(perturbation[name], alpha=sign)

    @property
    def param_groups(self):
        return self.base_optimizer.param_groups

    def state_dict(self):
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state):
        self.base_optimizer.load_state_dict(state)
