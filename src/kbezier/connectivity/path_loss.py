"""
Bézier path loss: combined loss for mode connectivity.

L_GMC(wᵥ) = E_θ[L_≤t(φ(θ))] + κ · E_θ[φ̇(θ)ᵀ F̄ φ̇(θ)]

Term 1 (path-loss): ensures low loss along the entire curve.
    Forward via torch.func.functional_call — CRITICAL for gradient flow to wᵥ.

Term 2 (Fisher energy): regularizer penalizing paths that disturb old-task distributions.
    F̄ detached, φ̇ differentiable w.r.t. wᵥ → quadratic in wᵥ.

Stochastic discretization: sample 1-2 θ values per step (à la Garipov).
"""

from __future__ import annotations

import random
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.func import functional_call

from kbezier.connectivity.bezier import BezierCurve
from kbezier.metrics.fisher_base import FisherMetric


class PathLoss(nn.Module):
    """
    Combined path loss for Bézier mode connectivity.

    L_GMC = path_loss + κ · fisher_energy
    """

    def __init__(
        self,
        model: nn.Module,
        bezier_curve: BezierCurve,
        fisher_metric: Optional[FisherMetric] = None,
        kappa: float = 0.1,
        n_theta_samples: int = 2,
        criterion: Optional[nn.Module] = None,
    ):
        """
        Args:
            model: Network (used as architecture template for functional_call).
            bezier_curve: Bézier curve connecting two anchors.
            fisher_metric: Fisher metric for energy term. If None, energy term = 0.
            kappa: Weight κ of Fisher-geodesic regularizer.
            n_theta_samples: Number of θ values sampled per step.
            criterion: Loss function (default: CrossEntropyLoss).
        """
        super().__init__()
        self.model = model
        self.bezier_curve = bezier_curve
        self.fisher_metric = fisher_metric
        self.kappa = kappa
        self.n_theta_samples = n_theta_samples
        self.criterion = criterion or nn.CrossEntropyLoss()

    def forward(
        self,
        x_replay: torch.Tensor,
        y_replay: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined path loss.

        Args:
            x_replay: Replay buffer input batch.
            y_replay: Replay buffer label batch.

        Returns:
            (total_loss, info_dict) where info_dict has breakdown.
        """
        path_loss_total = torch.tensor(0.0, device=x_replay.device)
        energy_total = torch.tensor(0.0, device=x_replay.device)

        for _ in range(self.n_theta_samples):
            theta = random.uniform(0.05, 0.95)  # avoid exact endpoints

            # ── Path-loss term: L(φ(θ)) ──
            phi_theta = self.bezier_curve.interpolate(theta)

            # CRITICAL: use functional_call for gradient flow to w_v
            # This does NOT modify model state; it creates a temporary
            # "view" of the model with weights phi_theta.
            logits = functional_call(self.model, phi_theta, (x_replay,))
            loss_at_theta = self.criterion(logits, y_replay)
            path_loss_total = path_loss_total + loss_at_theta

            # ── Fisher energy term: φ̇ᵀ F̄ φ̇ ──
            if self.fisher_metric is not None and self.kappa > 0:
                phi_dot = self.bezier_curve.velocity(theta)
                # F̄ is detached (treated as constant)
                # quad returns vᵀFv which is quadratic in w_v
                energy = self.fisher_metric.quad(phi_dot)
                energy_total = energy_total + energy

        # Average over θ samples
        n = self.n_theta_samples
        path_loss_avg = path_loss_total / n
        energy_avg = energy_total / n

        total = path_loss_avg + self.kappa * energy_avg

        info = {
            "path_loss": path_loss_avg.item(),
            "fisher_energy": energy_avg.item(),
            "total_gmc_loss": total.item(),
        }

        return total, info
