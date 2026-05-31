"""
R-SAM: K-FAC Riemannian SAM (our method, optimizer component).

ε* = ρ · F⁻¹g / ‖g‖_{F⁻¹}

Inherits SAMOptimizer and overrides ONLY _compute_perturbation().
This one-line difference is critical for fair ablation:
the ONLY difference between SAM and R-SAM is the geometry of the
perturbation ball (Euclidean vs Riemannian).

The perturbation ε* is the natural gradient direction scaled to
lie on the boundary of the Riemannian ellipsoid εᵀFε = ρ².
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from kbezier.metrics.fisher_base import FisherMetric
from kbezier.optimizers.base_sam import SAMOptimizer


class RSAMOptimizer(SAMOptimizer):
    """
    Riemannian SAM: perturbation on Fisher ellipsoid.

    ε* = ρ · F⁻¹g / ‖g‖_{F⁻¹}

    where ‖g‖_{F⁻¹} = √(gᵀ F⁻¹ g) is the dual norm.
    """

    def __init__(
        self,
        model: nn.Module,
        base_optimizer: torch.optim.Optimizer,
        fisher_metric: FisherMetric,
        rho: float = 0.05,
    ):
        """
        Args:
            model: Network to optimize.
            base_optimizer: Underlying optimizer.
            fisher_metric: Any FisherMetric instance (K-FAC, diagonal, mixture).
            rho: Perturbation radius on the Riemannian ellipsoid.
        """
        super().__init__(model, base_optimizer, rho)
        self.fisher_metric = fisher_metric

    def _compute_perturbation(
        self, grad_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Override: Riemannian perturbation.

        ε* = ρ · F⁻¹g / ‖g‖_{F⁻¹}

        This is the ONLY difference from SAM — making ablation
        (SAM vs R-SAM) a clean single-variable comparison.
        """
        # F⁻¹ @ g via vec-trick (two small matmuls per layer)
        finv_g = self.fisher_metric.inv_mv(grad_dict)

        # Dual norm: ‖g‖_{F⁻¹} = √(gᵀ F⁻¹ g)
        # Computed as inner product of g and F⁻¹g (already available)
        dual_norm_sq = sum(
            (grad_dict[name] * finv_g[name]).sum()
            for name in grad_dict
            if name in finv_g
        )
        dual_norm = torch.sqrt(dual_norm_sq.clamp(min=1e-16))

        # ε* = ρ · F⁻¹g / ‖g‖_{F⁻¹}
        perturbation = {
            name: self.rho * finv_g[name] / dual_norm
            for name in finv_g
        }

        return perturbation
