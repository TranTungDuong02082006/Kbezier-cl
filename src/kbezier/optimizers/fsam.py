"""
F-SAM: Fisher SAM (Kwon et al., 2021) — baseline.

Closest competitor to R-SAM. Also uses Fisher to shape SAM neighborhood,
but with key differences:
- Uses DIAGONAL Fisher only (not Kronecker-factored)
- Uses CURRENT TASK Fisher only (no mixture with old Fisher)
- No Bézier connectivity

Direct comparison highlights K-Bézier's contributions:
mixture Fisher + full K-FAC structure + path connectivity.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from kbezier.metrics.diagonal import DiagonalFisher
from kbezier.optimizers.base_sam import SAMOptimizer


class FSAMOptimizer(SAMOptimizer):
    """
    Fisher SAM (Kwon et al., 2021).

    Perturbation scaled by diagonal Fisher of current task:
        ε = ρ · (F_diag * g) / ‖F_diag * g‖₂

    Note: This uses Fisher to SCALE the gradient (element-wise product),
    not to define a Riemannian metric (which would use F⁻¹g).
    The original F-SAM paper normalizes by the Fisher-weighted norm.
    """

    def __init__(
        self,
        model: nn.Module,
        base_optimizer: torch.optim.Optimizer,
        rho: float = 0.05,
        damping: float = 1e-3,
    ):
        super().__init__(model, base_optimizer, rho)
        self.fisher = DiagonalFisher(damping=damping)
        self._fisher_initialized = False

    def update_fisher(self, data_loader, criterion=None, n_samples=None):
        """Estimate diagonal Fisher from current task data."""
        self.fisher.accumulate(self.model, data_loader, criterion, n_samples)
        self._fisher_initialized = True

    def _compute_perturbation(
        self, grad_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        F-SAM perturbation: Fisher-weighted gradient direction.

        If Fisher not yet estimated, falls back to standard SAM.
        """
        if not self._fisher_initialized:
            return super()._compute_perturbation(grad_dict)

        # Fisher-weighted gradient: F_diag * g (element-wise)
        weighted = {}
        for name, grad in grad_dict.items():
            if name in self.fisher.diag:
                weighted[name] = self.fisher.diag[name] * grad
            else:
                weighted[name] = grad

        # Normalize
        norm = torch.sqrt(
            sum(w.norm() ** 2 for w in weighted.values())
        ).clamp(min=1e-12)

        perturbation = {
            name: self.rho * weighted[name] / norm
            for name in weighted
        }
        return perturbation
