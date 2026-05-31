"""
Abstract base class for Fisher Information Metric.

Defines the interface that R-SAM and Bézier path connectivity share.
This shared interface is what makes the "unification" real in code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class FisherMetric(ABC):
    """
    Abstract Fisher Information Metric interface.

    All methods (R-SAM, Bézier energy, EWC) call this same interface.
    Implementations: KFACFisher, DiagonalFisher, MixtureFisher.
    """

    @abstractmethod
    def accumulate(self, model: nn.Module, data_loader, criterion=None) -> None:
        """
        Estimate Fisher factors from data.

        Args:
            model: Network to compute Fisher for.
            data_loader: DataLoader for Fisher estimation.
            criterion: Loss function (default: cross-entropy).
        """
        ...

    @abstractmethod
    def inv_mv(self, grad_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute F⁻¹ @ grad via vec-trick (for K-FAC: B̃⁻¹ ∇ Ã⁻¹ per layer).

        Args:
            grad_dict: Dictionary mapping parameter names → gradient tensors.

        Returns:
            Dictionary mapping parameter names → F⁻¹ @ grad tensors.
        """
        ...

    @abstractmethod
    def quad(self, vec_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute vᵀ F v (for Bézier Fisher-geodesic energy).

        Args:
            vec_dict: Dictionary mapping parameter names → vector tensors.

        Returns:
            Scalar tensor vᵀ F v.
        """
        ...

    @abstractmethod
    def top_eigs(self, k: int = 1) -> torch.Tensor:
        """
        Compute top-k eigenvalues of F via power iteration on factors.

        For K-FAC: λ_max(A⊗B) = λ_max(A) · λ_max(B).
        Used for bound computation (§4.5): ρ²/λ ≤ 2τ/λ_max.

        Args:
            k: Number of top eigenvalues.

        Returns:
            Tensor of top-k eigenvalues (descending).
        """
        ...

    @abstractmethod
    def state_dict(self) -> Dict[str, Any]:
        """Serialize Fisher state for checkpointing."""
        ...

    @abstractmethod
    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load Fisher state from checkpoint."""
        ...

    def dual_norm(self, grad_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute ‖g‖_{F⁻¹} = √(gᵀ F⁻¹ g), the dual norm.

        Used by R-SAM for normalization: ε* = ρ F⁻¹g / ‖g‖_{F⁻¹}.
        Computed from inv_mv result: ‖g‖_{F⁻¹}² = Σ_ℓ ⟨g_ℓ, F⁻¹g_ℓ⟩.
        """
        finv_g = self.inv_mv(grad_dict)
        total = torch.tensor(0.0, device=next(iter(grad_dict.values())).device)
        for name in grad_dict:
            if name in finv_g:
                total = total + (grad_dict[name] * finv_g[name]).sum()
        return torch.sqrt(total.clamp(min=1e-16))
