"""
K-FAC Fisher Information approximation.

F_ℓ ≈ A_ℓ ⊗ B_ℓ per layer (Martens & Grosse, 2015).

Key operations (never materialize full F):
- inv_mv: F⁻¹∇ = vec(B̃⁻¹ ∇_mat Ã⁻¹) — two small matmuls per layer
- quad: vᵀFv = Σ_ℓ tr(Δᵀ B_ℓ Δ A_ℓ) per layer
- top_eigs: λ_max(A⊗B) = λ_max(A) · λ_max(B) via power iteration

Factored damping (Martens–Grosse):
    π_ℓ = √(tr(A_ℓ)/d_in ÷ tr(B_ℓ)/d_out)
    Ã_ℓ = A_ℓ + π_ℓ · √λ · I
    B̃_ℓ = B_ℓ + (1/π_ℓ) · √λ · I
    so that Ã_ℓ ⊗ B̃_ℓ ≈ F_ℓ + λI
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from kbezier.metrics.fisher_base import FisherMetric
from kbezier.metrics.layer_hooks import KFACHookManager


class KFACFisher(FisherMetric):
    """
    Kronecker-Factored Approximate Curvature (K-FAC).

    Stores (A_ℓ, B_ℓ) per layer and provides efficient operations
    through Kronecker algebra.
    """

    def __init__(self, damping: float = 1e-3):
        """
        Args:
            damping: Tikhonov damping λ. Applied via Martens–Grosse factored scheme.
        """
        self.damping = damping
        # {layer_name: (A_ℓ, B_ℓ)} — raw factors
        self.factors: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        # {layer_name: (Ã⁻¹, B̃⁻¹)} — damped inverses (cached)
        self._inv_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        # Mapping from model param names → layer names (for grad_dict lookup)
        self._param_to_layer: Dict[str, str] = {}
        self._layer_to_weight_name: Dict[str, str] = {}

    def accumulate(
        self,
        model: nn.Module,
        data_loader,
        criterion=None,
        n_samples: Optional[int] = None,
    ) -> None:
        """
        Estimate K-FAC factors using hooks.

        Args:
            model: Network to compute Fisher for.
            data_loader: DataLoader for Fisher estimation.
            criterion: Loss function. Default: CrossEntropyLoss.
            n_samples: Max samples to use. None = full dataset.
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        # Build param name mapping
        self._build_param_mapping(model)

        # Set up hooks
        hook_manager = KFACHookManager(model, absorb_bias=True)
        hook_manager.register_hooks()
        hook_manager.reset_accumulators()

        model.eval()
        device = next(model.parameters()).device
        count = 0

        try:
            for x, y in data_loader:
                x, y = x.to(device), y.to(device)

                model.zero_grad()
                output = model(x)
                loss = criterion(output, y)
                loss.backward()

                hook_manager.accumulate_step()
                count += x.size(0)

                if n_samples is not None and count >= n_samples:
                    break

            self.factors = hook_manager.finalize()
            self._recompute_inverses()

        finally:
            # CRITICAL: always remove hooks to avoid leaks
            hook_manager.remove_hooks()

    def _build_param_mapping(self, model: nn.Module) -> None:
        """Map parameter names to layer names for grad_dict lookup."""
        self._param_to_layer.clear()
        self._layer_to_weight_name.clear()

        for layer_name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                weight_name = f"{layer_name}.weight"
                self._param_to_layer[weight_name] = layer_name
                self._layer_to_weight_name[layer_name] = weight_name
                if module.bias is not None:
                    bias_name = f"{layer_name}.bias"
                    self._param_to_layer[bias_name] = layer_name

    def _apply_factored_damping(
        self, A: torch.Tensor, B: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Martens–Grosse factored damping.

        π_ℓ = √(tr(A)/d_in ÷ tr(B)/d_out)
        Ã = A + π√λ·I
        B̃ = B + (1/π)√λ·I

        This ensures Ã⊗B̃ ≈ F + λI while preserving Kronecker structure.
        """
        d_in = A.size(0)
        d_out = B.size(0)

        tr_A = torch.trace(A).clamp(min=1e-10)
        tr_B = torch.trace(B).clamp(min=1e-10)

        pi = torch.sqrt((tr_A / d_in) / (tr_B / d_out)).clamp(min=1e-6, max=1e6)
        sqrt_lambda = math.sqrt(self.damping)

        A_damped = A + pi * sqrt_lambda * torch.eye(d_in, device=A.device, dtype=A.dtype)
        B_damped = B + (1.0 / pi) * sqrt_lambda * torch.eye(d_out, device=B.device, dtype=B.dtype)

        return A_damped, B_damped

    def _recompute_inverses(self) -> None:
        """Compute and cache damped inverses for all layers."""
        self._inv_cache.clear()
        for name, (A, B) in self.factors.items():
            A_damped, B_damped = self._apply_factored_damping(A, B)
            # Use Cholesky for numerically stable inversion of small PD matrices
            try:
                A_inv = torch.linalg.inv(A_damped)
                B_inv = torch.linalg.inv(B_damped)
            except torch.linalg.LinAlgError:
                # Fallback: add extra damping
                A_inv = torch.linalg.inv(A_damped + 1e-4 * torch.eye(A_damped.size(0), device=A_damped.device))
                B_inv = torch.linalg.inv(B_damped + 1e-4 * torch.eye(B_damped.size(0), device=B_damped.device))
            self._inv_cache[name] = (A_inv, B_inv)

    def inv_mv(self, grad_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute F⁻¹ @ grad via K-FAC vec-trick.

        For each layer ℓ:
            F_ℓ⁻¹ vec(∇W) = vec(B̃⁻¹ ∇W Ã⁻¹)

        This is TWO small matrix multiplications per layer, never
        materializing the full (d_in·d_out × d_in·d_out) matrix.

        Args:
            grad_dict: {param_name: gradient_tensor}

        Returns:
            {param_name: F⁻¹ @ gradient_tensor}
        """
        result = {}

        for param_name, grad in grad_dict.items():
            layer_name = self._param_to_layer.get(param_name)
            if layer_name is None or layer_name not in self._inv_cache:
                # Parameter not covered by K-FAC: return grad unchanged (identity)
                result[param_name] = grad
                continue

            A_inv, B_inv = self._inv_cache[layer_name]

            if "weight" in param_name:
                # grad is (d_out, d_in) for Linear, (C_out, C_in, k, k) for Conv2d
                grad_mat = grad.reshape(grad.size(0), -1)  # (d_out, d_in_flat)

                # If bias was absorbed, A_inv is (d_in+1, d_in+1)
                # but grad_mat is (d_out, d_in). Trim A_inv.
                d_out, d_in_flat = grad_mat.shape
                a_dim = A_inv.size(0)
                if a_dim > d_in_flat:
                    # Bias was absorbed: use top-left (d_in, d_in) block
                    A_inv_w = A_inv[:d_in_flat, :d_in_flat]
                else:
                    A_inv_w = A_inv

                # Vec-trick: F⁻¹ vec(∇W) = vec(B̃⁻¹ ∇W Ã⁻¹)
                nat_grad_mat = B_inv @ grad_mat @ A_inv_w
                result[param_name] = nat_grad_mat.reshape_as(grad)

            elif "bias" in param_name:
                # If bias absorbed: natural gradient for bias comes from
                # the last row/column of the Kronecker product.
                # Simplified: just scale by B_inv diagonal average
                result[param_name] = B_inv @ grad.unsqueeze(-1)
                result[param_name] = result[param_name].squeeze(-1)

        return result

    def quad(self, vec_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute vᵀ F v = Σ_ℓ tr(Δ_ℓᵀ B_ℓ Δ_ℓ A_ℓ) per layer.

        Uses Kronecker identity: vec(Δ)ᵀ (A⊗B) vec(Δ) = tr(Δᵀ B Δ A).
        No inverse needed — uses raw factors.
        """
        total = torch.tensor(0.0)

        for param_name, vec in vec_dict.items():
            layer_name = self._param_to_layer.get(param_name)
            if layer_name is None or layer_name not in self.factors:
                continue
            if "bias" in param_name:
                continue  # Handle bias separately if needed

            A, B = self.factors[layer_name]

            # Reshape to matrix (d_out, d_in)
            delta = vec.reshape(vec.size(0), -1)
            d_out, d_in_flat = delta.shape
            a_dim = A.size(0)
            if a_dim > d_in_flat:
                A_w = A[:d_in_flat, :d_in_flat]
            else:
                A_w = A

            # tr(Δᵀ B Δ A) = tr(A Δᵀ B Δ)
            # Efficient: (B @ Δ) then element-wise multiply with Δ, sum, then trace with A
            BΔ = B @ delta                    # (d_out, d_in)
            ΔᵀBΔ = delta.t() @ BΔ           # (d_in, d_in)
            val = (ΔᵀBΔ * A_w).sum()         # tr(Δᵀ B Δ A)

            if total.device != val.device:
                total = total.to(val.device)
            total = total + val

        return total

    def top_eigs(self, k: int = 1) -> torch.Tensor:
        """
        Compute top-k eigenvalues of F.

        Uses: λ_max(A⊗B) = λ_max(A) · λ_max(B).
        Each factor's eigenvalues via torch.linalg.eigvalsh (exact for small matrices).
        Returns the global top-k across all layers.
        """
        all_eigs = []

        for name, (A, B) in self.factors.items():
            # Eigenvalues of small factors (cheap — factors are d×d)
            eigs_A = torch.linalg.eigvalsh(A)  # ascending
            eigs_B = torch.linalg.eigvalsh(B)

            # Top eigenvalue of Kronecker product = product of tops
            top_A = eigs_A[-1].clamp(min=0)
            top_B = eigs_B[-1].clamp(min=0)
            all_eigs.append(top_A * top_B)

        if not all_eigs:
            return torch.zeros(k)

        all_eigs = torch.stack(all_eigs)
        k = min(k, len(all_eigs))
        return torch.topk(all_eigs, k).values

    def state_dict(self) -> Dict[str, Any]:
        """Serialize K-FAC state."""
        return {
            "factors": {
                name: (A.cpu(), B.cpu())
                for name, (A, B) in self.factors.items()
            },
            "damping": self.damping,
            "param_to_layer": dict(self._param_to_layer),
            "layer_to_weight_name": dict(self._layer_to_weight_name),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load K-FAC state."""
        self.damping = state["damping"]
        self._param_to_layer = state["param_to_layer"]
        self._layer_to_weight_name = state["layer_to_weight_name"]
        self.factors = {}
        for name, (A, B) in state["factors"].items():
            self.factors[name] = (A, B)
        self._recompute_inverses()

    def to(self, device: str) -> "KFACFisher":
        """Move all factors to a device."""
        self.factors = {
            name: (A.to(device), B.to(device))
            for name, (A, B) in self.factors.items()
        }
        self._recompute_inverses()
        return self
