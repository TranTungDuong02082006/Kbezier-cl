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
        self._layer_to_bias_name: Dict[str, str] = {}

    def accumulate(
        self,
        model: nn.Module,
        data_loader,
        criterion=None,
        n_samples: Optional[int] = None,
        empirical: bool = False,
    ) -> None:
        """
        Estimate K-FAC factors using hooks.

        Two correctness points (vs. a naive implementation):

        1. TRUE Fisher vs EMPIRICAL Fisher. The Fisher Information Matrix is an
           expectation over labels drawn from the MODEL's predictive
           distribution p_w(y|x), NOT the dataset's true labels. We therefore
           sample y ~ softmax(logits) by default. Set empirical=True to use
           dataset labels (empirical Fisher), a cruder approximation.

        2. PER-SAMPLE output gradients. K-FAC's B factor is E[g gᵀ] where g is
           the gradient of the PER-SAMPLE loss w.r.t. the layer pre-activation.
           With mean reduction the captured grad_output is divided by the batch
           size, which collapses the scale of B (trace(B) ~ 1/N² of trace(A))
           and breaks the Martens–Grosse damping balance, making F⁻¹g point in
           nearly a random direction. We use reduction='sum' so each captured g
           is a true per-sample gradient; finalize() then divides by the sample
           count to form the correct expectation.

        Args:
            model: Network to compute Fisher for.
            data_loader: DataLoader for Fisher estimation.
            criterion: IGNORED (we always use summed cross-entropy internally to
                       obtain per-sample gradients); kept for API compatibility.
            n_samples: Max samples to use. None = full dataset.
            empirical: If True, use dataset labels (empirical Fisher) instead of
                       model-sampled labels (true Fisher).
        """
        # Always use summed cross-entropy so grad_output is the per-sample
        # gradient (not divided by batch size). See docstring point 2.
        sum_ce = nn.CrossEntropyLoss(reduction="sum")

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

                if empirical:
                    target = y
                else:
                    # True Fisher: sample labels from the model distribution.
                    with torch.no_grad():
                        probs = torch.softmax(output, dim=1)
                        target = torch.multinomial(probs, num_samples=1).squeeze(1)

                loss = sum_ce(output, target)
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
        self._layer_to_bias_name.clear()

        for layer_name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                weight_name = f"{layer_name}.weight"
                self._param_to_layer[weight_name] = layer_name
                self._layer_to_weight_name[layer_name] = weight_name
                if module.bias is not None:
                    bias_name = f"{layer_name}.bias"
                    self._param_to_layer[bias_name] = layer_name
                    self._layer_to_bias_name[layer_name] = bias_name

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
            F_ℓ⁻¹ vec(∇W_aug) = vec(B̃⁻¹ ∇W_aug Ã⁻¹)

        BIAS HANDLING (correct, augmented form): the hook absorbs bias by
        appending a 1 to the activation, so Ã is (d_in+1) × (d_in+1) and the
        layer's true parameter is the AUGMENTED matrix W_aug = [W | b] of shape
        (d_out, d_in+1) — bias is the last column. The Fisher block is B ⊗ Ã, so
        the natural gradient is B̃⁻¹ [∇W | ∇b] Ã⁻¹, after which we split the last
        column back out as the bias. Processing weight and bias separately (the
        previous approach) drops the weight–bias cross terms in Ã⁻¹ and is wrong.

        This is TWO small matrix multiplications per layer, never materializing
        the full matrix.
        """
        result = {}
        processed_biases: set = set()

        for param_name, grad in grad_dict.items():
            layer_name = self._param_to_layer.get(param_name)
            if layer_name is None or layer_name not in self._inv_cache:
                # Parameter not covered by K-FAC: return grad unchanged (identity)
                result[param_name] = grad
                continue

            if "bias" in param_name and param_name in processed_biases:
                continue
            if "bias" in param_name and "weight" not in param_name:
                weight_name = self._layer_to_weight_name.get(layer_name)
                if weight_name is not None and weight_name in grad_dict:
                    continue  # handled jointly with its weight below
                # bias-only fallback: augmented [0 | b], full A_inv
                A_inv, B_inv = self._inv_cache[layer_name]
                a_dim = A_inv.size(0)
                W_aug = torch.zeros(grad.size(0), a_dim, device=grad.device,
                                    dtype=grad.dtype)
                W_aug[:, -1] = grad
                nat = B_inv @ W_aug @ A_inv
                result[param_name] = nat[:, -1].contiguous()
                continue

            if "weight" in param_name:
                A_inv, B_inv = self._inv_cache[layer_name]
                grad_mat = grad.reshape(grad.size(0), -1)  # (d_out, d_in_flat)
                d_out, d_in_flat = grad_mat.shape
                a_dim = A_inv.size(0)

                bias_name = self._layer_to_bias_name.get(layer_name)
                has_bias_absorbed = (a_dim == d_in_flat + 1)

                if bias_name is not None and bias_name in grad_dict and has_bias_absorbed:
                    # Joint augmented handling: W_aug = [∇W | ∇b]
                    b_grad = grad_dict[bias_name].reshape(d_out, 1)
                    W_aug = torch.cat([grad_mat, b_grad], dim=1)  # (d_out, d_in+1)
                    nat_aug = B_inv @ W_aug @ A_inv                # (d_out, d_in+1)
                    result[param_name] = nat_aug[:, :d_in_flat].reshape_as(grad)
                    result[bias_name] = nat_aug[:, d_in_flat:].reshape(-1)
                    processed_biases.add(bias_name)
                else:
                    # No bias (or no absorption): trim Ã⁻¹ to the weight block.
                    A_inv_w = A_inv[:d_in_flat, :d_in_flat] if a_dim > d_in_flat else A_inv
                    nat_grad_mat = B_inv @ grad_mat @ A_inv_w
                    result[param_name] = nat_grad_mat.reshape_as(grad)

        return result

    def quad(self, vec_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute vᵀ F v = Σ_ℓ tr(Δ_aug,ℓᵀ B_ℓ Δ_aug,ℓ A_ℓ) per layer.

        Uses Kronecker identity: vec(Δ)ᵀ (A⊗B) vec(Δ) = tr(Δᵀ B Δ A).
        No inverse needed — uses raw factors.

        BIAS HANDLING: when bias is absorbed (A is (d_in+1)×(d_in+1)), the
        displacement/velocity for the layer is the augmented matrix
        Δ_aug = [ΔW | Δb] and the quadratic uses the FULL A. Dropping bias (the
        previous approach) both ignored the bias energy and mismatched A's
        (d_in+1) dimension.
        """
        total = torch.tensor(0.0)
        processed_biases: set = set()

        for param_name, vec in vec_dict.items():
            layer_name = self._param_to_layer.get(param_name)
            if layer_name is None or layer_name not in self.factors:
                continue

            if "bias" in param_name and param_name in processed_biases:
                continue
            if "bias" in param_name and "weight" not in param_name:
                weight_name = self._layer_to_weight_name.get(layer_name)
                if weight_name is not None and weight_name in vec_dict:
                    continue  # handled jointly with the weight
                A, B = self.factors[layer_name]
                a_dim = A.size(0)
                delta = torch.zeros(vec.size(0), a_dim, device=vec.device, dtype=vec.dtype)
                delta[:, -1] = vec
                BΔ = B @ delta
                ΔᵀBΔ = delta.t() @ BΔ
                val = (ΔᵀBΔ * A).sum()
                if total.device != val.device:
                    total = total.to(val.device)
                total = total + val
                continue

            if "weight" not in param_name:
                continue

            A, B = self.factors[layer_name]
            delta = vec.reshape(vec.size(0), -1)  # (d_out, d_in_flat)
            d_out, d_in_flat = delta.shape
            a_dim = A.size(0)

            bias_name = self._layer_to_bias_name.get(layer_name)
            has_bias_absorbed = (a_dim == d_in_flat + 1)

            if bias_name is not None and bias_name in vec_dict and has_bias_absorbed:
                b_vec = vec_dict[bias_name].reshape(d_out, 1)
                delta = torch.cat([delta, b_vec], dim=1)  # (d_out, d_in+1)
                A_w = A
                processed_biases.add(bias_name)
            else:
                A_w = A[:d_in_flat, :d_in_flat] if a_dim > d_in_flat else A

            # tr(Δᵀ B Δ A)
            BΔ = B @ delta
            ΔᵀBΔ = delta.t() @ BΔ
            val = (ΔᵀBΔ * A_w).sum()

            if total.device != val.device:
                total = total.to(val.device)
            total = total + val

        return total

    def top_eigs(self, k: int = 1) -> torch.Tensor:
        """
        Compute top-k eigenvalues of the damped Fisher F + λI (per-layer max).

        For a SINGLE Kronecker block the identity λ_max(Ã⊗B̃)=λ_max(Ã)·λ_max(B̃)
        is exact, so we use it on the DAMPED factors (the same Ã, B̃ that inv_mv
        inverts). Returns the global top-k across layers.

        NOTE: exactness holds only because each layer's Fisher is a single
        Kronecker product. MixtureFisher overrides this with power iteration
        because a sum of Kronecker products is NOT a Kronecker product.
        """
        all_eigs = []

        for name, (A, B) in self.factors.items():
            A_d, B_d = self._apply_factored_damping(A, B)
            eigs_A = torch.linalg.eigvalsh(A_d)  # ascending
            eigs_B = torch.linalg.eigvalsh(B_d)
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
