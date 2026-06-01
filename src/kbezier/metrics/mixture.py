"""
Mixture Fisher: F = γ F_old + (1-γ) F_t + λI

This is the stability–plasticity knob of K-Bézier.
γ → 1: prioritize protecting old tasks (stability)
γ → 0: prioritize learning new task (plasticity)

Online accumulation per Kronecker factor:
    A_old ← β · A_old_prev + A_{t-1}   (same for B)

Delegates inv_mv, quad, top_eigs to damped mixed factors.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from kbezier.metrics.fisher_base import FisherMetric
from kbezier.metrics.kfac import KFACFisher


class MixtureFisher(FisherMetric):
    """
    Mixture of old and current Fisher:
        F = γ · F_old + (1-γ) · F_current + λI

    γ is the ablation knob for stability–plasticity.
    Stores Kronecker factors and mixes them for inv_mv/quad/top_eigs.
    """

    def __init__(
        self,
        gamma: float = 0.75,
        damping: float = 1e-3,
        decay: float = 1.0,
    ):
        """
        Args:
            gamma: Mixture coefficient. F = γ F_old + (1-γ) F_current + λI.
            damping: Tikhonov damping λ.
            decay: Decay factor β for online Fisher accumulation:
                   F_old ← β · F_old_prev + F_{t-1}
        """
        self.gamma = gamma
        self.damping = damping
        self.decay = decay

        # Old Fisher factors (accumulated across tasks)
        # {layer_name: (A_old, B_old)}
        self._old_factors: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

        # Current task Fisher (re-estimated each task)
        self._current_fisher = KFACFisher(damping=damping)

        # Mixed factors (recomputed after accumulate)
        # {layer_name: (A_mix, B_mix)}
        self._mixed_factors: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

        # Cached damped inverses of mixed factors
        self._inv_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

        # Copy param mapping from current fisher
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
        Estimate current-task Fisher and mix with accumulated old Fisher.
        """
        # Estimate current task Fisher
        self._current_fisher.accumulate(
            model, data_loader, criterion, n_samples, empirical=empirical
        )
        self._param_to_layer = dict(self._current_fisher._param_to_layer)
        self._layer_to_weight_name = dict(self._current_fisher._layer_to_weight_name)
        self._layer_to_bias_name = dict(
            getattr(self._current_fisher, "_layer_to_bias_name", {})
        )

        # Mix factors
        self._mix_factors()

    def _mix_factors(self) -> None:
        """
        Compute mixed factors: A_mix = γ A_old + (1-γ) A_current + √λ·I
        (with Martens–Grosse style damping applied separately).
        """
        import math

        self._mixed_factors.clear()
        self._inv_cache.clear()

        for layer_name, (A_cur, B_cur) in self._current_fisher.factors.items():
            d_in = A_cur.size(0)
            d_out = B_cur.size(0)
            device = A_cur.device

            if layer_name in self._old_factors:
                A_old, B_old = self._old_factors[layer_name]
                A_old = A_old.to(device)
                B_old = B_old.to(device)

                # Ensure shape compatibility
                if A_old.shape != A_cur.shape or B_old.shape != B_cur.shape:
                    # Shape mismatch (head expansion) — use current only
                    A_mix = A_cur
                    B_mix = B_cur
                else:
                    A_mix = self.gamma * A_old + (1 - self.gamma) * A_cur
                    B_mix = self.gamma * B_old + (1 - self.gamma) * B_cur
            else:
                # No old Fisher yet (first task)
                A_mix = A_cur
                B_mix = B_cur

            self._mixed_factors[layer_name] = (A_mix, B_mix)

            # Factored damping on mixed factors
            A_damped, B_damped = self._apply_factored_damping(A_mix, B_mix)

            try:
                A_inv = torch.linalg.inv(A_damped)
                B_inv = torch.linalg.inv(B_damped)
            except torch.linalg.LinAlgError:
                A_inv = torch.linalg.inv(
                    A_damped + 1e-4 * torch.eye(d_in, device=device)
                )
                B_inv = torch.linalg.inv(
                    B_damped + 1e-4 * torch.eye(d_out, device=device)
                )

            self._inv_cache[layer_name] = (A_inv, B_inv)

    def _apply_factored_damping(
        self, A: torch.Tensor, B: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Martens–Grosse factored damping (same as in KFACFisher)."""
        import math

        d_in = A.size(0)
        d_out = B.size(0)

        tr_A = torch.trace(A).clamp(min=1e-10)
        tr_B = torch.trace(B).clamp(min=1e-10)

        pi = torch.sqrt((tr_A / d_in) / (tr_B / d_out)).clamp(min=1e-6, max=1e6)
        sqrt_lambda = math.sqrt(self.damping)

        A_damped = A + pi * sqrt_lambda * torch.eye(d_in, device=A.device, dtype=A.dtype)
        B_damped = B + (1.0 / pi) * sqrt_lambda * torch.eye(d_out, device=B.device, dtype=B.dtype)

        return A_damped, B_damped

    def update_old_fisher(self, task_id: int) -> None:
        """
        Update accumulated old Fisher after completing a task.

        F_old ← β · F_old_prev + F_current
        Applied per Kronecker factor.
        """
        for layer_name, (A_cur, B_cur) in self._current_fisher.factors.items():
            if layer_name in self._old_factors:
                A_old, B_old = self._old_factors[layer_name]
                A_old = A_old.to(A_cur.device)
                B_old = B_old.to(B_cur.device)

                if A_old.shape == A_cur.shape and B_old.shape == B_cur.shape:
                    A_new = self.decay * A_old + A_cur
                    B_new = self.decay * B_old + B_cur
                else:
                    A_new = A_cur
                    B_new = B_cur
            else:
                A_new = A_cur.clone()
                B_new = B_cur.clone()

            self._old_factors[layer_name] = (A_new.detach().cpu(), B_new.detach().cpu())

    def inv_mv(self, grad_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """F_mix⁻¹ @ grad via vec-trick on (pre-mixed, factored) mixed factors.

        NOTE: inverting the TRUE mixture γ(A_old⊗B_old)+(1-γ)(A_cur⊗B_cur)+λI is
        intractable (sum of Kroneckers), so we use the standard K-FAC
        approximation of pre-mixing factors A_mix, B_mix and inverting that single
        Kronecker block. Bias is handled jointly as the augmented matrix [∇W|∇b]
        (see KFACFisher.inv_mv)."""
        result = {}
        processed_biases: set = set()

        for param_name, grad in grad_dict.items():
            layer_name = self._param_to_layer.get(param_name)
            if layer_name is None or layer_name not in self._inv_cache:
                result[param_name] = grad
                continue

            if "bias" in param_name and param_name in processed_biases:
                continue
            if "bias" in param_name and "weight" not in param_name:
                weight_name = self._layer_to_weight_name.get(layer_name)
                if weight_name is not None and weight_name in grad_dict:
                    continue
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
                grad_mat = grad.reshape(grad.size(0), -1)
                d_out, d_in_flat = grad_mat.shape
                a_dim = A_inv.size(0)

                bias_name = self._layer_to_bias_name.get(layer_name)
                has_bias_absorbed = (a_dim == d_in_flat + 1)

                if bias_name is not None and bias_name in grad_dict and has_bias_absorbed:
                    b_grad = grad_dict[bias_name].reshape(d_out, 1)
                    W_aug = torch.cat([grad_mat, b_grad], dim=1)
                    nat_aug = B_inv @ W_aug @ A_inv
                    result[param_name] = nat_aug[:, :d_in_flat].reshape_as(grad)
                    result[bias_name] = nat_aug[:, d_in_flat:].reshape(-1)
                    processed_biases.add(bias_name)
                else:
                    A_inv_w = A_inv[:d_in_flat, :d_in_flat] if a_dim > d_in_flat else A_inv
                    nat_grad_mat = B_inv @ grad_mat @ A_inv_w
                    result[param_name] = nat_grad_mat.reshape_as(grad)

        return result

    def quad(self, vec_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """vᵀ F_mix v using (pre-mixed) factors, with augmented bias handling."""
        total = torch.tensor(0.0)
        processed_biases: set = set()

        for param_name, vec in vec_dict.items():
            layer_name = self._param_to_layer.get(param_name)
            if layer_name is None or layer_name not in self._mixed_factors:
                continue

            if "bias" in param_name and param_name in processed_biases:
                continue
            if "bias" in param_name and "weight" not in param_name:
                weight_name = self._layer_to_weight_name.get(layer_name)
                if weight_name is not None and weight_name in vec_dict:
                    continue
                A, B = self._mixed_factors[layer_name]
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

            A, B = self._mixed_factors[layer_name]
            delta = vec.reshape(vec.size(0), -1)
            d_out, d_in_flat = delta.shape
            a_dim = A.size(0)

            bias_name = self._layer_to_bias_name.get(layer_name)
            has_bias_absorbed = (a_dim == d_in_flat + 1)

            if bias_name is not None and bias_name in vec_dict and has_bias_absorbed:
                b_vec = vec_dict[bias_name].reshape(d_out, 1)
                delta = torch.cat([delta, b_vec], dim=1)
                A_w = A
                processed_biases.add(bias_name)
            else:
                A_w = A[:d_in_flat, :d_in_flat] if a_dim > d_in_flat else A

            BΔ = B @ delta
            ΔᵀBΔ = delta.t() @ BΔ
            val = (ΔᵀBΔ * A_w).sum()

            if total.device != val.device:
                total = total.to(val.device)
            total = total + val

        return total

    @staticmethod
    def _kron_matvec(A: torch.Tensor, B: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """(A⊗B) vec_row(X) = vec_row(B X Aᵀ), X shape (d_out, d_in)."""
        return B @ X @ A.t()

    def _mixture_block_matvec(self, layer_name: str):
        """Matvec of the TRUE damped mixture block of one layer:

            M = γ (A_old ⊗ B_old) + (1-γ) (A_cur ⊗ B_cur) + λI

        a SUM of Kronecker products plus identity — NOT a single Kronecker
        product, so λ_max(M) ≠ λ_max(A_mix)·λ_max(B_mix). We expose only the
        matvec and let power iteration find λ_max(M)."""
        A_cur, B_cur = self._current_fisher.factors[layer_name]
        device = A_cur.device
        has_old = layer_name in self._old_factors
        if has_old:
            A_old, B_old = self._old_factors[layer_name]
            A_old = A_old.to(device)
            B_old = B_old.to(device)
            if A_old.shape != A_cur.shape or B_old.shape != B_cur.shape:
                has_old = False

        lam = self.damping
        g = self.gamma

        def matvec(X: torch.Tensor) -> torch.Tensor:
            out = lam * X
            out = out + (1.0 - g) * self._kron_matvec(A_cur, B_cur, X)
            if has_old:
                out = out + g * self._kron_matvec(A_old, B_old, X)
            else:
                out = out + g * self._kron_matvec(A_cur, B_cur, X)
            return out

        return matvec, B_cur.size(0), A_cur.size(0), device

    @staticmethod
    def _power_iter_top_eig(d_out, d_in, device, matvec, n_iter=50, tol=1e-6):
        v = torch.randn(d_out, d_in, device=device)
        v = v / v.norm().clamp(min=1e-12)
        eig = torch.tensor(0.0, device=device)
        for _ in range(n_iter):
            w = matvec(v)
            new_eig = (v * w).sum()
            v = w / w.norm().clamp(min=1e-12)
            if (new_eig - eig).abs() <= tol * new_eig.abs().clamp(min=1e-12):
                eig = new_eig
                break
            eig = new_eig
        return eig.clamp(min=0)

    def top_eigs(self, k: int = 1) -> torch.Tensor:
        """Top-k eigenvalues of the TRUE mixture Fisher, per layer, via power
        iteration on each layer's mixture block matvec. The product shortcut
        λ_max(A)·λ_max(B) is INVALID here (sum of Kronecker products)."""
        all_eigs = []
        for layer_name in self._current_fisher.factors:
            matvec, d_out, d_in, device = self._mixture_block_matvec(layer_name)
            all_eigs.append(self._power_iter_top_eig(d_out, d_in, device, matvec))

        if not all_eigs:
            return torch.zeros(k)
        all_eigs = torch.stack(all_eigs)
        k = min(k, len(all_eigs))
        return torch.topk(all_eigs, k).values

    def get_old_fisher_top_eig(self) -> torch.Tensor:
        """λ_max(F_old) for the Proposition 2 bound ρ²/λ ≤ 2τ/λ_max(F_old).

        F_old per layer IS a single Kronecker product A_old⊗B_old (accumulated,
        undamped), so the product-of-tops identity is exact here."""
        all_eigs = []
        for name, (A, B) in self._old_factors.items():
            eigs_A = torch.linalg.eigvalsh(A)
            eigs_B = torch.linalg.eigvalsh(B)
            all_eigs.append(eigs_A[-1].clamp(min=0) * eigs_B[-1].clamp(min=0))

        if not all_eigs:
            return torch.tensor(0.0)
        return torch.stack(all_eigs).max()

    def state_dict(self) -> Dict[str, Any]:
        return {
            "gamma": self.gamma,
            "damping": self.damping,
            "decay": self.decay,
            "old_factors": {
                k: (A.cpu(), B.cpu())
                for k, (A, B) in self._old_factors.items()
            },
            "current_fisher": self._current_fisher.state_dict(),
            "param_to_layer": dict(self._param_to_layer),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.gamma = state["gamma"]
        self.damping = state["damping"]
        self.decay = state["decay"]
        self._param_to_layer = state["param_to_layer"]
        self._old_factors = {
            k: (A, B) for k, (A, B) in state["old_factors"].items()
        }
        self._current_fisher.load_state_dict(state["current_fisher"])
        self._mix_factors()
