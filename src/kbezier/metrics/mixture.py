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

    def accumulate(
        self,
        model: nn.Module,
        data_loader,
        criterion=None,
        n_samples: Optional[int] = None,
    ) -> None:
        """
        Estimate current-task Fisher and mix with accumulated old Fisher.
        """
        # Estimate current task Fisher
        self._current_fisher.accumulate(model, data_loader, criterion, n_samples)
        self._param_to_layer = dict(self._current_fisher._param_to_layer)

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
        """F_mix⁻¹ @ grad via vec-trick on mixed factors."""
        result = {}
        for param_name, grad in grad_dict.items():
            layer_name = self._param_to_layer.get(param_name)
            if layer_name is None or layer_name not in self._inv_cache:
                result[param_name] = grad
                continue

            A_inv, B_inv = self._inv_cache[layer_name]

            if "weight" in param_name:
                grad_mat = grad.reshape(grad.size(0), -1)
                d_out, d_in_flat = grad_mat.shape
                a_dim = A_inv.size(0)
                A_inv_w = A_inv[:d_in_flat, :d_in_flat] if a_dim > d_in_flat else A_inv

                nat_grad_mat = B_inv @ grad_mat @ A_inv_w
                result[param_name] = nat_grad_mat.reshape_as(grad)
            elif "bias" in param_name:
                result[param_name] = (B_inv @ grad.unsqueeze(-1)).squeeze(-1)

        return result

    def quad(self, vec_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """vᵀ F_mix v using mixed factors."""
        total = torch.tensor(0.0)
        for param_name, vec in vec_dict.items():
            layer_name = self._param_to_layer.get(param_name)
            if layer_name is None or layer_name not in self._mixed_factors:
                continue
            if "bias" in param_name:
                continue

            A, B = self._mixed_factors[layer_name]
            delta = vec.reshape(vec.size(0), -1)
            d_in_flat = delta.size(1)
            a_dim = A.size(0)
            A_w = A[:d_in_flat, :d_in_flat] if a_dim > d_in_flat else A

            BΔ = B @ delta
            ΔᵀBΔ = delta.t() @ BΔ
            val = (ΔᵀBΔ * A_w).sum()

            if total.device != val.device:
                total = total.to(val.device)
            total = total + val

        return total

    def top_eigs(self, k: int = 1) -> torch.Tensor:
        """Top eigenvalues of mixed Fisher."""
        all_eigs = []
        for name, (A, B) in self._mixed_factors.items():
            eigs_A = torch.linalg.eigvalsh(A)
            eigs_B = torch.linalg.eigvalsh(B)
            all_eigs.append(eigs_A[-1].clamp(min=0) * eigs_B[-1].clamp(min=0))

        if not all_eigs:
            return torch.zeros(k)
        all_eigs = torch.stack(all_eigs)
        k = min(k, len(all_eigs))
        return torch.topk(all_eigs, k).values

    def get_old_fisher_top_eig(self) -> torch.Tensor:
        """
        Get λ_max(F_old) — needed for Proposition 2 bound:
        ρ²/λ ≤ 2τ/λ_max(F_old)
        """
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
