"""
K-FAC hook manager: forward/backward hooks to capture activations and gradients.

This is the #1 engineering risk — handles:
- Forward hook: captures input activations a_{ℓ-1}
- Backward hook: captures output gradients g_ℓ = ∂L/∂s_ℓ
- Layer type → A,B recipe mapping (Linear vs Conv2d with unfold)
- Bias absorption (append 1 to activation for homogeneous coordinates)
- Hook lifecycle management (register/remove to avoid memory leaks)

Layer type → factor computation:
┌──────────────┬──────────────────────────────┬───────────────────────────────┐
│ Layer Type   │ A_ℓ = E[a a^T]               │ B_ℓ = E[g g^T]                │
├──────────────┼──────────────────────────────┼───────────────────────────────┤
│ nn.Linear    │ a = input[0], (B, d_in)      │ g = grad_output[0], (B, d_out)│
│              │ A: (d_in+1)×(d_in+1) w/bias  │ B: d_out × d_out              │
├──────────────┼──────────────────────────────┼───────────────────────────────┤
│ nn.Conv2d    │ a = unfold(input[0])          │ g = reshape grad_output[0]    │
│              │   → avg over spatial          │   → avg over spatial          │
│              │ A: (C_in·k²+1)×(C_in·k²+1)  │ B: C_out × C_out              │
├──────────────┼──────────────────────────────┼───────────────────────────────┤
│ BatchNorm    │ SKIPPED (not meaningful)      │ —                             │
└──────────────┴──────────────────────────────┴───────────────────────────────┘
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# Supported layer types for K-FAC
KFAC_SUPPORTED_LAYERS = (nn.Linear, nn.Conv2d)


class KFACHookManager:
    """
    Manages forward/backward hooks for K-FAC factor estimation.

    Usage:
        manager = KFACHookManager(model)
        manager.register_hooks()

        # Run forward + backward passes to accumulate statistics
        for x, y in loader:
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            manager.accumulate_step()

        # Get factors
        factors = manager.finalize()

        # Clean up (CRITICAL — avoid memory leaks)
        manager.remove_hooks()
    """

    def __init__(self, model: nn.Module, absorb_bias: bool = True):
        """
        Args:
            model: Neural network to hook.
            absorb_bias: If True, append 1 to activations to absorb bias
                         into the weight matrix (homogeneous coordinates).
        """
        self.model = model
        self.absorb_bias = absorb_bias

        # Storage for captured data
        self._activations: Dict[str, torch.Tensor] = {}
        self._gradients: Dict[str, torch.Tensor] = {}

        # Running accumulators for factors
        self._A_sum: Dict[str, torch.Tensor] = {}  # Σ (a aᵀ)
        self._B_sum: Dict[str, torch.Tensor] = {}  # Σ (g gᵀ)
        self._n_samples: Dict[str, int] = {}

        # Hook handles for removal
        self._fwd_handles: List[torch.utils.hooks.RemovableHook] = []
        self._bwd_handles: List[torch.utils.hooks.RemovableHook] = []

        # Track which layers we're hooking
        self._layer_names: List[str] = []
        self._layer_types: Dict[str, type] = {}
        self._layer_modules: Dict[str, nn.Module] = {}

    def get_hookable_layers(self) -> List[Tuple[str, nn.Module]]:
        """Return list of (name, module) for layers eligible for K-FAC hooks."""
        layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, KFAC_SUPPORTED_LAYERS):
                layers.append((name, module))
        return layers

    def register_hooks(self) -> None:
        """Attach forward and backward hooks to all eligible layers."""
        for name, module in self.get_hookable_layers():
            self._layer_names.append(name)
            self._layer_types[name] = type(module)
            self._layer_modules[name] = module

            # Forward hook: capture input activation
            fwd_handle = module.register_forward_hook(
                self._make_forward_hook(name)
            )
            self._fwd_handles.append(fwd_handle)

            # Backward hook: capture output gradient
            bwd_handle = module.register_full_backward_hook(
                self._make_backward_hook(name)
            )
            self._bwd_handles.append(bwd_handle)

    def _make_forward_hook(self, layer_name: str):
        """Create a forward hook closure for the given layer."""
        def hook(module: nn.Module, input: tuple, output: torch.Tensor):
            # Extract activation from input
            a = input[0].detach()  # (batch, ...)
            a = self._process_activation(a, layer_name)
            self._activations[layer_name] = a
        return hook

    def _make_backward_hook(self, layer_name: str):
        """Create a backward hook closure for the given layer."""
        def hook(module: nn.Module, grad_input: tuple, grad_output: tuple):
            # Extract output gradient
            g = grad_output[0].detach()  # (batch, ...)
            g = self._process_gradient(g, layer_name)
            self._gradients[layer_name] = g
        return hook

    def _process_activation(self, a: torch.Tensor, layer_name: str) -> torch.Tensor:
        """
        Process raw activation into K-FAC format.

        For Linear: a is (batch, d_in) → optionally append 1 for bias.
        For Conv2d: unfold input → average over spatial dimensions.
        """
        layer_type = self._layer_types[layer_name]
        module = self._layer_modules[layer_name]

        if layer_type == nn.Linear:
            # a: (batch, d_in)
            if self.absorb_bias and module.bias is not None:
                ones = torch.ones(a.size(0), 1, device=a.device, dtype=a.dtype)
                a = torch.cat([a, ones], dim=1)  # (batch, d_in + 1)
            return a

        elif layer_type == nn.Conv2d:
            # a: (batch, C_in, H, W)
            # Unfold to patches: (batch, C_in * k_h * k_w, H_out * W_out)
            a_unfold = F.unfold(
                a,
                kernel_size=module.kernel_size,
                dilation=module.dilation,
                padding=module.padding,
                stride=module.stride,
            )  # (batch, C_in * k² , L) where L = H_out * W_out

            # Average over spatial locations
            a_avg = a_unfold.mean(dim=2)  # (batch, C_in * k²)

            if self.absorb_bias and module.bias is not None:
                ones = torch.ones(a_avg.size(0), 1, device=a_avg.device, dtype=a_avg.dtype)
                a_avg = torch.cat([a_avg, ones], dim=1)

            return a_avg

        else:
            raise TypeError(f"Unsupported layer type: {layer_type}")

    def _process_gradient(self, g: torch.Tensor, layer_name: str) -> torch.Tensor:
        """
        Process raw output gradient into K-FAC format.

        For Linear: g is (batch, d_out).
        For Conv2d: g is (batch, C_out, H_out, W_out) → average over spatial.
        """
        layer_type = self._layer_types[layer_name]

        if layer_type == nn.Linear:
            # g: (batch, d_out)
            return g

        elif layer_type == nn.Conv2d:
            # g: (batch, C_out, H_out, W_out)
            # Average over spatial dimensions
            g_avg = g.mean(dim=[2, 3])  # (batch, C_out)
            return g_avg

        else:
            raise TypeError(f"Unsupported layer type: {layer_type}")

    def accumulate_step(self) -> None:
        """
        Accumulate outer products from captured activations and gradients.

        Call this AFTER each forward + backward pass.
        Updates running sums A_sum += Σ_batch (a aᵀ), B_sum += Σ_batch (g gᵀ).
        """
        for name in self._layer_names:
            if name not in self._activations or name not in self._gradients:
                continue

            a = self._activations[name]  # (batch, d_a)
            g = self._gradients[name]    # (batch, d_g)
            batch_size = a.size(0)

            # Outer products: A = aᵀa / batch, B = gᵀg / batch
            # We sum over batch and track count for final averaging
            A_batch = (a.t() @ a)  # (d_a, d_a)
            B_batch = (g.t() @ g)  # (d_g, d_g)

            if name not in self._A_sum:
                self._A_sum[name] = torch.zeros_like(A_batch)
                self._B_sum[name] = torch.zeros_like(B_batch)
                self._n_samples[name] = 0

            self._A_sum[name] += A_batch
            self._B_sum[name] += B_batch
            self._n_samples[name] += batch_size

        # Clear captured data to free memory
        self._activations.clear()
        self._gradients.clear()

    def finalize(self) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Finalize factor estimation by dividing by sample count.

        Returns:
            Dict mapping layer_name → (A_ℓ, B_ℓ) where:
                A_ℓ = E[a a^T] ≈ (1/N) Σ a aᵀ
                B_ℓ = E[g g^T] ≈ (1/N) Σ g gᵀ
        """
        factors = {}
        for name in self._layer_names:
            if name not in self._A_sum:
                continue
            n = self._n_samples[name]
            if n == 0:
                continue
            A = self._A_sum[name] / n
            B = self._B_sum[name] / n
            factors[name] = (A, B)
        return factors

    def reset_accumulators(self) -> None:
        """Reset running accumulators for a fresh estimation."""
        self._A_sum.clear()
        self._B_sum.clear()
        self._n_samples.clear()
        self._activations.clear()
        self._gradients.clear()

    def remove_hooks(self) -> None:
        """Remove all hooks. CRITICAL to avoid memory leaks."""
        for handle in self._fwd_handles:
            handle.remove()
        for handle in self._bwd_handles:
            handle.remove()
        self._fwd_handles.clear()
        self._bwd_handles.clear()
        self._activations.clear()
        self._gradients.clear()

    @property
    def layer_names(self) -> List[str]:
        return list(self._layer_names)

    def get_layer_dims(self) -> Dict[str, Tuple[int, int]]:
        """
        Return input/output dimensions for each hooked layer.

        Returns:
            Dict mapping layer_name → (d_in, d_out) where d_in may include
            +1 for bias absorption.
        """
        dims = {}
        for name in self._layer_names:
            if name in self._A_sum:
                d_in = self._A_sum[name].size(0)
                d_out = self._B_sum[name].size(0)
                dims[name] = (d_in, d_out)
        return dims
