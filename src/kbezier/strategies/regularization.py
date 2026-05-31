"""
Regularization-based CL strategies: EWC and SI.

These are baselines that use parameter importance to prevent forgetting.
"""

from __future__ import annotations

import copy
from typing import Dict, Optional

import torch
import torch.nn as nn

from kbezier.engine.registry import Registry
from kbezier.metrics.diagonal import DiagonalFisher
from kbezier.strategies.base_strategy import BaseStrategy


@Registry.register_strategy("ewc")
class EWCStrategy(BaseStrategy):
    """
    Elastic Weight Consolidation (Kirkpatrick et al., 2017).

    Loss = L_task + (λ/2) Σ_i F_i (w - w*_i)²

    Online variant: running average of Fisher across tasks.
    """

    def __init__(self, model: nn.Module, config: dict, device: str = "cuda"):
        super().__init__(model, config, device)

        method_cfg = config.get("method", {})
        self.ewc_lambda = method_cfg.get("ewc_lambda", 400.0)
        self.online = method_cfg.get("online", True)
        self.decay = method_cfg.get("decay", 0.999)
        n_samples = config.get("fisher", {}).get("samples", 1000)
        self.fisher_samples = n_samples

        # Accumulated Fisher and parameter anchors
        self.fisher = DiagonalFisher(damping=0.0)
        self._old_params: Optional[Dict[str, torch.Tensor]] = None
        self._accumulated_fisher: Dict[str, torch.Tensor] = {}

    def before_task(self, task_id: int, train_loader) -> None:
        self.current_task = task_id
        if self.optimizer is None:
            self.optimizer = self._make_optimizer()

    def observe(self, x: torch.Tensor, y: torch.Tensor, task_id: int) -> float:
        self.model.train()
        self.optimizer.zero_grad()

        output = self.model(x)
        loss = self.criterion(output, y)

        # EWC penalty
        if self._old_params is not None:
            ewc_loss = torch.tensor(0.0, device=self.device)
            for name, param in self.model.named_parameters():
                if name in self._accumulated_fisher and name in self._old_params:
                    fisher_val = self._accumulated_fisher[name].to(self.device)
                    old_val = self._old_params[name].to(self.device)
                    ewc_loss = ewc_loss + (fisher_val * (param - old_val) ** 2).sum()
            loss = loss + (self.ewc_lambda / 2) * ewc_loss

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def after_task(self, task_id: int, train_loader=None) -> None:
        # Estimate Fisher for this task
        if train_loader is not None:
            self.fisher.accumulate(
                self.model, train_loader, n_samples=self.fisher_samples
            )

        # Update accumulated Fisher
        if self.online:
            for name, f_val in self.fisher.diag.items():
                if name in self._accumulated_fisher:
                    self._accumulated_fisher[name] = (
                        self.decay * self._accumulated_fisher[name].to(f_val.device) + f_val
                    )
                else:
                    self._accumulated_fisher[name] = f_val.clone()
        else:
            # Non-online: store per-task Fisher (more memory)
            for name, f_val in self.fisher.diag.items():
                if name not in self._accumulated_fisher:
                    self._accumulated_fisher[name] = torch.zeros_like(f_val)
                self._accumulated_fisher[name] = self._accumulated_fisher[name].to(f_val.device) + f_val

        # Save parameter snapshot
        self._old_params = {
            name: param.detach().clone().cpu()
            for name, param in self.model.named_parameters()
        }


@Registry.register_strategy("si")
class SIStrategy(BaseStrategy):
    """
    Synaptic Intelligence (Zenke et al., 2017).

    Tracks online importance of parameters based on their
    contribution to loss decrease along the training path.
    """

    def __init__(self, model: nn.Module, config: dict, device: str = "cuda"):
        super().__init__(model, config, device)

        method_cfg = config.get("method", {})
        self.si_lambda = method_cfg.get("si_lambda", 1.0)
        self.si_epsilon = method_cfg.get("si_epsilon", 0.1)

        # Accumulated importance
        self._omega: Dict[str, torch.Tensor] = {}
        # Running parameter changes * gradient
        self._W: Dict[str, torch.Tensor] = {}
        # Parameter snapshot at task start
        self._old_params: Optional[Dict[str, torch.Tensor]] = None
        # Previous step parameters (for Δw computation)
        self._prev_params: Dict[str, torch.Tensor] = {}

    def before_task(self, task_id: int, train_loader) -> None:
        self.current_task = task_id
        if self.optimizer is None:
            self.optimizer = self._make_optimizer()

        # Reset path integral accumulator
        self._W = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self._W[name] = torch.zeros_like(param)
                self._prev_params[name] = param.detach().clone()

        if task_id == 0:
            self._old_params = {
                name: param.detach().clone()
                for name, param in self.model.named_parameters()
                if param.requires_grad
            }

    def observe(self, x: torch.Tensor, y: torch.Tensor, task_id: int) -> float:
        self.model.train()
        self.optimizer.zero_grad()

        output = self.model(x)
        loss = self.criterion(output, y)

        # SI penalty
        if task_id > 0 and self._old_params is not None:
            si_loss = torch.tensor(0.0, device=self.device)
            for name, param in self.model.named_parameters():
                if name in self._omega and name in self._old_params:
                    omega = self._omega[name].to(self.device)
                    old = self._old_params[name].to(self.device)
                    si_loss = si_loss + (omega * (param - old) ** 2).sum()
            loss = loss + self.si_lambda * si_loss

        loss.backward()

        # Update path integral W (before optimizer step)
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and name in self._prev_params:
                delta = param.detach() - self._prev_params[name]
                self._W[name] += (-param.grad.data * delta)

        self.optimizer.step()

        # Save current params for next delta computation
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self._prev_params[name] = param.detach().clone()

        return loss.item()

    def after_task(self, task_id: int, train_loader=None) -> None:
        # Compute importance from path integral
        for name, param in self.model.named_parameters():
            if name in self._W and name in self._old_params:
                delta = param.detach() - self._old_params[name].to(self.device)
                importance = self._W[name].to(self.device) / (delta ** 2 + self.si_epsilon)
                importance = importance.clamp(min=0)  # importance should be non-negative

                if name not in self._omega:
                    self._omega[name] = torch.zeros_like(importance)
                self._omega[name] = self._omega[name].to(self.device) + importance

        # Update parameter anchor
        self._old_params = {
            name: param.detach().clone().cpu()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
