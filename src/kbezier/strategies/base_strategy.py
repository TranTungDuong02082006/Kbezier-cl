"""
Base CL strategy interface and Finetune baseline.

All CL methods implement this interface. The trainer calls
before_task / observe / after_task without knowing the method.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn

from kbezier.engine.registry import Registry


class BaseStrategy(ABC):
    """
    Abstract CL strategy.

    Lifecycle per task:
        1. trainer calls before_task(task_id, train_loader)
        2. trainer iterates batches, calling observe(x, y, task_id) each
        3. trainer calls after_task(task_id, train_loader)
    """

    def __init__(self, model: nn.Module, config: dict, device: str = "cuda"):
        self.model = model
        self.config = config
        self.device = device
        self.current_task = -1
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.criterion = nn.CrossEntropyLoss()

    def _make_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer from config."""
        lr = self.config.get("lr", 0.01)
        momentum = self.config.get("momentum", 0.9)
        weight_decay = self.config.get("weight_decay", 0.0)
        return torch.optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )

    @abstractmethod
    def before_task(self, task_id: int, train_loader) -> None:
        """Called before training on a new task."""
        ...

    @abstractmethod
    def observe(self, x: torch.Tensor, y: torch.Tensor, task_id: int) -> float:
        """
        Process one batch. Return loss value.

        This method handles the full training step:
        forward, loss computation, backward, optimizer step.
        """
        ...

    @abstractmethod
    def after_task(self, task_id: int, train_loader=None) -> None:
        """Called after completing training on a task."""
        ...


@Registry.register_strategy("finetune")
class FinetuneStrategy(BaseStrategy):
    """
    Finetune baseline: pure SGD, no regularization.

    The worst case for forgetting — establishes the lower bound.
    """

    def before_task(self, task_id: int, train_loader) -> None:
        self.current_task = task_id
        if self.optimizer is None:
            self.optimizer = self._make_optimizer()

    def observe(self, x: torch.Tensor, y: torch.Tensor, task_id: int) -> float:
        self.model.train()
        self.optimizer.zero_grad()

        output = self.model(x)
        loss = self.criterion(output, y)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def after_task(self, task_id: int, train_loader=None) -> None:
        pass  # Nothing to do
