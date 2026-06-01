"""
MLP model for Permuted MNIST (smoke test).

Single-head with all 10 classes — suitable for domain-IL.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from kbezier.engine.registry import Registry


@Registry.register_model("mlp")
class MLP(nn.Module):
    """Simple feedforward MLP for smoke testing on Permuted MNIST."""

    def __init__(
        self,
        input_size: int = 784,
        hidden_sizes: List[int] | None = None,
        n_classes: int = 10,
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [400, 400]

        layers = []
        prev_size = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_size, h))
            layers.append(nn.ReLU(inplace=False))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_size = h

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten if needed (batch, 1, 28, 28) -> (batch, 784)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        features = self.features(x)
        return self.classifier(features)
