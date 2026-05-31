"""
ResNet-18 variants for continual learning with IncrementalHead.

IncrementalHead: expanding single-head classifier for Class-IL.
Grows output neurons as new tasks arrive. At test time, predicts
over ALL seen classes without task-id.

Reduced ResNet-18 (nf=20): common in CL literature for CIFAR-scale.
"""

from __future__ import annotations

import copy
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from kbezier.engine.registry import Registry


# ── Incremental Head ────────────────────────────────────────────────

class IncrementalHead(nn.Module):
    """
    Expanding single-head classifier for Class-IL.

    Starts with `initial_classes` output neurons. When `add_classes(n)` is
    called, new output neurons are appended and old weights are preserved.
    At test time, predicts over ALL seen classes (no task-id needed).
    """

    def __init__(self, in_features: int, initial_classes: int = 0):
        super().__init__()
        self.in_features = in_features
        self.n_classes = initial_classes

        if initial_classes > 0:
            self.head = nn.Linear(in_features, initial_classes)
        else:
            self.head = None

    def add_classes(self, n_new: int) -> None:
        """
        Expand the head by n_new output neurons.

        Old weights are copied exactly; new weights are initialized
        using Kaiming uniform (matching nn.Linear default).
        """
        new_total = self.n_classes + n_new
        new_head = nn.Linear(self.in_features, new_total)

        if self.head is not None:
            # Copy old weights
            with torch.no_grad():
                new_head.weight[:self.n_classes] = self.head.weight
                new_head.bias[:self.n_classes] = self.head.bias

        self.head = new_head
        self.n_classes = new_total

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.head is None:
            raise RuntimeError("IncrementalHead has no classes. Call add_classes() first.")
        return self.head(x)

    @property
    def weight(self) -> torch.Tensor:
        return self.head.weight

    @property
    def bias(self) -> torch.Tensor:
        return self.head.bias


# ── Basic Block ─────────────────────────────────────────────────────

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, self.expansion * planes,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# ── ResNet ──────────────────────────────────────────────────────────

class ResNet(nn.Module):
    """
    ResNet with configurable width (nf = number of base filters).

    nf=20: "Reduced ResNet-18" common in CL literature.
    nf=64: Standard ResNet-18.
    """

    def __init__(
        self,
        block: type = BasicBlock,
        num_blocks: List[int] | None = None,
        n_classes: int = 100,
        nf: int = 20,
        use_incremental_head: bool = True,
        initial_classes: int = 0,
    ):
        super().__init__()
        if num_blocks is None:
            num_blocks = [2, 2, 2, 2]

        self.in_planes = nf
        self.nf = nf

        self.conv1 = nn.Conv2d(3, nf, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(nf)
        self.layer1 = self._make_layer(block, nf, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)

        self.feature_dim = nf * 8 * block.expansion

        if use_incremental_head:
            self.classifier = IncrementalHead(self.feature_dim, initial_classes)
        else:
            self.classifier = nn.Linear(self.feature_dim, n_classes)

    def _make_layer(
        self, block: type, planes: int, num_blocks: int, stride: int
    ) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before the classifier head."""
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        return self.classifier(feat)

    def add_classes(self, n_new: int) -> None:
        """Expand the classifier head (only works with IncrementalHead)."""
        if isinstance(self.classifier, IncrementalHead):
            self.classifier.add_classes(n_new)
        else:
            raise TypeError(
                "Cannot add classes: classifier is not IncrementalHead. "
                "Set use_incremental_head=True."
            )


@Registry.register_model("reduced_resnet18")
def make_reduced_resnet18(
    n_classes: int = 100,
    nf: int = 20,
    initial_classes: int = 0,
    **kwargs,
) -> ResNet:
    """Reduced ResNet-18 (nf=20) for CIFAR-scale CL benchmarks."""
    return ResNet(
        block=BasicBlock,
        num_blocks=[2, 2, 2, 2],
        n_classes=n_classes,
        nf=nf,
        use_incremental_head=True,
        initial_classes=initial_classes,
    )


@Registry.register_model("resnet18")
def make_resnet18(
    n_classes: int = 200,
    nf: int = 64,
    initial_classes: int = 0,
    **kwargs,
) -> ResNet:
    """Standard ResNet-18 for TinyImageNet-scale benchmarks."""
    return ResNet(
        block=BasicBlock,
        num_blocks=[2, 2, 2, 2],
        n_classes=n_classes,
        nf=nf,
        use_incremental_head=True,
        initial_classes=initial_classes,
    )
