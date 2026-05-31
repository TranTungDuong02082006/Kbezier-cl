"""
Benchmark data loading: task-split logic, transforms, deterministic class ordering.

Supports: Permuted MNIST, Split-CIFAR100, Split-TinyImageNet.
All DataLoaders use seeded generators for reproducible batch ordering.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms

from kbezier.engine.registry import Registry


# ── Transforms ──────────────────────────────────────────────────────

def get_cifar100_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    return train_transform, test_transform


def get_mnist_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    return transform, transform


def get_tinyimagenet_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose([
        transforms.RandomCrop(64, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)),
    ])
    return train_transform, test_transform


# ── Permutation dataset wrapper ──────────────────────────────────────

class PermutedDataset(Dataset):
    """Wraps a dataset and applies a fixed pixel permutation."""

    def __init__(self, dataset: Dataset, permutation: torch.Tensor):
        self.dataset = dataset
        self.permutation = permutation

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        # x: (C, H, W) → flatten → permute → reshape
        shape = x.shape
        x_flat = x.view(-1)
        x_perm = x_flat[self.permutation]
        x = x_perm.view(shape)
        return x, y


# ── Task splitters ──────────────────────────────────────────────────

def _get_class_order(n_classes: int, seed: int) -> List[int]:
    """Deterministic class ordering from seed."""
    rng = np.random.RandomState(seed)
    order = list(range(n_classes))
    rng.shuffle(order)
    return order


def _split_by_classes(
    dataset: Dataset,
    classes: List[int],
) -> Subset:
    """Return subset of dataset containing only specified classes."""
    if hasattr(dataset, 'targets'):
        targets = np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):
        targets = np.array(dataset.labels)
    else:
        # Fallback: iterate (slow but universal)
        targets = np.array([dataset[i][1] for i in range(len(dataset))])

    mask = np.isin(targets, classes)
    indices = np.where(mask)[0].tolist()
    return Subset(dataset, indices)


# ── Benchmark builders ──────────────────────────────────────────────

@Registry.register_benchmark("perm_mnist")
def build_perm_mnist(config: dict) -> List[Dict[str, Any]]:
    """
    Permuted MNIST: each task applies a different pixel permutation.
    Domain-IL: same 10 classes, different input distribution.
    """
    n_tasks = config.get("benchmark", {}).get("n_tasks", 10)
    seed = config.get("seed", 42)
    data_dir = config.get("data_dir", "./data")

    train_transform, test_transform = get_mnist_transforms()
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=test_transform)

    rng = np.random.RandomState(seed)
    n_pixels = 28 * 28
    task_datasets = []

    for t in range(n_tasks):
        if t == 0:
            perm = torch.arange(n_pixels)  # identity for task 0
        else:
            perm = torch.tensor(rng.permutation(n_pixels))

        task_datasets.append({
            "train": PermutedDataset(train_dataset, perm),
            "test": PermutedDataset(test_dataset, perm),
            "classes": list(range(10)),
            "class_mask": list(range(10)),
            "task_id": t,
        })

    return task_datasets


@Registry.register_benchmark("split_cifar100")
def build_split_cifar100(config: dict) -> List[Dict[str, Any]]:
    """
    Split CIFAR-100: divide 100 classes into tasks.
    Default: 20 tasks × 5 classes (Class-IL).
    """
    benchmark_cfg = config.get("benchmark", {})
    n_tasks = benchmark_cfg.get("n_tasks", 20)
    classes_per_task = benchmark_cfg.get("classes_per_task", 5)
    seed = config.get("seed", 42)
    data_dir = config.get("data_dir", "./data")

    train_transform, test_transform = get_cifar100_transforms()
    train_full = datasets.CIFAR100(data_dir, train=True, download=True, transform=train_transform)
    test_full = datasets.CIFAR100(data_dir, train=False, download=True, transform=test_transform)

    class_order = _get_class_order(100, seed)

    task_datasets = []
    for t in range(n_tasks):
        start = t * classes_per_task
        end = start + classes_per_task
        task_classes = class_order[start:end]

        task_datasets.append({
            "train": _split_by_classes(train_full, task_classes),
            "test": _split_by_classes(test_full, task_classes),
            "classes": task_classes,
            "class_mask": task_classes,
            "task_id": t,
        })

    return task_datasets


@Registry.register_benchmark("seq_cifar100_50t")
def build_seq_cifar100_50t(config: dict) -> List[Dict[str, Any]]:
    """
    Sequential CIFAR-100 with 50 tasks × 2 classes.
    Long-sequence benchmark for demonstrating K-Bézier's advantage.
    """
    config_copy = dict(config)
    config_copy.setdefault("benchmark", {})
    config_copy["benchmark"]["n_tasks"] = 50
    config_copy["benchmark"]["classes_per_task"] = 2
    return build_split_cifar100.__wrapped__(config_copy) if hasattr(build_split_cifar100, '__wrapped__') else _build_split_cifar100_impl(config_copy)


def _build_split_cifar100_impl(config: dict) -> List[Dict[str, Any]]:
    """Internal implementation shared by split_cifar100 and seq_cifar100_50t."""
    benchmark_cfg = config.get("benchmark", {})
    n_tasks = benchmark_cfg.get("n_tasks", 20)
    classes_per_task = benchmark_cfg.get("classes_per_task", 5)
    seed = config.get("seed", 42)
    data_dir = config.get("data_dir", "./data")

    train_transform, test_transform = get_cifar100_transforms()
    train_full = datasets.CIFAR100(data_dir, train=True, download=True, transform=train_transform)
    test_full = datasets.CIFAR100(data_dir, train=False, download=True, transform=test_transform)

    class_order = _get_class_order(100, seed)

    task_datasets = []
    for t in range(n_tasks):
        start = t * classes_per_task
        end = start + classes_per_task
        task_classes = class_order[start:end]

        task_datasets.append({
            "train": _split_by_classes(train_full, task_classes),
            "test": _split_by_classes(test_full, task_classes),
            "classes": task_classes,
            "class_mask": task_classes,
            "task_id": t,
        })

    return task_datasets


# Re-register with shared implementation
Registry._benchmarks["seq_cifar100_50t"] = lambda config: _build_split_cifar100_impl(
    {**config, "benchmark": {**config.get("benchmark", {}), "n_tasks": 50, "classes_per_task": 2}}
)


@Registry.register_benchmark("split_tinyimagenet")
def build_split_tinyimagenet(config: dict) -> List[Dict[str, Any]]:
    """
    Split TinyImageNet: 200 classes → 20 tasks × 10 classes.
    Requires data to be pre-downloaded or uses download_tinyimagenet().
    """
    benchmark_cfg = config.get("benchmark", {})
    n_tasks = benchmark_cfg.get("n_tasks", 20)
    classes_per_task = benchmark_cfg.get("classes_per_task", 10)
    seed = config.get("seed", 42)
    data_dir = benchmark_cfg.get("data_dir", "./data/tiny-imagenet-200")

    if not Path(data_dir).exists():
        download_tinyimagenet(Path(data_dir).parent)

    train_transform, test_transform = get_tinyimagenet_transforms()
    train_full = datasets.ImageFolder(
        os.path.join(data_dir, "train"), transform=train_transform
    )
    test_full = datasets.ImageFolder(
        os.path.join(data_dir, "val"), transform=test_transform
    )

    class_order = _get_class_order(200, seed)

    task_datasets = []
    for t in range(n_tasks):
        start = t * classes_per_task
        end = start + classes_per_task
        task_classes = class_order[start:end]

        task_datasets.append({
            "train": _split_by_classes(train_full, task_classes),
            "test": _split_by_classes(test_full, task_classes),
            "classes": task_classes,
            "class_mask": task_classes,
            "task_id": t,
        })

    return task_datasets


def download_tinyimagenet(data_dir: str | Path) -> None:
    """Download and extract TinyImageNet with md5 checksum verification."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_path = data_dir / "tiny-imagenet-200.zip"
    expected_md5 = "90528d7ca1a48142e341f4ef8d21d0de"

    if zip_path.exists():
        # Verify md5
        md5 = hashlib.md5(open(zip_path, "rb").read()).hexdigest()
        if md5 == expected_md5:
            print("TinyImageNet zip already downloaded and verified.")
        else:
            print("MD5 mismatch, re-downloading...")
            zip_path.unlink()

    if not zip_path.exists():
        print(f"Downloading TinyImageNet to {zip_path}...")
        urllib.request.urlretrieve(url, zip_path)

    # Extract
    extract_dir = data_dir / "tiny-imagenet-200"
    if not extract_dir.exists():
        print("Extracting...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(data_dir)
        print("Done.")
