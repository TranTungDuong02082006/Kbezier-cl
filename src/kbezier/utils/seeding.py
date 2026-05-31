"""
Deterministic seeding for full reproducibility.

Covers: torch, numpy, cuda, cudnn, and DataLoader workers.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """
    Set seeds for all random number generators.

    Args:
        seed: Integer seed value.
        deterministic: If True, set cudnn to deterministic mode (slower but reproducible).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi-GPU

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_dataloader_generator(seed: int) -> torch.Generator:
    """
    Create a torch.Generator with a fixed seed for DataLoader.

    Pass this as the `generator` argument to DataLoader to ensure
    reproducible batch ordering:

        loader = DataLoader(dataset, generator=get_dataloader_generator(42))
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def worker_init_fn(worker_id: int) -> None:
    """
    Per-worker seeding for multiprocess DataLoader.

    Pass this as `worker_init_fn` to DataLoader:

        loader = DataLoader(dataset, num_workers=4, worker_init_fn=worker_init_fn)

    Each worker gets a deterministic but distinct seed based on the
    base seed (from torch initial seed) and worker_id.
    """
    # torch sets a unique seed per worker based on base_seed + worker_id
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
