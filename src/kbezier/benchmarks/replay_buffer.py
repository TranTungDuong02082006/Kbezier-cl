"""
Replay buffer with reservoir sampling for path-loss computation.

The buffer stores a fixed number of samples per task, used by the
Bézier path-loss term to evaluate L_≤t along the curve.
Memory budget is configurable for fair comparison with baselines.
"""

from __future__ import annotations

import random
from typing import List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset


class ReplayBuffer:
    """
    Reservoir sampling replay buffer.

    Stores up to `buffer_size` samples per task. When the buffer for a
    task is full, new samples replace old ones with decreasing probability
    (reservoir sampling), ensuring a uniform random subset.
    """

    def __init__(self, buffer_size_per_task: int = 200):
        """
        Args:
            buffer_size_per_task: Maximum samples to store per task.
        """
        self.buffer_size_per_task = buffer_size_per_task

        # Storage: {task_id: (x_tensor, y_tensor)}
        self._buffers: dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        # Count of samples seen per task (for reservoir sampling)
        self._counts: dict[int, int] = {}

    def update(self, task_id: int, x: torch.Tensor, y: torch.Tensor) -> None:
        """
        Add a batch of samples to the buffer using reservoir sampling.

        Args:
            task_id: Task identifier.
            x: Input tensor (batch, ...).
            y: Label tensor (batch,).
        """
        x, y = x.detach().cpu(), y.detach().cpu()
        batch_size = x.size(0)

        if task_id not in self._buffers:
            # First batch: take up to buffer_size
            n_take = min(batch_size, self.buffer_size_per_task)
            self._buffers[task_id] = (x[:n_take].clone(), y[:n_take].clone())
            self._counts[task_id] = batch_size
            # If batch was larger than buffer, do reservoir for the rest
            if batch_size > self.buffer_size_per_task:
                for i in range(self.buffer_size_per_task, batch_size):
                    j = random.randint(0, i)
                    if j < self.buffer_size_per_task:
                        self._buffers[task_id][0][j] = x[i]
                        self._buffers[task_id][1][j] = y[i]
            return

        buf_x, buf_y = self._buffers[task_id]
        current_count = self._counts[task_id]

        for i in range(batch_size):
            current_count += 1
            if buf_x.size(0) < self.buffer_size_per_task:
                # Buffer not full yet: append
                buf_x = torch.cat([buf_x, x[i:i+1]], dim=0)
                buf_y = torch.cat([buf_y, y[i:i+1]], dim=0)
            else:
                # Reservoir sampling
                j = random.randint(0, current_count - 1)
                if j < self.buffer_size_per_task:
                    buf_x[j] = x[i]
                    buf_y[j] = y[i]

        self._buffers[task_id] = (buf_x, buf_y)
        self._counts[task_id] = current_count

    def add_task_data(self, task_id: int, data_loader: DataLoader) -> None:
        """
        Convenience: fill buffer from a DataLoader.
        Iterates through the loader using reservoir sampling.
        """
        for x, y in data_loader:
            self.update(task_id, x, y)

    def get_task_data(self, task_id: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Get stored data for a specific task."""
        return self._buffers.get(task_id)

    def get_all_data(self, up_to_task: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get concatenated data from all tasks (or up to a specific task).

        Args:
            up_to_task: If specified, only include tasks 0..up_to_task (inclusive).

        Returns:
            (x, y) tensors concatenated across tasks.
        """
        all_x, all_y = [], []
        for tid in sorted(self._buffers.keys()):
            if up_to_task is not None and tid > up_to_task:
                break
            buf_x, buf_y = self._buffers[tid]
            all_x.append(buf_x)
            all_y.append(buf_y)

        if not all_x:
            raise ValueError("Replay buffer is empty.")

        return torch.cat(all_x, dim=0), torch.cat(all_y, dim=0)

    def sample(
        self,
        batch_size: int,
        up_to_task: Optional[int] = None,
        device: str = "cpu",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample a random batch from the buffer.

        Args:
            batch_size: Number of samples to draw.
            up_to_task: Only sample from tasks 0..up_to_task.
            device: Device to place returned tensors on.

        Returns:
            (x, y) batch on the specified device.
        """
        all_x, all_y = self.get_all_data(up_to_task)
        n = all_x.size(0)
        indices = torch.randperm(n)[:min(batch_size, n)]
        return all_x[indices].to(device), all_y[indices].to(device)

    def __len__(self) -> int:
        return sum(buf[0].size(0) for buf in self._buffers.values())

    @property
    def n_tasks(self) -> int:
        return len(self._buffers)

    def total_memory_bytes(self) -> int:
        """Estimate total memory usage in bytes."""
        total = 0
        for buf_x, buf_y in self._buffers.values():
            total += buf_x.nelement() * buf_x.element_size()
            total += buf_y.nelement() * buf_y.element_size()
        return total
