"""
Main training loop for continual learning experiments.

Iterates over tasks, delegates to Strategy for before/after task hooks
and per-batch training. Builds accuracy matrix R[i,j] for CL metrics.
Tracks compute (wall-clock, forward/backward counts).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from kbezier.engine.config import (
    build_run_name,
    config_hash,
    get_nested,
    save_resolved_config,
)
from kbezier.evaluation.compute_tracker import ComputeTracker
from kbezier.evaluation.metrics import compute_cl_metrics
from kbezier.utils.checkpoint import save_checkpoint
from kbezier.utils.logging import setup_logger
from kbezier.utils.seeding import (
    get_dataloader_generator,
    seed_everything,
    worker_init_fn,
)


class Trainer:
    """
    CL Trainer: iterates tasks, trains via Strategy, builds accuracy matrix.

    The trainer is agnostic to the CL method — all method-specific logic
    lives in the Strategy (before_task, observe, after_task).
    """

    def __init__(
        self,
        config: dict,
        model: nn.Module,
        strategy,  # BaseStrategy instance
        task_datasets: List[Dict[str, Any]],  # [{train: Dataset, test: Dataset, classes: [...]}]
        device: str = "cuda",
    ):
        self.config = config
        self.model = model.to(device)
        self.strategy = strategy
        self.task_datasets = task_datasets
        self.device = device
        self.n_tasks = len(task_datasets)

        # Config values
        self.seed = get_nested(config, "seed", 42)
        self.epochs_per_task = get_nested(config, "epochs_per_task", 50)
        self.batch_size = get_nested(config, "batch_size", 64)
        self.num_workers = get_nested(config, "num_workers", 4)
        self.log_interval = get_nested(config, "log_interval", 50)

        # Output directory
        self.run_name = build_run_name(config)
        log_dir = get_nested(config, "log_dir", "results")
        self.output_dir = Path(log_dir) / self.run_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Logger
        self.logger = setup_logger(
            log_dir=self.output_dir,
            config_hash=config_hash(config),
            seed=self.seed,
        )

        # Accuracy matrix R[i, j] = accuracy on task j after training on task i
        self.accuracy_matrix = torch.zeros(self.n_tasks, self.n_tasks)

        # Compute tracker
        self.compute_tracker = ComputeTracker()

        # Save resolved config
        save_resolved_config(config, self.output_dir)

    def _make_loader(self, dataset, shuffle: bool = True) -> DataLoader:
        """Create a DataLoader with reproducible seeding."""
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            generator=get_dataloader_generator(self.seed),
            worker_init_fn=worker_init_fn,
        )

    def train(self) -> Dict[str, Any]:
        """
        Run the full continual learning experiment.

        Returns:
            Dictionary with accuracy matrix and CL metrics.
        """
        self.logger.info(f"Starting experiment: {self.run_name}")
        self.logger.info(f"Tasks: {self.n_tasks}, Epochs/task: {self.epochs_per_task}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        total_start = time.time()

        for task_id in range(self.n_tasks):
            self._train_task(task_id)
            self._evaluate_all_tasks(task_id)

            # Log current CL metrics
            metrics = compute_cl_metrics(self.accuracy_matrix, up_to_task=task_id)
            self.logger.info(
                f"After task {task_id}: "
                f"ACC={metrics['acc']:.4f}, "
                f"BWT={metrics['bwt']:.4f}, "
                f"FWT={metrics['fwt']:.4f}"
            )

        total_time = time.time() - total_start
        self.compute_tracker.set_total_time(total_time)

        # Final results
        final_metrics = compute_cl_metrics(self.accuracy_matrix)
        results = {
            "accuracy_matrix": self.accuracy_matrix.tolist(),
            "metrics": final_metrics,
            "compute": self.compute_tracker.summary(),
            "run_name": self.run_name,
            "config_hash": config_hash(self.config),
        }

        # Save results
        self._save_results(results)
        self.logger.info(f"Experiment complete in {total_time:.1f}s")
        self.logger.info(f"Final ACC={final_metrics['acc']:.4f}, BWT={final_metrics['bwt']:.4f}")

        return results

    def _train_task(self, task_id: int) -> None:
        """Train on a single task."""
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Task {task_id}/{self.n_tasks - 1}")
        self.logger.info(f"{'='*60}")

        task_data = self.task_datasets[task_id]
        train_loader = self._make_loader(task_data["train"], shuffle=True)

        # Strategy hook: before task
        task_start = time.time()
        self.strategy.before_task(task_id, train_loader)

        # Training loop
        self.model.train()
        for epoch in range(self.epochs_per_task):
            epoch_loss = 0.0
            n_batches = 0

            for batch_idx, (x, y) in enumerate(train_loader):
                x, y = x.to(self.device), y.to(self.device)

                # Strategy handles the actual training step
                loss = self.strategy.observe(x, y, task_id)

                # Track compute
                self.compute_tracker.add_forward(1)
                self.compute_tracker.add_backward(1)

                epoch_loss += loss
                n_batches += 1

                if (batch_idx + 1) % self.log_interval == 0:
                    avg_loss = epoch_loss / n_batches
                    self.logger.info(
                        f"  Epoch {epoch+1}/{self.epochs_per_task}, "
                        f"Batch {batch_idx+1}/{len(train_loader)}, "
                        f"Loss: {avg_loss:.4f}"
                    )

            avg_loss = epoch_loss / max(n_batches, 1)
            self.logger.info(
                f"  Epoch {epoch+1}/{self.epochs_per_task} complete, "
                f"Avg Loss: {avg_loss:.4f}"
            )

        # Strategy hook: after task
        self.strategy.after_task(task_id, train_loader)

        task_time = time.time() - task_start
        self.compute_tracker.add_task_time(task_id, task_time)
        self.logger.info(f"  Task {task_id} took {task_time:.1f}s")

    @torch.no_grad()
    def _evaluate_all_tasks(self, current_task: int) -> None:
        """Evaluate on all tasks seen so far, filling accuracy matrix row."""
        self.model.eval()

        for task_id in range(current_task + 1):
            task_data = self.task_datasets[task_id]
            test_loader = self._make_loader(task_data["test"], shuffle=False)

            correct = 0
            total = 0
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)

                # For task-IL: mask to current task's classes
                # For class-IL: use all classes (handled by scenario)
                if "class_mask" in task_data and self.config.get("benchmark", {}).get("scenario") == "task-IL":
                    mask = task_data["class_mask"]
                    logits = logits[:, mask]
                    y = self._remap_labels(y, mask)

                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

            acc = correct / max(total, 1)
            self.accuracy_matrix[current_task, task_id] = acc

        self.logger.info(
            f"  Accuracies after task {current_task}: "
            + ", ".join(
                f"T{j}={self.accuracy_matrix[current_task, j]:.3f}"
                for j in range(current_task + 1)
            )
        )

    def _remap_labels(self, y: torch.Tensor, class_mask: list) -> torch.Tensor:
        """Remap global class labels to local [0, n_classes_in_task) for task-IL."""
        mapping = {c: i for i, c in enumerate(class_mask)}
        return torch.tensor([mapping[yi.item()] for yi in y], device=y.device)

    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save results to JSON."""
        path = self.output_dir / "results.json"
        with open(path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        self.logger.info(f"Results saved to {path}")

        # Also save accuracy matrix as CSV for easy plotting
        import pandas as pd
        df = pd.DataFrame(
            self.accuracy_matrix.numpy(),
            index=[f"after_task_{i}" for i in range(self.n_tasks)],
            columns=[f"task_{j}" for j in range(self.n_tasks)],
        )
        csv_path = self.output_dir / "accuracy_matrix.csv"
        df.to_csv(csv_path)
