"""
Tests for CL evaluation metrics.

Verifies ACC, BWT, FWT, forgetting on hand-crafted accuracy matrices.
"""

import torch
import pytest

from kbezier.evaluation.metrics import compute_cl_metrics


class TestCLMetrics:
    """Test CL metrics on known accuracy matrices."""

    def test_perfect_memory(self):
        """Perfect retention: no forgetting, BWT = 0."""
        # 3 tasks, all retain 100% accuracy
        R = torch.tensor([
            [1.0, 0.0, 0.0],  # after task 0: only task 0 evaluated
            [1.0, 1.0, 0.0],  # after task 1: tasks 0,1 at 100%
            [1.0, 1.0, 1.0],  # after task 2: all at 100%
        ])

        metrics = compute_cl_metrics(R)
        assert abs(metrics["acc"] - 1.0) < 1e-6, f"ACC should be 1.0, got {metrics['acc']}"
        assert abs(metrics["bwt"] - 0.0) < 1e-6, f"BWT should be 0.0, got {metrics['bwt']}"
        assert abs(metrics["avg_forgetting"] - 0.0) < 1e-6

    def test_complete_forgetting(self):
        """Complete forgetting: BWT should be very negative."""
        R = torch.tensor([
            [0.9, 0.0, 0.0],
            [0.0, 0.8, 0.0],  # forgot task 0 completely
            [0.0, 0.0, 0.7],  # forgot tasks 0,1 completely
        ])

        metrics = compute_cl_metrics(R)
        assert metrics["bwt"] < 0, "BWT should be negative for complete forgetting"
        # BWT = (1/2) * ((0-0.9) + (0-0.8)) = -0.85
        assert abs(metrics["bwt"] - (-0.85)) < 1e-6

    def test_single_task(self):
        """Single task: no forgetting possible."""
        R = torch.tensor([[0.95]])
        metrics = compute_cl_metrics(R)
        assert abs(metrics["acc"] - 0.95) < 1e-6
        assert abs(metrics["bwt"] - 0.0) < 1e-6
        assert abs(metrics["fwt"] - 0.0) < 1e-6

    def test_partial_forgetting(self):
        """Partial forgetting with forward transfer."""
        R = torch.tensor([
            [0.8, 0.0, 0.0],
            [0.6, 0.9, 0.0],  # task 0: 80% → 60% (forgot 20%)
            [0.5, 0.7, 0.85], # task 0: 60% → 50%, task 1: 90% → 70%
        ])

        metrics = compute_cl_metrics(R)

        # ACC = mean of last row's non-zero: (0.5 + 0.7 + 0.85) / 3
        expected_acc = (0.5 + 0.7 + 0.85) / 3
        assert abs(metrics["acc"] - expected_acc) < 1e-6

        # BWT = (1/2) * ((0.5 - 0.8) + (0.7 - 0.9)) = (1/2) * (-0.5) = -0.25
        expected_bwt = (-0.3 + -0.2) / 2
        assert abs(metrics["bwt"] - expected_bwt) < 1e-6

        # FWT = (1/2) * (R[0,1] + R[1,2]) = (0.0 + 0.0) / 2 = 0
        expected_fwt = (0.0 + 0.0) / 2
        assert abs(metrics["fwt"] - expected_fwt) < 1e-6

    def test_up_to_task(self):
        """Metrics computed only up to specified task."""
        R = torch.tensor([
            [0.9, 0.0, 0.0],
            [0.7, 0.85, 0.0],
            [0.5, 0.6, 0.8],
        ])

        # After task 1 only
        metrics = compute_cl_metrics(R, up_to_task=1)
        expected_acc = (0.7 + 0.85) / 2
        assert abs(metrics["acc"] - expected_acc) < 1e-6
        assert metrics["n_tasks_seen"] == 2

    def test_forgetting_per_task(self):
        """Per-task forgetting: f_j = max_{i≤j} R[i,j] - R[T,j]."""
        R = torch.tensor([
            [0.9, 0.0, 0.0],
            [0.8, 0.95, 0.0],
            [0.6, 0.7, 0.9],
        ])

        metrics = compute_cl_metrics(R)
        forgetting = metrics["forgetting_per_task"]

        # f_0 = max(0.9, 0.8, 0.6) - 0.6 = 0.3
        assert abs(forgetting[0] - 0.3) < 1e-6
        # f_1 = max(0.95, 0.7) - 0.7 = 0.25
        assert abs(forgetting[1] - 0.25) < 1e-6
        # f_2 = max(0.9) - 0.9 = 0.0
        assert abs(forgetting[2] - 0.0) < 1e-6
