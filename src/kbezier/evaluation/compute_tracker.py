"""
Compute fairness tracker: wall-clock, forward/backward counts, Fisher cost.

Ensures reviewer can verify "time only slightly more" claim quantitatively.
Logs fisher_update_freq since it's the compute vs accuracy tradeoff knob.
"""

from __future__ import annotations

from typing import Any, Dict, List


class ComputeTracker:
    """
    Track computational costs across methods for fair comparison.

    Metrics tracked:
    - Wall-clock time per task and total
    - Forward pass count, backward pass count
    - Fisher accumulation time (separated from training)
    """

    def __init__(self):
        self._task_times: Dict[int, float] = {}
        self._total_time: float = 0.0
        self._forward_count: int = 0
        self._backward_count: int = 0
        self._fisher_accum_time: float = 0.0
        self._fisher_accum_count: int = 0

    def add_task_time(self, task_id: int, seconds: float) -> None:
        self._task_times[task_id] = seconds

    def set_total_time(self, seconds: float) -> None:
        self._total_time = seconds

    def add_forward(self, count: int = 1) -> None:
        self._forward_count += count

    def add_backward(self, count: int = 1) -> None:
        self._backward_count += count

    def add_fisher_time(self, seconds: float) -> None:
        self._fisher_accum_time += seconds
        self._fisher_accum_count += 1

    def summary(self) -> Dict[str, Any]:
        """Return summary for JSON serialization."""
        return {
            "total_time_seconds": self._total_time,
            "task_times": self._task_times,
            "forward_passes": self._forward_count,
            "backward_passes": self._backward_count,
            "fisher_accumulation_time": self._fisher_accum_time,
            "fisher_accumulation_count": self._fisher_accum_count,
            "training_time_excluding_fisher": self._total_time - self._fisher_accum_time,
        }

    def comparison_row(self, method_name: str) -> Dict[str, Any]:
        """Return a row for the compute comparison table."""
        return {
            "method": method_name,
            "wall_clock_s": round(self._total_time, 1),
            "fwd_passes": self._forward_count,
            "bwd_passes": self._backward_count,
            "fisher_time_s": round(self._fisher_accum_time, 1),
            "train_time_s": round(self._total_time - self._fisher_accum_time, 1),
        }
