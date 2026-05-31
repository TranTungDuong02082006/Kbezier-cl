"""
CL evaluation metrics computed from accuracy matrix R[i,j].

R[i,j] = accuracy on task j after training through task i.

All metrics computed centrally (not scattered in trainer) to avoid
definition inconsistencies across methods.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch


def compute_cl_metrics(
    accuracy_matrix: torch.Tensor,
    up_to_task: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Compute standard CL metrics from accuracy matrix.

    Args:
        accuracy_matrix: R[i,j] tensor of shape (T, T).
        up_to_task: If specified, compute metrics only up to this task.
                    Default: use all tasks (T-1).

    Returns:
        Dict with: acc, bwt, fwt, forgetting (per-task list), avg_forgetting.
    """
    T = accuracy_matrix.size(0)
    if up_to_task is None:
        up_to_task = T - 1
    t = up_to_task  # shorthand

    # ── Average Accuracy ──
    # ACC = (1 / (t+1)) Σ_j R[t, j]
    acc = accuracy_matrix[t, :t + 1].mean().item()

    # ── Backward Transfer ──
    # BWT = (1/t) Σ_{j<t} (R[t,j] - R[j,j])
    if t > 0:
        bwt_vals = [
            (accuracy_matrix[t, j] - accuracy_matrix[j, j]).item()
            for j in range(t)
        ]
        bwt = sum(bwt_vals) / len(bwt_vals)
    else:
        bwt = 0.0

    # ── Forward Transfer ──
    # FWT = (1/t) Σ_{j>0} R[j-1, j]
    # Measures zero-shot performance on future tasks
    if t > 0:
        fwt_vals = [
            accuracy_matrix[j - 1, j].item()
            for j in range(1, t + 1)
        ]
        fwt = sum(fwt_vals) / len(fwt_vals)
    else:
        fwt = 0.0

    # ── Per-task Forgetting ──
    # f_j = max_{i ≤ j} R[i,j] - R[t,j]
    forgetting = []
    for j in range(t + 1):
        max_acc_j = accuracy_matrix[:j + 1, j].max().item()
        final_acc_j = accuracy_matrix[t, j].item()
        forgetting.append(max_acc_j - final_acc_j)

    avg_forgetting = sum(forgetting) / max(len(forgetting), 1)

    return {
        "acc": acc,
        "bwt": bwt,
        "fwt": fwt,
        "forgetting_per_task": forgetting,
        "avg_forgetting": avg_forgetting,
        "n_tasks_seen": t + 1,
    }
