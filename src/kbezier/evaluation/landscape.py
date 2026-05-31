"""
Loss landscape and mode connectivity evaluation.

Computes loss barriers along linear and Bézier paths between task anchors.
This produces the "money figure" of the paper: linear path barriers
increase with T, while Bézier curves maintain low barriers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.func import functional_call
from torch.utils.data import DataLoader


def compute_loss_barrier(
    model: nn.Module,
    path,  # BezierCurve or LinearPath
    data_loader: DataLoader,
    criterion: Optional[nn.Module] = None,
    n_points: int = 21,
    device: str = "cuda",
) -> Dict[str, list]:
    """
    Compute loss along an interpolation path.

    Args:
        model: Network architecture (used as template for functional_call).
        path: BezierCurve or LinearPath with interpolate(theta) method.
        data_loader: Data to evaluate loss on.
        criterion: Loss function. Default: CrossEntropyLoss.
        n_points: Number of θ values to evaluate.
        device: Device.

    Returns:
        Dict with 'thetas', 'losses', 'accuracies', 'barrier'.
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    thetas = [i / (n_points - 1) for i in range(n_points)]
    losses = []
    accuracies = []

    model.eval()
    model.to(device)

    for theta in thetas:
        phi_theta = path.interpolate(theta)
        # Move to device
        phi_theta = {k: v.to(device) for k, v in phi_theta.items()}

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in data_loader:
                x, y = x.to(device), y.to(device)
                logits = functional_call(model, phi_theta, (x,))
                loss = criterion(logits, y)

                total_loss += loss.item() * x.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += x.size(0)

        avg_loss = total_loss / max(total, 1)
        acc = correct / max(total, 1)
        losses.append(avg_loss)
        accuracies.append(acc)

    # Loss barrier = max loss along path - max of endpoint losses
    endpoint_max = max(losses[0], losses[-1])
    barrier = max(losses) - endpoint_max

    return {
        "thetas": thetas,
        "losses": losses,
        "accuracies": accuracies,
        "barrier": barrier,
        "max_loss": max(losses),
        "endpoint_losses": (losses[0], losses[-1]),
    }


def compute_connectivity_profile(
    model: nn.Module,
    anchor_dir: str | Path,
    task_pairs: List[Tuple[int, int]],
    data_loaders: List[DataLoader],
    device: str = "cuda",
    n_points: int = 21,
) -> Dict[str, list]:
    """
    Compute loss barriers for multiple task pairs (for the long-sequence plot).

    Args:
        model: Network architecture.
        anchor_dir: Directory containing saved Bézier anchors.
        task_pairs: List of (task_i, task_j) pairs to evaluate.
        data_loaders: Test loaders per task.
        device: Device.
        n_points: Number of θ per path.

    Returns:
        Dict with per-pair barriers and metadata.
    """
    from kbezier.connectivity.bezier import BezierCurve
    from kbezier.connectivity.linear_lmc import LinearPath

    anchor_dir = Path(anchor_dir)
    results = []

    for (t_start, t_end) in task_pairs:
        # Load Bézier anchors
        bezier_path_file = anchor_dir / f"bezier_task_{t_end}.pt"
        if bezier_path_file.exists():
            bezier_curve = BezierCurve.load_anchors(bezier_path_file, device=device)

            # Also create linear path from same endpoints
            w0 = {name: bezier_curve._get_anchor("w0", name) for name in bezier_curve._w0_keys}
            w1 = {name: bezier_curve._get_anchor("w1", name) for name in bezier_curve._w0_keys}
            linear_path = LinearPath(w0, w1).to(device)

            # Evaluate on all tasks up to t_end
            for eval_task in range(t_end + 1):
                if eval_task < len(data_loaders):
                    bezier_result = compute_loss_barrier(
                        model, bezier_curve, data_loaders[eval_task],
                        n_points=n_points, device=device,
                    )
                    linear_result = compute_loss_barrier(
                        model, linear_path, data_loaders[eval_task],
                        n_points=n_points, device=device,
                    )

                    results.append({
                        "pair": (t_start, t_end),
                        "eval_task": eval_task,
                        "bezier_barrier": bezier_result["barrier"],
                        "linear_barrier": linear_result["barrier"],
                        "bezier_losses": bezier_result["losses"],
                        "linear_losses": linear_result["losses"],
                    })

    return {"profiles": results}
