"""
Checkpoint management: save/load model, optimizer, Fisher state, and Bézier anchors.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    fisher_state: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save a training checkpoint.

    Args:
        path: File path to save.
        model: Model to save.
        optimizer: Optional optimizer state.
        fisher_state: Optional Fisher metric state dict.
        extra: Any additional metadata (task_id, epoch, etc.).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "model_state_dict": model.state_dict(),
    }
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    if fisher_state is not None:
        state["fisher_state"] = fisher_state
    if extra is not None:
        state["extra"] = extra

    torch.save(state, path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Load a training checkpoint.

    Args:
        path: Checkpoint file path.
        model: Model to load weights into.
        optimizer: Optional optimizer to load state into.
        device: Device to map tensors to.

    Returns:
        Dictionary containing fisher_state and extra metadata (if present).
    """
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])

    result = {}
    if "fisher_state" in state:
        result["fisher_state"] = state["fisher_state"]
    if "extra" in state:
        result["extra"] = state["extra"]

    return result


def save_anchor(
    path: str | Path,
    params: Dict[str, torch.Tensor],
    label: str = "anchor",
) -> None:
    """
    Save a parameter snapshot (Bézier anchor) to disk.

    Used for disk-offloading: only the current pair's anchors live in GPU RAM,
    previous anchors are saved to disk.

    Args:
        path: File path.
        params: Dictionary of {param_name: tensor} (detached, CPU).
        label: Descriptive label for logging.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Move to CPU for storage
    cpu_params = {k: v.detach().cpu() for k, v in params.items()}
    torch.save({"params": cpu_params, "label": label}, path)


def load_anchor(
    path: str | Path,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    Load a parameter snapshot (Bézier anchor) from disk.

    Args:
        path: File path.
        device: Device to load tensors to.

    Returns:
        Dictionary of {param_name: tensor}.
    """
    state = torch.load(path, map_location=device, weights_only=False)
    return state["params"]
