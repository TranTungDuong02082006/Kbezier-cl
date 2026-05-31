"""
Configuration system: YAML parsing, merging, validation.

Supports merging multiple config files in order (base → benchmark → method → ablation).
Resolved configs are saved alongside results for reproducibility.
"""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. Override wins on conflicts."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_yaml(path: str | Path) -> dict:
    """Load a single YAML file."""
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def merge_configs(config_names: List[str], config_dir: str | Path = "configs") -> dict:
    """
    Merge multiple config files in order.

    Args:
        config_names: List of config names (e.g. ["method/kbezier", "benchmark/split_cifar100"]).
                      Each is resolved relative to config_dir with .yaml extension.
        config_dir: Root config directory.

    Returns:
        Merged configuration dictionary.
    """
    config_dir = Path(config_dir)
    # Always start with base.yaml
    merged = load_yaml(config_dir / "base.yaml")

    for name in config_names:
        path = config_dir / f"{name}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        override = load_yaml(path)
        merged = _deep_merge(merged, override)

    return merged


def config_hash(config: dict) -> str:
    """Compute a deterministic hash of the config for run identification."""
    serialized = json.dumps(config, sort_keys=True, default=str)
    return hashlib.md5(serialized.encode()).hexdigest()[:12]


def get_git_hash() -> Optional[str]:
    """Get current git commit hash, or None if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def save_resolved_config(config: dict, output_dir: str | Path) -> Path:
    """
    Save the fully resolved config + metadata to the output directory.

    Returns the path to the saved config file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Add metadata
    config_with_meta = copy.deepcopy(config)
    config_with_meta["_meta"] = {
        "config_hash": config_hash(config),
        "git_hash": get_git_hash(),
    }

    path = output_dir / "resolved_config.yaml"
    with open(path, "w") as f:
        yaml.dump(config_with_meta, f, default_flow_style=False, sort_keys=False)

    return path


# ── Convenience accessors ────────────────────────────────────────────────

def get_nested(config: dict, dotted_key: str, default: Any = None) -> Any:
    """
    Access nested config values with dot notation.

    Example:
        get_nested(config, "method.fisher.damping", 1e-3)
    """
    keys = dotted_key.split(".")
    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value


def set_nested(config: dict, dotted_key: str, value: Any) -> None:
    """
    Set nested config values with dot notation.

    Example:
        set_nested(config, "method.gamma", 0.5)
    """
    keys = dotted_key.split(".")
    d = config
    for key in keys[:-1]:
        if key not in d or not isinstance(d[key], dict):
            d[key] = {}
        d = d[key]
    d[keys[-1]] = value


def build_run_name(config: dict) -> str:
    """Build a descriptive run name from config."""
    parts = []
    benchmark_name = get_nested(config, "benchmark.name", "unknown")
    method_name = get_nested(config, "method.name", "unknown")
    seed = get_nested(config, "seed", 0)
    parts.append(benchmark_name)
    parts.append(method_name)
    parts.append(f"s{seed}")
    return "_".join(parts)
