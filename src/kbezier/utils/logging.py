"""
Structured logging with run metadata.

Provides a configured logger that includes config hash, seed, and
method name in every log line for easy filtering.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "kbezier",
    log_dir: Optional[str | Path] = None,
    level: int = logging.INFO,
    config_hash: str = "",
    seed: int = 0,
) -> logging.Logger:
    """
    Set up a structured logger.

    Args:
        name: Logger name.
        log_dir: If provided, also log to a file in this directory.
        level: Logging level.
        config_hash: Config hash for identification.
        seed: Random seed for identification.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    fmt = f"[%(asctime)s][{config_hash[:8]}][s{seed}] %(levelname)s - %(message)s"
    formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler (if log_dir provided)
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / "train.log")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
