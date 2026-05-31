"""
Main entry point for running experiments.

Usage:
    python scripts/run.py --config method/kbezier benchmark/split_cifar100
    python scripts/run.py --config method/ewc benchmark/perm_mnist --seed 1 --device cpu
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from kbezier.engine.config import get_nested, merge_configs, set_nested
from kbezier.engine.registry import Registry
from kbezier.engine.trainer import Trainer
from kbezier.utils.seeding import seed_everything

# Import all modules to trigger @register decorators
import kbezier.models.mlp
import kbezier.models.resnet18
import kbezier.benchmarks.loaders
import kbezier.strategies.base_strategy
import kbezier.strategies.regularization
import kbezier.strategies.kbezier_strategy


def main():
    parser = argparse.ArgumentParser(description="K-Bézier CL Experiments")
    parser.add_argument(
        "--config",
        nargs="+",
        required=True,
        help="Config names to merge (e.g., method/kbezier benchmark/split_cifar100)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override seed")
    parser.add_argument("--device", type=str, default=None, help="Override device")
    parser.add_argument(
        "--config-dir",
        type=str,
        default=str(project_root / "configs"),
        help="Config directory",
    )
    args = parser.parse_args()

    # Merge configs
    config = merge_configs(args.config, args.config_dir)

    # Apply overrides
    if args.seed is not None:
        set_nested(config, "seed", args.seed)
    if args.device is not None:
        set_nested(config, "device", args.device)

    seed = get_nested(config, "seed", 42)
    device = get_nested(config, "device", "cuda")

    # Seed everything
    seed_everything(seed)

    # Build benchmark
    benchmark_name = get_nested(config, "benchmark.name")
    if benchmark_name is None:
        raise ValueError("No benchmark specified in config")
    benchmark_builder = Registry.get_benchmark(benchmark_name)
    task_datasets = benchmark_builder(config)

    # Build model
    model_name = get_nested(config, "model.name", "mlp")
    model_cls = Registry.get_model(model_name)

    model_kwargs = dict(config.get("model", {}))
    model_kwargs.pop("name", None)

    # For incremental head: start with first task's classes
    if "initial_classes" not in model_kwargs:
        first_classes = task_datasets[0].get("classes", [])
        model_kwargs["initial_classes"] = len(first_classes)

    if callable(model_cls):
        model = model_cls(**model_kwargs)
    else:
        model = model_cls(**model_kwargs)

    # Expand head for all tasks if class-IL
    scenario = get_nested(config, "benchmark.scenario", "task-IL")
    if scenario == "class-IL":
        # Add remaining classes
        for t in range(1, len(task_datasets)):
            n_new = len(task_datasets[t].get("classes", []))
            if hasattr(model, "add_classes"):
                model.add_classes(n_new)

    # Build strategy
    method_name = get_nested(config, "method.name", "finetune")
    strategy_cls = Registry.get_strategy(method_name)
    strategy = strategy_cls(model, config, device=device)

    # Build trainer and run
    trainer = Trainer(
        config=config,
        model=model,
        strategy=strategy,
        task_datasets=task_datasets,
        device=device,
    )

    results = trainer.train()

    print(f"\n{'='*60}")
    print(f"FINAL RESULTS: {method_name} on {benchmark_name}")
    print(f"{'='*60}")
    print(f"  ACC: {results['metrics']['acc']:.4f}")
    print(f"  BWT: {results['metrics']['bwt']:.4f}")
    print(f"  FWT: {results['metrics']['fwt']:.4f}")
    print(f"  Avg Forgetting: {results['metrics']['avg_forgetting']:.4f}")
    print(f"  Total Time: {results['compute']['total_time_seconds']:.1f}s")
    print(f"  Results saved to: {trainer.output_dir}")


if __name__ == "__main__":
    main()
