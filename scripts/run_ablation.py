"""
Ablation sweep: run experiments over a grid of hyperparameters.

Usage:
    python scripts/run_ablation.py --config ablation/gamma_sweep benchmark/split_cifar100
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from kbezier.engine.config import (
    get_nested,
    merge_configs,
    set_nested,
)


def main():
    parser = argparse.ArgumentParser(description="K-Bézier Ablation Sweep")
    parser.add_argument(
        "--config",
        nargs="+",
        required=True,
        help="Config names including ablation config",
    )
    parser.add_argument("--config-dir", type=str, default=str(project_root / "configs"))
    parser.add_argument("--dry-run", action="store_true", help="Print configs without running")
    args = parser.parse_args()

    config = merge_configs(args.config, args.config_dir)
    ablation_cfg = config.get("ablation", {})

    sweep_param = ablation_cfg.get("sweep_param")
    values = ablation_cfg.get("values", [])
    base_method = ablation_cfg.get("base_method", "kbezier")
    seeds = ablation_cfg.get("seeds", [1, 2, 3, 4, 5])

    if not sweep_param or not values:
        print("ERROR: ablation config must specify sweep_param and values")
        sys.exit(1)

    print(f"Ablation sweep: {sweep_param} = {values}")
    print(f"Seeds: {seeds}")
    print(f"Total runs: {len(values) * len(seeds)}")
    print()

    # Merge base method config
    method_config = merge_configs([f"method/{base_method}"], args.config_dir)
    base_config = copy.deepcopy(config)
    for k, v in method_config.items():
        if k not in base_config:
            base_config[k] = v
        elif isinstance(v, dict) and isinstance(base_config.get(k), dict):
            base_config[k].update(v)

    for val in values:
        for seed in seeds:
            run_config = copy.deepcopy(base_config)
            set_nested(run_config, sweep_param, val)
            set_nested(run_config, "seed", seed)

            label = f"{sweep_param}={val}, seed={seed}"

            if args.dry_run:
                print(f"  [DRY] {label}")
                continue

            print(f"\n{'='*60}")
            print(f"  Running: {label}")
            print(f"{'='*60}")

            # Import and run (inline to avoid circular imports)
            from kbezier.engine.config import get_nested as gn
            from kbezier.engine.registry import Registry
            from kbezier.engine.trainer import Trainer
            from kbezier.utils.seeding import seed_everything

            import kbezier.models.mlp
            import kbezier.models.resnet18
            import kbezier.benchmarks.loaders
            import kbezier.strategies.base_strategy
            import kbezier.strategies.regularization
            import kbezier.strategies.kbezier_strategy

            seed_everything(seed)

            benchmark_name = gn(run_config, "benchmark.name")
            builder = Registry.get_benchmark(benchmark_name)
            task_datasets = builder(run_config)

            model_name = gn(run_config, "model.name", "mlp")
            model_cls = Registry.get_model(model_name)
            model_kwargs = dict(run_config.get("model", {}))
            model_kwargs.pop("name", None)
            model_kwargs["initial_classes"] = len(task_datasets[0].get("classes", []))
            import inspect
            sig = inspect.signature(model_cls)
            has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
            if not has_var_keyword:
                model_kwargs = {k: v for k, v in model_kwargs.items() if k in sig.parameters}
            model = model_cls(**model_kwargs)

            scenario = gn(run_config, "benchmark.scenario", "task-IL")
            if scenario == "class-IL" and hasattr(model, "add_classes"):
                for t in range(1, len(task_datasets)):
                    model.add_classes(len(task_datasets[t].get("classes", [])))

            method_name = gn(run_config, "method.name", "kbezier")
            strategy_cls = Registry.get_strategy(method_name)
            device = gn(run_config, "device", "cuda")
            strategy = strategy_cls(model, run_config, device=device)

            trainer = Trainer(run_config, model, strategy, task_datasets, device)
            results = trainer.train()

            print(f"  → ACC={results['metrics']['acc']:.4f}, BWT={results['metrics']['bwt']:.4f}")

    print("\nAblation sweep complete.")


if __name__ == "__main__":
    main()
