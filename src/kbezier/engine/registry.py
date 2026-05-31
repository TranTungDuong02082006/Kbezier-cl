"""
Central registry for models, strategies, and benchmarks.

Usage:
    from kbezier.engine.registry import Registry

    @Registry.register_model("mlp")
    class MLP(nn.Module): ...

    model_cls = Registry.get_model("mlp")
"""

from __future__ import annotations

from typing import Any, Dict, Type


class Registry:
    """Singleton registry mapping names → classes for models, strategies, benchmarks."""

    _models: Dict[str, Type] = {}
    _strategies: Dict[str, Type] = {}
    _benchmarks: Dict[str, Type] = {}

    # ── Model registration ──────────────────────────────────────────

    @classmethod
    def register_model(cls, name: str):
        """Decorator to register a model class."""
        def decorator(model_cls: Type) -> Type:
            if name in cls._models:
                raise ValueError(f"Model '{name}' already registered.")
            cls._models[name] = model_cls
            return model_cls
        return decorator

    @classmethod
    def get_model(cls, name: str) -> Type:
        if name not in cls._models:
            raise KeyError(
                f"Model '{name}' not found. Available: {list(cls._models.keys())}"
            )
        return cls._models[name]

    @classmethod
    def list_models(cls) -> list[str]:
        return list(cls._models.keys())

    # ── Strategy registration ───────────────────────────────────────

    @classmethod
    def register_strategy(cls, name: str):
        """Decorator to register a CL strategy class."""
        def decorator(strategy_cls: Type) -> Type:
            if name in cls._strategies:
                raise ValueError(f"Strategy '{name}' already registered.")
            cls._strategies[name] = strategy_cls
            return strategy_cls
        return decorator

    @classmethod
    def get_strategy(cls, name: str) -> Type:
        if name not in cls._strategies:
            raise KeyError(
                f"Strategy '{name}' not found. Available: {list(cls._strategies.keys())}"
            )
        return cls._strategies[name]

    @classmethod
    def list_strategies(cls) -> list[str]:
        return list(cls._strategies.keys())

    # ── Benchmark registration ──────────────────────────────────────

    @classmethod
    def register_benchmark(cls, name: str):
        """Decorator to register a benchmark class."""
        def decorator(bench_cls: Type) -> Type:
            if name in cls._benchmarks:
                raise ValueError(f"Benchmark '{name}' already registered.")
            cls._benchmarks[name] = bench_cls
            return bench_cls
        return decorator

    @classmethod
    def get_benchmark(cls, name: str) -> Type:
        if name not in cls._benchmarks:
            raise KeyError(
                f"Benchmark '{name}' not found. Available: {list(cls._benchmarks.keys())}"
            )
        return cls._benchmarks[name]

    @classmethod
    def list_benchmarks(cls) -> list[str]:
        return list(cls._benchmarks.keys())

    # ── Utility ─────────────────────────────────────────────────────

    @classmethod
    def clear(cls):
        """Clear all registrations (for testing)."""
        cls._models.clear()
        cls._strategies.clear()
        cls._benchmarks.clear()

    @classmethod
    def summary(cls) -> Dict[str, list[str]]:
        """Return a summary of all registered components."""
        return {
            "models": cls.list_models(),
            "strategies": cls.list_strategies(),
            "benchmarks": cls.list_benchmarks(),
        }
