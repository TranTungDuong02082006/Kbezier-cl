"""
Statistical analysis: multi-seed aggregation, significance tests.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from scipy import stats


def aggregate_seeds(
    results_list: List[Dict[str, float]],
    metric_keys: List[str] | None = None,
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate results across multiple seeds.

    Args:
        results_list: List of result dicts (one per seed).
        metric_keys: Which metrics to aggregate. Default: all numeric keys.

    Returns:
        Dict mapping metric_name → {mean, std, min, max, n_seeds}.
    """
    if not results_list:
        return {}

    if metric_keys is None:
        metric_keys = [
            k for k in results_list[0]
            if isinstance(results_list[0][k], (int, float))
        ]

    aggregated = {}
    for key in metric_keys:
        values = [r[key] for r in results_list if key in r]
        if values:
            aggregated[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "n_seeds": len(values),
            }

    return aggregated


def paired_t_test(
    values_a: List[float],
    values_b: List[float],
    alternative: str = "two-sided",
) -> Dict[str, float]:
    """
    Paired t-test for significance between two methods (same seeds).

    Args:
        values_a: Metric values for method A across seeds.
        values_b: Metric values for method B across seeds.
        alternative: 'two-sided', 'less', or 'greater'.

    Returns:
        Dict with t_statistic, p_value, significant_at_005, mean_diff.
    """
    assert len(values_a) == len(values_b), "Must have same number of seeds"

    a = np.array(values_a)
    b = np.array(values_b)
    diff = a - b

    t_stat, p_val = stats.ttest_rel(a, b, alternative=alternative)

    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_val),
        "significant_at_005": p_val < 0.05,
        "significant_at_001": p_val < 0.01,
        "mean_diff": float(np.mean(diff)),
        "std_diff": float(np.std(diff, ddof=1)),
        "n_seeds": len(values_a),
    }


def format_result(mean: float, std: float, fmt: str = ".2f") -> str:
    """Format as 'mean ± std' for tables."""
    return f"{mean:{fmt}} ± {std:{fmt}}"
