"""
Generate paper-quality figures from results.

Reads results/ directory and produces:
1. Accuracy vs Tasks curve (per method)
2. Loss barrier comparison (linear vs Bézier)
3. Ablation heatmaps (γ sweep, ρ sweep)
4. Compute comparison table

Usage:
    python scripts/make_figures.py --results-dir results/ --output-dir figures/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_results(results_dir: Path) -> dict:
    """Load all experiment results from a directory."""
    all_results = {}
    for run_dir in sorted(results_dir.iterdir()):
        if run_dir.is_dir():
            results_file = run_dir / "results.json"
            if results_file.exists():
                with open(results_file) as f:
                    data = json.load(f)
                all_results[run_dir.name] = data
    return all_results


def plot_accuracy_curves(all_results: dict, output_dir: Path) -> None:
    """Plot accuracy vs number of tasks for each method."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for run_name, data in all_results.items():
        acc_matrix = np.array(data["accuracy_matrix"])
        T = acc_matrix.shape[0]

        # Average accuracy after each task
        accs = [acc_matrix[t, :t+1].mean() for t in range(T)]
        method = run_name.split("_")[1] if "_" in run_name else run_name
        ax.plot(range(T), accs, marker="o", markersize=3, label=method)

    ax.set_xlabel("Task", fontsize=12)
    ax.set_ylabel("Average Accuracy", fontsize=12)
    ax.set_title("Continual Learning: Accuracy vs Tasks", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "accuracy_curves.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(output_dir / "accuracy_curves.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved accuracy curves to {output_dir}")


def plot_forgetting(all_results: dict, output_dir: Path) -> None:
    """Plot per-task forgetting comparison."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for run_name, data in all_results.items():
        forgetting = data.get("metrics", {}).get("forgetting_per_task", [])
        method = run_name.split("_")[1] if "_" in run_name else run_name
        ax.bar(
            np.arange(len(forgetting)) + list(all_results.keys()).index(run_name) * 0.15,
            forgetting,
            width=0.15,
            label=method,
            alpha=0.8,
        )

    ax.set_xlabel("Task", fontsize=12)
    ax.set_ylabel("Forgetting", fontsize=12)
    ax.set_title("Per-Task Forgetting", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "forgetting.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)


def make_compute_table(all_results: dict, output_dir: Path) -> None:
    """Generate compute comparison table."""
    rows = []
    for run_name, data in all_results.items():
        compute = data.get("compute", {})
        method = run_name.split("_")[1] if "_" in run_name else run_name
        rows.append({
            "Method": method,
            "Wall Clock (s)": compute.get("total_time_seconds", 0),
            "Forward Passes": compute.get("forward_passes", 0),
            "Backward Passes": compute.get("backward_passes", 0),
            "Fisher Time (s)": compute.get("fisher_accumulation_time", 0),
            "ACC": data.get("metrics", {}).get("acc", 0),
            "BWT": data.get("metrics", {}).get("bwt", 0),
        })

    df = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "compute_comparison.csv", index=False)
    print(f"\nCompute Comparison:")
    print(df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--output-dir", type=str, default="figures")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        sys.exit(1)

    all_results = load_results(results_dir)
    if not all_results:
        print("No results found.")
        sys.exit(1)

    print(f"Found {len(all_results)} experiment runs")

    plot_accuracy_curves(all_results, output_dir)
    plot_forgetting(all_results, output_dir)
    make_compute_table(all_results, output_dir)

    print(f"\nAll figures saved to {output_dir}")


if __name__ == "__main__":
    main()
