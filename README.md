# K-Bézier: Unifying Information Geometry and Mode Connectivity for Continual Learning

> **K-FAC Riemannian SAM + Bézier Path Connectivity** — a unified framework
> for long-sequence continual learning that addresses the stability–plasticity
> dilemma through information-geometric optimization.

## Key Components

| Module | Theory Section | Purpose |
|--------|---------------|---------|
| `metrics/` | §A — K-FAC Metric | Fisher Information as Riemannian metric tensor |
| `optimizers/` | §B — R-SAM | Perturbation on Riemannian ellipsoid (local flatness) |
| `connectivity/` | §C — Bézier Path | Curved mode connectivity (global structure) |
| `strategies/` | Full Method | Unified K-Bézier continual learning strategy |

## Installation

```bash
git clone <repo-url> && cd kbezier-cl
pip install -e ".[dev]"
```

## Quick Start

```bash
# Smoke test: Finetune on Permuted MNIST
python scripts/run.py --config method/finetune benchmark/perm_mnist

# Full method: K-Bézier on Split-CIFAR100
python scripts/run.py --config method/kbezier benchmark/split_cifar100

# Ablation: γ sweep
python scripts/run_ablation.py --config ablation/gamma_sweep benchmark/split_cifar100
```

## Running Experiments

All experiments are YAML-driven. Configs are merged in order:

```bash
python scripts/run.py --config <base> <method> <benchmark> [--seed 42] [--device cuda]
```

Results are saved to `results/{benchmark}_{method}_{seed}/` with:
- Resolved config (YAML) + git commit hash
- Accuracy matrix R[i,j]
- CL metrics (ACC, BWT, FWT, forgetting)
- Compute tracker (wall-clock, forward/backward counts)
- TensorBoard logs

## Project Structure

```
kbezier-cl/
├── configs/            # YAML experiment definitions
│   ├── base.yaml
│   ├── benchmark/      # perm_mnist, split_cifar100, split_tinyimagenet, seq_cifar100_50t
│   ├── method/         # ewc, si, sam, fsam, rsam, bezier, kbezier, finetune
│   └── ablation/       # gamma_sweep, rho_sweep, fisher_update_freq
├── src/kbezier/
│   ├── metrics/        # Fisher metric: K-FAC, diagonal, mixture, layer hooks
│   ├── optimizers/     # SAM, R-SAM, F-SAM
│   ├── connectivity/   # Bézier curves, path loss, linear LMC
│   ├── strategies/     # CL methods: finetune, EWC, SI, K-Bézier
│   ├── benchmarks/     # Data loading, scenarios, replay buffer
│   ├── models/         # MLP, ResNet-18 with IncrementalHead
│   ├── evaluation/     # CL metrics, landscape, statistics, compute tracking
│   ├── engine/         # Registry, config, trainer
│   └── utils/          # Seeding, logging, checkpointing
├── scripts/            # Entry points: run, ablation, figures
├── tests/              # Mathematical correctness tests
└── results/            # Auto-generated experiment outputs
```

## Tests

```bash
pytest tests/ -v
```

Tests verify mathematical correctness:
- K-FAC `inv_mv` matches direct F⁻¹g
- R-SAM constraint ε*ᵀFε* = ρ²
- Proposition 2 bound on forgetting
- Bézier endpoints + gradient flow through `functional_call`
- CL metric calculations on known matrices

## Citation

```bibtex
@article{kbezier2025,
  title={Unifying Information Geometry and Mode Connectivity:
         A Kronecker-Factored Riemannian Approach to Long-Sequence Continual Learning},
  year={2025}
}
```
