"""
K-Bézier Strategy: the full unified method.

Combines:
- R-SAM optimizer (local flatness via Riemannian ellipsoid perturbation)
- Bézier path connectivity (global structure via curved mode connectivity)
- Mixture Fisher (stability–plasticity knob γ)

Training step:
    L_total = L_task(w + ε*) + α · L_GMC(w_v)

where ε* is the R-SAM perturbation and L_GMC is the combined path loss.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn

from kbezier.benchmarks.replay_buffer import ReplayBuffer
from kbezier.connectivity.bezier import BezierCurve
from kbezier.connectivity.path_loss import PathLoss
from kbezier.engine.config import get_nested
from kbezier.engine.registry import Registry
from kbezier.metrics.mixture import MixtureFisher
from kbezier.optimizers.rsam import RSAMOptimizer
from kbezier.strategies.base_strategy import BaseStrategy


@Registry.register_strategy("kbezier")
class KBezierStrategy(BaseStrategy):
    """
    Full K-Bézier continual learning strategy.

    before_task(t):
        - Expand head for new classes
        - If t > 0: snapshot w_{t-1}, create BezierCurve
        - Set up R-SAM optimizer with mixture Fisher

    observe(x, y, t):
        - R-SAM step on task-t data (local flatness)
        - If t > 0: path_loss on replay data (global connectivity)

    after_task(t):
        - Estimate Fisher for task t
        - Update F_old ← β·F_old + F_t
        - Save anchors to disk, release GPU memory
        - Update replay buffer
    """

    def __init__(self, model: nn.Module, config: dict, device: str = "cuda"):
        super().__init__(model, config, device)

        method_cfg = config.get("method", {})

        # R-SAM config
        self.rho = method_cfg.get("rho", 0.05)
        self.gamma = method_cfg.get("gamma", 0.75)

        # Fisher config
        fisher_cfg = method_cfg.get("fisher", config.get("fisher", {}))
        self.damping = fisher_cfg.get("damping", 1e-3)
        self.fisher_samples = fisher_cfg.get("samples", 1000)
        self.fisher_update_freq = fisher_cfg.get("update_freq", 1)

        # Connectivity config
        conn_cfg = method_cfg.get("connectivity", {})
        self.kappa = conn_cfg.get("kappa", 0.1)
        self.alpha = conn_cfg.get("alpha", 1.0)
        self.n_theta_samples = conn_cfg.get("n_theta_samples", 2)
        self.cp_lr = conn_cfg.get("control_point_lr", 0.01)
        self.cp_init = conn_cfg.get("control_point_init", "midpoint")

        # Replay config
        replay_cfg = method_cfg.get("replay", config.get("replay", {}))
        buffer_size = replay_cfg.get("buffer_size", 200)

        # Initialize components
        self.fisher_metric = MixtureFisher(
            gamma=self.gamma,
            damping=self.damping,
        )
        self.replay_buffer = ReplayBuffer(buffer_size_per_task=buffer_size)

        # State
        self.bezier_curve: Optional[BezierCurve] = None
        self.path_loss_fn: Optional[PathLoss] = None
        self.cp_optimizer: Optional[torch.optim.Adam] = None
        self.rsam_optimizer: Optional[RSAMOptimizer] = None
        self._prev_params: Optional[Dict[str, torch.Tensor]] = None
        self._anchors_dir = Path(get_nested(config, "log_dir", "results")) / "anchors"

    def before_task(self, task_id: int, train_loader) -> None:
        self.current_task = task_id

        # Create base optimizer (SGD)
        base_optimizer = self._make_optimizer()

        # Create R-SAM optimizer wrapping the base optimizer
        self.rsam_optimizer = RSAMOptimizer(
            model=self.model,
            base_optimizer=base_optimizer,
            fisher_metric=self.fisher_metric,
            rho=self.rho,
        )
        self.optimizer = base_optimizer  # for compatibility

        # Set up Bézier connectivity for task > 0
        if task_id > 0 and self._prev_params is not None:
            # Snapshot current model as w₀ (start of curve)
            w0 = self._prev_params  # saved from after_task

            # w₁ will be updated after training (placeholder = current model)
            w1 = {
                name: param.detach().clone()
                for name, param in self.model.named_parameters()
            }

            self.bezier_curve = BezierCurve(w0, w1, init=self.cp_init).to(self.device)

            # Optimizer for control point w_v
            self.cp_optimizer = torch.optim.Adam(
                self.bezier_curve.w_v.parameters(),
                lr=self.cp_lr,
            )

            # Path loss function
            self.path_loss_fn = PathLoss(
                model=self.model,
                bezier_curve=self.bezier_curve,
                fisher_metric=self.fisher_metric,
                kappa=self.kappa,
                n_theta_samples=self.n_theta_samples,
            )
        else:
            self.bezier_curve = None
            self.path_loss_fn = None
            self.cp_optimizer = None

        # Estimate Fisher if we have accumulated old tasks
        if task_id > 0:
            self.fisher_metric.accumulate(
                self.model, train_loader, n_samples=self.fisher_samples
            )

    def observe(self, x: torch.Tensor, y: torch.Tensor, task_id: int) -> float:
        self.model.train()

        # ── R-SAM step on current task (local flatness) ──
        def loss_fn():
            return self.criterion(self.model(x), y)

        task_loss = self.rsam_optimizer.step(loss_fn)

        # ── Bézier path-loss on replay data (global connectivity) ──
        gmc_loss = 0.0
        if (
            self.path_loss_fn is not None
            and self.cp_optimizer is not None
            and self.replay_buffer.n_tasks > 0
        ):
            # Sample from replay buffer (old tasks)
            x_replay, y_replay = self.replay_buffer.sample(
                batch_size=x.size(0),
                up_to_task=task_id - 1,
                device=self.device,
            )

            # Update Bézier endpoint w₁ to current model weights
            current_params = {
                name: param.detach().clone()
                for name, param in self.model.named_parameters()
            }
            self.bezier_curve.update_endpoint(current_params)

            # Compute path loss and update control point
            self.cp_optimizer.zero_grad()
            path_loss, info = self.path_loss_fn(x_replay, y_replay)
            scaled_path_loss = self.alpha * path_loss
            scaled_path_loss.backward()
            self.cp_optimizer.step()

            gmc_loss = info["total_gmc_loss"]

        # Update replay buffer with current batch
        self.replay_buffer.update(task_id, x, y)

        return task_loss + gmc_loss

    def after_task(self, task_id: int, train_loader=None) -> None:
        # ── Estimate Fisher for completed task ──
        if train_loader is not None:
            self.fisher_metric.accumulate(
                self.model, train_loader, n_samples=self.fisher_samples
            )
            # Update accumulated old Fisher: F_old ← β·F_old + F_t
            self.fisher_metric.update_old_fisher(task_id)

        # ── Save parameter snapshot for next task's Bézier start ──
        self._prev_params = {
            name: param.detach().clone().cpu()
            for name, param in self.model.named_parameters()
        }

        # ── Save Bézier anchors to disk, release GPU memory ──
        if self.bezier_curve is not None:
            anchor_path = self._anchors_dir / f"bezier_task_{task_id}.pt"
            self.bezier_curve.save_anchors(anchor_path)
            # Release GPU memory
            del self.bezier_curve
            self.bezier_curve = None
            del self.path_loss_fn
            self.path_loss_fn = None
            del self.cp_optimizer
            self.cp_optimizer = None
            torch.cuda.empty_cache()


# ── SAM-only strategy (for ablation) ──

@Registry.register_strategy("sam")
class SAMStrategy(BaseStrategy):
    """SAM Euclidean baseline strategy."""

    def __init__(self, model, config, device="cuda"):
        super().__init__(model, config, device)
        from kbezier.optimizers.base_sam import SAMOptimizer
        self.rho = config.get("method", {}).get("rho", 0.05)
        self.sam_optimizer = None

    def before_task(self, task_id, train_loader):
        self.current_task = task_id
        base_opt = self._make_optimizer()
        from kbezier.optimizers.base_sam import SAMOptimizer
        self.sam_optimizer = SAMOptimizer(self.model, base_opt, self.rho)
        self.optimizer = base_opt

    def observe(self, x, y, task_id):
        self.model.train()
        def loss_fn():
            return self.criterion(self.model(x), y)
        return self.sam_optimizer.step(loss_fn)

    def after_task(self, task_id, train_loader=None):
        pass


@Registry.register_strategy("rsam")
class RSAMStrategy(BaseStrategy):
    """R-SAM only strategy (no Bézier connectivity). For ablation."""

    def __init__(self, model, config, device="cuda"):
        super().__init__(model, config, device)
        method_cfg = config.get("method", {})
        self.rho = method_cfg.get("rho", 0.05)
        self.gamma = method_cfg.get("gamma", 0.75)
        fisher_cfg = method_cfg.get("fisher", config.get("fisher", {}))
        self.damping = fisher_cfg.get("damping", 1e-3)
        self.fisher_samples = fisher_cfg.get("samples", 1000)

        self.fisher_metric = MixtureFisher(gamma=self.gamma, damping=self.damping)
        self.rsam_optimizer = None

    def before_task(self, task_id, train_loader):
        self.current_task = task_id
        base_opt = self._make_optimizer()
        self.rsam_optimizer = RSAMOptimizer(
            self.model, base_opt, self.fisher_metric, self.rho
        )
        self.optimizer = base_opt

        if task_id > 0:
            self.fisher_metric.accumulate(
                self.model, train_loader, n_samples=self.fisher_samples
            )

    def observe(self, x, y, task_id):
        self.model.train()
        def loss_fn():
            return self.criterion(self.model(x), y)
        return self.rsam_optimizer.step(loss_fn)

    def after_task(self, task_id, train_loader=None):
        if train_loader is not None:
            self.fisher_metric.accumulate(
                self.model, train_loader, n_samples=self.fisher_samples
            )
            self.fisher_metric.update_old_fisher(task_id)


@Registry.register_strategy("fsam")
class FSAMStrategy(BaseStrategy):
    """F-SAM (Kwon et al.) baseline strategy."""

    def __init__(self, model, config, device="cuda"):
        super().__init__(model, config, device)
        self.rho = config.get("method", {}).get("rho", 0.05)
        self.damping = config.get("method", {}).get("fisher", {}).get("damping", 1e-3)
        self.fsam_optimizer = None

    def before_task(self, task_id, train_loader):
        self.current_task = task_id
        base_opt = self._make_optimizer()
        from kbezier.optimizers.fsam import FSAMOptimizer
        self.fsam_optimizer = FSAMOptimizer(self.model, base_opt, self.rho, self.damping)
        self.optimizer = base_opt

        # Estimate Fisher from current task data
        self.fsam_optimizer.update_fisher(train_loader, n_samples=1000)

    def observe(self, x, y, task_id):
        self.model.train()
        def loss_fn():
            return self.criterion(self.model(x), y)
        return self.fsam_optimizer.step(loss_fn)

    def after_task(self, task_id, train_loader=None):
        pass


# ── Bézier-only strategy (for ablation: connectivity without R-SAM) ──

@Registry.register_strategy("bezier")
class BezierStrategy(KBezierStrategy):
    """Bézier path connectivity with a PLAIN optimizer (no R-SAM).

    Isolates the contribution of curved mode connectivity for the ablation
    grid {SAM, R-SAM} × {linear, Bézier}. Everything (mixture Fisher, path
    loss, control-point optimization, anchor offload) is identical to
    KBezierStrategy except the per-batch task step uses the base optimizer
    directly instead of the Riemannian-SAM perturbation.
    """

    def observe(self, x: torch.Tensor, y: torch.Tensor, task_id: int) -> float:
        self.model.train()

        # ── Plain SGD step on current task (NO R-SAM perturbation) ──
        self.optimizer.zero_grad()
        loss = self.criterion(self.model(x), y)
        loss.backward()
        self.optimizer.step()
        task_loss = float(loss.item())

        # ── Bézier path-loss on replay data (global connectivity) ──
        gmc_loss = 0.0
        if (
            self.path_loss_fn is not None
            and self.cp_optimizer is not None
            and self.replay_buffer.n_tasks > 0
        ):
            x_replay, y_replay = self.replay_buffer.sample(
                batch_size=x.size(0),
                up_to_task=task_id - 1,
                device=self.device,
            )
            current_params = {
                name: param.detach().clone()
                for name, param in self.model.named_parameters()
            }
            self.bezier_curve.update_endpoint(current_params)

            self.cp_optimizer.zero_grad()
            path_loss, info = self.path_loss_fn(x_replay, y_replay)
            scaled_path_loss = self.alpha * path_loss
            scaled_path_loss.backward()
            self.cp_optimizer.step()

            gmc_loss = info["total_gmc_loss"]

        self.replay_buffer.update(task_id, x, y)
        return task_loss + gmc_loss
