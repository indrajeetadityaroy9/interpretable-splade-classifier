"""Optimization schedules and stabilization utilities."""

import copy
import math

import torch
import torch.nn
from transformers import AutoConfig

from src.models.components import _EPS


def _compute_base_lr(model_name: str) -> float:
    """Scale base LR by hidden width relative to DistilBERT-base."""
    config = AutoConfig.from_pretrained(model_name)
    return 2e-5 * (768 / config.hidden_size)


def _adaptive_gradient_clip(
    model: torch.nn.Module,
    clip_factor: float = 0.01,
    eps: float = 1e-3,
) -> None:
    """Clip per-parameter gradients by parameter norm."""
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        parameter_norm = parameter.data.norm(2)
        gradient_norm = parameter.grad.data.norm(2)
        max_norm = parameter_norm * clip_factor + eps
        if gradient_norm > max_norm:
            parameter.grad.data.mul_(max_norm / (gradient_norm + 1e-8))


class _LRScheduler:
    """Cosine LR schedule with linear warmup."""

    def __init__(self, base_lr: float, total_steps: int, warmup_steps: int):
        self.base_lr = base_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self._step = 0

    def step(self) -> float:
        current_step = self._step
        self._step += 1
        if current_step < self.warmup_steps:
            return self.base_lr * (current_step / max(self.warmup_steps, 1))
        progress = (current_step - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
        return self.base_lr * 0.5 * (1 + math.cos(math.pi * progress))


class _AdaptiveLambdaSchedule:
    """Balance regularization loss magnitude against classification loss."""

    def __init__(self, warmup_steps: int, target_ratio: float = 0.5, ema_decay: float = 0.99):
        self.target_ratio = target_ratio
        self.ema_decay = ema_decay
        self.warmup_steps = warmup_steps
        self._step = 0
        self._lambda = 1.0
        self._ema_cls: float | None = None
        self._ema_reg: float | None = None
        self._current_sparsity = 0.0

    def compute_lambda(
        self,
        activations: torch.Tensor,
        cls_loss_val: float = 0.0,
        reg_loss_val: float = 0.0,
    ) -> float:
        with torch.no_grad():
            self._current_sparsity = (
                (activations.abs() < _EPS[activations.dtype]["div"]).float().mean().item()
            )

        if self._step < self.warmup_steps:
            self._step += 1
            return self._lambda * (self._step / self.warmup_steps) ** 2

        if self._ema_cls is None:
            self._ema_cls = cls_loss_val
            self._ema_reg = reg_loss_val
        else:
            self._ema_cls = self.ema_decay * self._ema_cls + (1 - self.ema_decay) * cls_loss_val
            self._ema_reg = self.ema_decay * self._ema_reg + (1 - self.ema_decay) * reg_loss_val

        if self._ema_reg is not None and self._ema_reg > 1e-12:
            self._lambda = (self.target_ratio * self._ema_cls) / self._ema_reg

        self._step += 1
        return max(0.01, min(self._lambda, 1e4))


class _EarlyStopping:
    """Track best training loss and stop after patience is exceeded."""

    def __init__(self, patience: int = 3, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.best_state: dict | None = None

    def step(self, val_loss: float, model: torch.nn.Module) -> bool:
        """Return True when training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = copy.deepcopy(model.state_dict())
            return False
        self.counter += 1
        return self.counter >= self.patience


def _compute_warmup_steps(total_steps: int) -> int:
    """Compute warmup steps from total training steps."""
    return max(1, min(int(math.sqrt(total_steps) * 2), total_steps // 3))
