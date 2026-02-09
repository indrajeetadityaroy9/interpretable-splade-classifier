import copy
import itertools
import math

import numpy as np
import torch
from transformers import AutoConfig

from splade.training.constants import (LR_FIND_DIVERGE_FACTOR, LR_FIND_END,
                                       LR_FIND_STEPS, WEIGHT_DECAY)
from splade.utils.cuda import COMPUTE_DTYPE, DEVICE, unwrap_compiled


def find_lr(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    num_labels: int,
) -> float:
    _orig = unwrap_compiled(model)
    saved_state = copy.deepcopy(_orig.state_dict())

    temp_optimizer = torch.optim.AdamW(_orig.parameters(), lr=LR_FIND_END, fused=True)
    criterion = (
        torch.nn.BCEWithLogitsLoss()
        if num_labels == 1
        else torch.nn.CrossEntropyLoss()
    )

    start_lr = 1e-7
    lr_mult = (LR_FIND_END / start_lr) ** (1.0 / LR_FIND_STEPS)
    current_lr = start_lr
    best_loss = float("inf")
    lrs: list[float] = []
    losses: list[float] = []

    data_iter = itertools.cycle(train_loader)
    _orig.train()

    for _ in range(LR_FIND_STEPS):
        batch = next(data_iter)

        batch_ids, batch_mask, batch_labels = (b.to(DEVICE, non_blocking=True) for b in batch)

        for g in temp_optimizer.param_groups:
            g["lr"] = current_lr

        temp_optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=COMPUTE_DTYPE):
            logits, _, _, _ = model(batch_ids, batch_mask)
            loss = (
                criterion(logits.squeeze(-1), batch_labels)
                if num_labels == 1
                else criterion(logits, batch_labels.view(-1))
            )
        loss.backward()
        temp_optimizer.step()

        loss_val = loss.item()
        lrs.append(current_lr)
        losses.append(loss_val)
        best_loss = min(best_loss, loss_val)

        if loss_val > LR_FIND_DIVERGE_FACTOR * best_loss and len(losses) > 10:
            break

        current_lr *= lr_mult

    _orig.load_state_dict(saved_state)

    if len(losses) < 10:
        return 3e-5

    window = min(10, len(losses) // 3)
    kernel = np.ones(window) / window
    smoothed = np.convolve(losses, kernel, mode="valid")
    gradients = np.gradient(smoothed)
    best_idx = int(np.argmin(gradients))
    offset = window // 2
    found_lr = lrs[best_idx + offset]

    found_lr = max(1e-6, min(1e-3, found_lr))
    return found_lr


def _infer_batch_size(model_name: str, max_length: int) -> int:
    config = AutoConfig.from_pretrained(model_name)
    mem_ratio = (768 * 128) / (config.hidden_size * max_length)
    power = min(6, max(3, int(math.log2(max(1, 32 * mem_ratio)))))
    return 2 ** power


def _build_param_groups(model: torch.nn.Module, base_lr: float) -> list[dict]:
    _orig = unwrap_compiled(model)
    decay_params: list[torch.nn.Parameter] = []
    no_decay_params: list[torch.nn.Parameter] = []
    for name, param in _orig.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or "LayerNorm" in name or "layer_norm" in name or "bias" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    return [
        {"params": decay_params, "lr": base_lr, "weight_decay": WEIGHT_DECAY},
        {"params": no_decay_params, "lr": base_lr, "weight_decay": 0.0},
    ]


def _gradient_centralization(
    model: torch.nn.Module,
    skip_params: set | None = None,
) -> None:
    """Parameter-free gradient conditioning via centralization.

    Subtracts the mean from each gradient tensor (for weight matrices only),
    constraining gradients to the hyperplane of zero-mean vectors. This
    improves the Lipschitz smoothness of the loss landscape without any
    tunable hyperparameters.

    Reference: Yong et al., "Gradient Centralization" (arXiv:2004.01461).
    """
    skip_ids = frozenset(id(p) for p in skip_params) if skip_params else frozenset()
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        if id(parameter) in skip_ids:
            continue
        if parameter.data.ndim >= 2:
            # Centralize: subtract mean across all dims except the output dim
            parameter.grad.data -= parameter.grad.data.mean(
                dim=tuple(range(1, parameter.grad.data.ndim)),
                keepdim=True,
            )


class _LRScheduler:

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
