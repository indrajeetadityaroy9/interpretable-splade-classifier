import os

import torch
import torch.nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer

from splade.data.loader import infer_max_length
from splade.circuits.geco import GECOController
from splade.circuits.losses import (
    AttributionCentroidTracker,
    UncertaintyWeighting,
    compute_completeness_loss,
    compute_separation_loss,
    compute_sharpness_loss,
)
from splade.training import constants as _C
from splade.training.constants import (
    EARLY_STOP_PATIENCE,
    EMA_DECAY,
    LABEL_SMOOTHING,
    MAX_EPOCHS,
    WARMUP_RATIO,
)
from splade.training.losses import DocumentFrequencyTracker, compute_df_flops_reg
from splade.training.optim import (_gradient_centralization,
                                   _build_param_groups,
                                   _infer_batch_size, _LRScheduler, find_lr)
from splade.utils.cuda import COMPUTE_DTYPE, DEVICE, unwrap_compiled

_NUM_WORKERS = min(os.cpu_count() or 4, 8)
_PREFETCH_FACTOR = 2


class ModelEMA:
    def __init__(self, model: torch.nn.Module, decay: float = EMA_DECAY):
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        self._backup: dict[str, torch.Tensor] = {}
        _orig = unwrap_compiled(model)
        for name, param in _orig.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        _orig = unwrap_compiled(model)
        for name, param in _orig.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].lerp_(param.data, 1.0 - self.decay)

    def apply_shadow(self, model: torch.nn.Module) -> None:
        _orig = unwrap_compiled(model)
        self._backup.clear()
        for name, param in _orig.named_parameters():
            if name in self.shadow:
                self._backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: torch.nn.Module) -> None:
        _orig = unwrap_compiled(model)
        for name, param in _orig.named_parameters():
            if name in self._backup:
                param.data.copy_(self._backup[name])
        self._backup.clear()

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.shadow.items()}


def train_model(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    texts: list[str],
    labels: list[int],
    model_name: str,
    num_labels: int,
    val_texts: list[str],
    val_labels: list[int],
    max_length: int | None = None,
    batch_size: int | None = None,
) -> "AttributionCentroidTracker":
    if max_length is None:
        max_length = infer_max_length(texts, tokenizer)
    if batch_size is None:
        batch_size = _infer_batch_size(model_name, max_length)

    print(f"Auto-inferred: max_length={max_length}, batch_size={batch_size}")

    label_tensor = torch.tensor(
        labels,
        dtype=torch.float32 if num_labels == 1 else torch.long,
    )
    encoding = tokenizer(
        texts,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    loader = DataLoader(
        TensorDataset(encoding["input_ids"], encoding["attention_mask"], label_tensor),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=_NUM_WORKERS,
        prefetch_factor=_PREFETCH_FACTOR,
        persistent_workers=True,
    )

    optimal_lr = find_lr(model, loader, num_labels)
    print(f"LR range test -> optimal_lr={optimal_lr:.2e}")

    # Uncertainty weighting: 3 learnable log-variance params for circuit losses
    uncertainty_weighting = UncertaintyWeighting(num_tasks=3)

    param_groups = _build_param_groups(model, optimal_lr)
    # Add uncertainty weighting params to optimizer (same LR, no decay)
    param_groups.append({
        "params": list(uncertainty_weighting.parameters()),
        "lr": optimal_lr,
        "weight_decay": 0.0,
    })
    optimizer = torch.optim.AdamW(param_groups, fused=True)

    steps_per_epoch = -(-len(texts) // batch_size)
    total_steps = steps_per_epoch * MAX_EPOCHS
    warmup_steps = max(1, int(WARMUP_RATIO * total_steps))

    lr_scheduler = _LRScheduler(optimal_lr, total_steps, warmup_steps)

    _orig = unwrap_compiled(model)
    vocab_size = _orig.padded_vocab_size
    df_tracker = DocumentFrequencyTracker(vocab_size=vocab_size)
    classifier_params = set(_orig.classifier_parameters())

    criterion = (
        torch.nn.BCEWithLogitsLoss()
        if num_labels == 1
        else torch.nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    )

    ema = ModelEMA(model, decay=EMA_DECAY)

    # CIS circuit losses with GECO adaptive weighting
    centroid_tracker = AttributionCentroidTracker(
        num_classes=num_labels, vocab_size=vocab_size,
    )
    geco = GECOController(steps_per_epoch=steps_per_epoch)
    circuit_delay_steps = int(_C.CIRCUIT_WARMUP_FRACTION * total_steps)
    # Patience reset at 20% of total steps regardless of whether circuits activate.
    # This ensures ablation baseline and full CIS have identical training horizons.
    _PATIENCE_RESET_FRACTION = 0.2
    patience_reset_step = int(_PATIENCE_RESET_FRACTION * total_steps)
    patience_was_reset = False
    global_step = 0
    warmup_finalized = False

    val_encoding = tokenizer(
        val_texts,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    val_label_gpu = torch.tensor(
        val_labels,
        dtype=torch.float32 if num_labels == 1 else torch.long,
    ).to(DEVICE)
    val_ids_gpu = val_encoding["input_ids"].to(DEVICE)
    val_mask_gpu = val_encoding["attention_mask"].to(DEVICE)

    best_val_loss = float("inf")
    patience_counter = 0
    best_ema_state: dict[str, torch.Tensor] | None = None

    model.train()
    for epoch_index in range(MAX_EPOCHS):
        df_tracker.soft_reset()
        total_loss = torch.zeros(1, device=DEVICE)
        batch_count = 0

        for batch_ids, batch_mask, batch_labels in tqdm(loader, desc=f"Epoch {epoch_index + 1}/{MAX_EPOCHS}"):
            batch_ids = batch_ids.to(DEVICE, non_blocking=True)
            batch_mask = batch_mask.to(DEVICE, non_blocking=True)
            batch_labels = batch_labels.to(DEVICE, non_blocking=True)

            learning_rate = lr_scheduler.step()
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = learning_rate

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", dtype=COMPUTE_DTYPE):
                logits, sparse, W_eff, _ = model(batch_ids, batch_mask)
                classification_loss = (
                    criterion(logits.squeeze(-1), batch_labels)
                    if num_labels == 1
                    else criterion(logits, batch_labels.view(-1))
                )

                df_tracker.update(sparse)
                df_loss = compute_df_flops_reg(sparse, df_tracker.get_weights())

                global_step += 1

                # Reset patience at fixed point for both baseline and CIS
                if not patience_was_reset and global_step >= patience_reset_step:
                    patience_was_reset = True
                    patience_counter = 0

                if global_step < circuit_delay_steps:
                    # Warmup: CE + DF-FLOPS only, record CE for GECO tau
                    loss = classification_loss + df_loss
                    geco.record_warmup_ce(classification_loss.detach().item())
                else:
                    if not warmup_finalized:
                        tau = geco.finalize_warmup()
                        print(f"GECO: tau_ce={tau:.4f}")
                        warmup_finalized = True

                    cc_loss = compute_completeness_loss(
                        sparse, W_eff, batch_labels.view(-1),
                        _orig.classifier_logits_only,
                    )
                    sep_loss = compute_separation_loss(centroid_tracker)
                    sharp_loss = compute_sharpness_loss(
                        sparse, W_eff, batch_labels.view(-1),
                    )

                    # Uncertainty weighting: learnable σ² per loss
                    weighted_circuit_loss = uncertainty_weighting(cc_loss, sep_loss, sharp_loss)
                    circuit_objective = weighted_circuit_loss + df_loss
                    loss = geco.compute_loss(classification_loss, circuit_objective)

                    centroid_tracker.update(
                        sparse.detach(), W_eff.detach(),
                        batch_labels.view(-1),
                    )

            loss.backward()
            _gradient_centralization(model, skip_params=classifier_params)
            optimizer.step()

            ema.update(model)

            total_loss += loss.detach()
            batch_count += 1

        average_loss = total_loss.item() / batch_count
        epoch_msg = f"Epoch {epoch_index + 1}: Loss = {average_loss:.4f}"

        stats = df_tracker.get_stats()
        epoch_msg += f", Top-1 DF: {stats['top1_df_pct']:.1f}%"

        if warmup_finalized:
            epoch_msg += f", GECO lambda={geco.lambda_ce:.4f}"
        else:
            epoch_msg += ", GECO: warming up"

        ema.apply_shadow(model)
        model.eval()
        with torch.inference_mode():
            with torch.amp.autocast("cuda", dtype=COMPUTE_DTYPE):
                val_logits, _, _, _ = model(val_ids_gpu, val_mask_gpu)
                val_loss = (
                    criterion(val_logits.squeeze(-1), val_label_gpu)
                    if num_labels == 1
                    else criterion(val_logits, val_label_gpu.view(-1))
                )
        val_loss_val = val_loss.item()
        epoch_msg += f", Val Loss (EMA): {val_loss_val:.4f}"

        ema.restore(model)
        model.train()

        if val_loss_val < best_val_loss:
            best_val_loss = val_loss_val
            patience_counter = 0
            best_ema_state = ema.state_dict()
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOP_PATIENCE:
            epoch_msg += f" [Early stop: no improvement for {EARLY_STOP_PATIENCE} epochs]"
            print(epoch_msg)
            break

        print(epoch_msg)

    _orig_model = unwrap_compiled(model)
    for name, param in _orig_model.named_parameters():
        if name in best_ema_state:
            param.data.copy_(best_ema_state[name])
    print(f"Applied best EMA weights (val loss: {best_val_loss:.4f})")

    model.eval()

    return centroid_tracker
