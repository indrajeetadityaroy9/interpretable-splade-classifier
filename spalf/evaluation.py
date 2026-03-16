"""SPALF evaluation: patching utilities, checkpoint loading, self-terminating metrics."""

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F_fn
from omegaconf import DictConfig
from safetensors.torch import load_file
from torch import Tensor
from torchmetrics import MeanMetric

from spalf.data.store import ActivationStore
from spalf.data.buffer import ActivationBuffer
from spalf.model.sae import StratifiedSAE
from spalf.whitening import SoftZCAWhitener


@torch.no_grad()
def run_patched_forward(
    store: ActivationStore, sae: StratifiedSAE,
    whitener: SoftZCAWhitener, tokens: Tensor,
) -> tuple[Tensor, Tensor]:
    """Run original and SAE-patched forwards, return (orig_logits, patched_logits)."""
    model = store.model

    orig_logits = model(tokens).logits
    x_raw = store._captured_activations
    B, S, d = x_raw.shape

    x_flat = x_raw.reshape(-1, d).float()
    x_tilde = whitener.forward(x_flat)
    x_hat_flat, _, _, _, _ = sae(x_tilde)
    x_hat = x_hat_flat.reshape(B, S, d).to(x_raw.dtype)

    def inject_hook(_module, _input, output):
        return (x_hat,) + output[1:]

    handle = store.swap_hook(inject_hook)
    patched_logits = model(tokens).logits
    handle.remove()
    store.restore_hook()

    return orig_logits, patched_logits


def compute_kl(orig_logits: Tensor, patched_logits: Tensor) -> Tensor:
    """KL divergence between original and patched next-token distributions."""
    V_vocab = orig_logits.shape[-1]
    log_p = F_fn.log_softmax(orig_logits[:, :-1].float(), dim=-1).reshape(-1, V_vocab)
    log_q = F_fn.log_softmax(patched_logits[:, :-1].float(), dim=-1).reshape(-1, V_vocab)
    return F_fn.kl_div(log_q, log_p, reduction="batchmean", log_target=True)


def _load_checkpoint(
    checkpoint_path: str,
) -> tuple[StratifiedSAE, SoftZCAWhitener, torch.Tensor, dict]:
    """Load SAE, whitener, W_vocab, and metadata from a checkpoint directory."""
    ckpt = Path(checkpoint_path)

    with open(ckpt / "metadata.json") as f:
        metadata = json.load(f)

    cal = metadata["calibration"]
    sae = StratifiedSAE(cal["d"], cal["F"], cal["V"]).cuda()

    training_state = torch.load(
        ckpt / "training_state.pt", map_location="cuda", weights_only=False,
    )
    sae.load_state_dict(training_state["sae"])
    sae.eval()

    cal_sd = load_file(str(ckpt / "calibration.safetensors"), device="cuda")
    whitener = SoftZCAWhitener(
        mean=cal_sd["mean"],
        eigenvalues=cal_sd["eigenvalues"],
        eigenvectors=cal_sd["eigenvectors"],
        reg_eigenvalues=cal_sd["reg_eigenvalues"],
        n_samples=int(cal_sd["n_samples"].item()),
        total_trace=float(cal_sd["total_trace"].item()),
    )
    W_vocab = cal_sd["W_vocab"]

    return sae, whitener, W_vocab, metadata


def _rse(n: int, mean: float, m2: float) -> float:
    """Relative standard error from Welford running state."""
    if n < 2 or mean == 0.0:
        return float("inf")
    return math.sqrt(m2 / (n * (n - 1))) / abs(mean)


@torch.no_grad()
def evaluate_checkpoint(config: DictConfig) -> dict:
    """Evaluate a trained SAE checkpoint with self-terminating convergence."""
    sae, whitener, W_vocab, metadata = _load_checkpoint(config.checkpoint_path)
    cal = metadata["calibration"]
    d = cal["d"]
    convergence_threshold = 1.0 / d

    store = ActivationStore(
        model_name=config.model_name,
        dataset_name=config.dataset,
        batch_size=config.batch_size,
        seq_len=config.seq_len,
        text_column=config.text_column,
        dataset_split=config.dataset_split,
        dataset_config=config.dataset_config,
        seed=config.seed + 1,
    )

    buffer = ActivationBuffer(store, buffer_size=config.seq_len * config.batch_size)

    # torchmetrics accumulators for reported metrics.
    mse_metric = MeanMetric().cuda()
    var_metric = MeanMetric().cuda()
    l0_metric = MeanMetric().cuda()
    kl_metric = MeanMetric().cuda()
    feature_counts = torch.zeros(cal["F"], device="cuda")

    # Welford running state for RSE convergence detection.
    rse_state = {k: {"n": 0, "mean": 0.0, "m2": 0.0} for k in ("r2", "l0", "kl")}

    def _welford_update(key: str, x: float) -> None:
        s = rse_state[key]
        s["n"] += 1
        delta = x - s["mean"]
        s["mean"] += delta / s["n"]
        s["m2"] += delta * (x - s["mean"])

    min_batches = math.ceil(cal["F"] / config.batch_size)
    batch_num = 0
    token_iter = store._token_generator(config.batch_size)

    while True:
        batch_num += 1

        x = buffer.next_batch(config.batch_size)
        x_tilde = whitener.forward(x)
        x_hat, z, gate_mask, _, _ = sae(x_tilde)

        batch_mse = (x - x_hat).pow(2).sum(dim=1).mean()
        batch_var = (x - x.mean(dim=0)).pow(2).sum(dim=1).mean()
        mse_metric.update(batch_mse)
        var_metric.update(batch_var)

        batch_l0 = gate_mask.sum(dim=1).float().mean()
        l0_metric.update(batch_l0)

        feature_counts += gate_mask.any(dim=0).long()

        tokens = next(token_iter).cuda()
        orig_logits, patched_logits = run_patched_forward(
            store, sae, whitener, tokens,
        )
        batch_kl = compute_kl(orig_logits, patched_logits)
        kl_metric.update(batch_kl)

        # Update RSE convergence state.
        batch_r2 = 1.0 - batch_mse.item() / batch_var.item()
        _welford_update("r2", batch_r2)
        _welford_update("l0", batch_l0.item())
        _welford_update("kl", batch_kl.item())

        if batch_num >= min_batches:
            converged = all(
                _rse(rse_state[k]["n"], rse_state[k]["mean"], rse_state[k]["m2"])
                < convergence_threshold
                for k in ("r2", "l0", "kl")
            )
            if converged:
                break

        if batch_num >= d:
            break

    global_mse = mse_metric.compute().item()
    global_var = var_metric.compute().item()

    return {
        "r2": 1.0 - global_mse / global_var,
        "l0": l0_metric.compute().item(),
        "kl_divergence": kl_metric.compute().item(),
        "dead_fraction": (feature_counts == 0).float().mean().item(),
    }
