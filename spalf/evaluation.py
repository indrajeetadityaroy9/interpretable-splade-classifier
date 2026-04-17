"""SPALF evaluation: patching, checkpoint loading, self-terminating metrics."""

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F_fn
from omegaconf import DictConfig
from safetensors.torch import load_file
from torch import Tensor
from torchmetrics import MeanMetric
from torchmetrics.regression import R2Score

from spalf.data.store import ActivationStore
from spalf.data.buffer import ActivationBuffer
from spalf.model.sae import StratifiedSAE
from spalf.whitening import SoftZCAWhitener


@torch.no_grad()
def run_patched_forward(store: ActivationStore, sae: StratifiedSAE,
                        whitener: SoftZCAWhitener, tokens: Tensor) -> tuple[Tensor, Tensor]:
    """Original + SAE-patched forwards via scoped capture/patch hooks."""
    model = store.model
    with store.capture() as cap:
        orig_logits = model(tokens).logits
    x_raw = cap[0]
    B, S, d = x_raw.shape

    x_hat_flat, *_ = sae(whitener.forward(x_raw.reshape(-1, d).float()))
    x_hat = x_hat_flat.reshape(B, S, d).to(x_raw.dtype)

    with store.patch(x_hat):
        patched_logits = model(tokens).logits
    return orig_logits, patched_logits


def compute_kl(orig_logits: Tensor, patched_logits: Tensor) -> Tensor:
    V = orig_logits.shape[-1]
    log_p = F_fn.log_softmax(orig_logits[:, :-1].float(), dim=-1).reshape(-1, V)
    log_q = F_fn.log_softmax(patched_logits[:, :-1].float(), dim=-1).reshape(-1, V)
    return F_fn.kl_div(log_q, log_p, reduction="batchmean", log_target=True)


@torch.no_grad()
def evaluate_checkpoint(config: DictConfig) -> dict:
    ckpt = Path(config.checkpoint_path)
    with open(ckpt / "metadata.json") as f:
        metadata = json.load(f)
    cal = metadata["calibration"]
    d, F = cal["d"], cal["F"]

    sae = StratifiedSAE(d, F, cal["V"]).cuda()
    ts = torch.load(ckpt / "training_state.pt", map_location="cuda", weights_only=False)
    sae.load_state_dict(ts["sae"])
    sae.eval()

    cal_sd = load_file(str(ckpt / "calibration.safetensors"), device="cuda")
    whitener = SoftZCAWhitener(
        mean=cal_sd["mean"], eigenvalues=cal_sd["eigenvalues"],
        eigenvectors=cal_sd["eigenvectors"], reg_eigenvalues=cal_sd["reg_eigenvalues"],
        n_samples=int(cal_sd["n_samples"].item()),
        total_trace=float(cal_sd["total_trace"].item()),
    )

    store = ActivationStore(
        model_name=config.model_name, dataset_name=config.dataset,
        batch_size=config.batch_size, seq_len=config.seq_len,
        text_column=config.text_column, dataset_split=config.dataset_split,
        dataset_config=config.dataset_config, seed=config.seed + 1,
    )
    buffer = ActivationBuffer(store, buffer_size=config.seq_len * config.batch_size)

    r2_m = R2Score(num_outputs=d, multioutput="variance_weighted").cuda()
    l0_m = MeanMetric().cuda()
    kl_m = MeanMetric().cuda()
    feature_counts = torch.zeros(F, device="cuda")

    # Welford running state [n, mean, M2] for RSE convergence.
    rse = {k: [0, 0.0, 0.0] for k in ("r2", "l0", "kl")}

    def welford(k: str, x: float) -> float:
        s = rse[k]; s[0] += 1
        delta = x - s[1]; s[1] += delta / s[0]; s[2] += delta * (x - s[1])
        n, mean, m2 = s
        return float("inf") if n < 2 or mean == 0.0 else math.sqrt(m2 / (n * (n - 1))) / abs(mean)

    conv_thresh = 1.0 / d
    min_batches = math.ceil(F / config.batch_size)
    token_iter = store._token_generator(config.batch_size)

    for batch_num in range(1, d + 1):
        x = buffer.next_batch(config.batch_size)
        x_hat, _z, gate_mask, *_ = sae(whitener.forward(x))

        # R2Score.forward() returns per-batch R² AND accumulates running state.
        batch_r2 = r2_m(x_hat, x).item()
        l0 = gate_mask.sum(dim=1).float().mean()
        l0_m.update(l0)
        feature_counts += gate_mask.any(dim=0).long()

        orig_logits, patched_logits = run_patched_forward(store, sae, whitener, next(token_iter).cuda())
        kl = compute_kl(orig_logits, patched_logits)
        kl_m.update(kl)

        r2_rse = welford("r2", batch_r2)
        l0_rse = welford("l0", l0.item())
        kl_rse = welford("kl", kl.item())

        if batch_num >= min_batches and max(r2_rse, l0_rse, kl_rse) < conv_thresh:
            break

    return {
        "r2": r2_m.compute().item(),
        "l0": l0_m.compute().item(),
        "kl_divergence": kl_m.compute().item(),
        "dead_fraction": (feature_counts == 0).float().mean().item(),
    }
