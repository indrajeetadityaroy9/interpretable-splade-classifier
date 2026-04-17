import json
import math
from pathlib import Path

import torch
from torch import Tensor
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_file, save_file
from transformers import get_wsd_schedule, set_seed

from spalf.data.store import ActivationStore
from spalf.data.buffer import ActivationBuffer
from spalf.evaluation import run_patched_forward, compute_kl
from spalf.model.sae import StratifiedSAE
from spalf.model.constraints import compute_orthogonality_violation
from spalf.model.initialization import initialize_from_calibration
from spalf.optim.controller import DualController
from spalf.optim.lagrangian import compute_augmented_lagrangian
from spalf.whitening import FrequentDirections, SoftZCAWhitener


def _calibrate(config: DictConfig, store: ActivationStore) -> dict:
    """Build FD sketch, whitener, W_vocab slice, and derive all hyperparameters."""
    d = store.model.config.hidden_size

    sketch = FrequentDirections(d)
    while not sketch._converged:
        sketch.update(store.next_batch())
    whitener = SoftZCAWhitener.from_sketch(sketch)

    W_full = store.get_unembedding_matrix()
    V_cap = config.V_cap
    if 0 < V_cap < W_full.shape[1]:
        idx = W_full.norm(dim=0).topk(V_cap).indices.sort().values
        W_vocab = W_full[:, idx]
    else:
        W_vocab = W_full
    V = W_vocab.shape[1]

    eff_rank = whitener.effective_rank
    F = V + eff_rank
    L0 = max(1, eff_rank)
    total_steps = config.total_tokens // config.batch_size
    n_params = 2 * F * d + 2 * F + d

    cal = {
        "d": d, "V": V, "F": F, "L0_target": L0, "eff_rank": eff_rank,
        "tau_faith": max(whitener.noise_fraction, 1.0 / d) * d,
        "tau_drift": W_vocab.pow(2).sum().item() / d,
        "tau_ortho": 0.0, "tau_kl": None,
        "total_steps": total_steps,
        "lr": 1.0 / n_params ** 0.5,                          # McCandlish et al.
        "warmup_steps": round(total_steps ** 0.5),
        "n_cal_batches": math.ceil(F / config.batch_size),
        "buffer_size": config.seq_len * config.batch_size,
        "l0_scale": L0 / F,
        "n_primal": 3, "n_onset": 1,
        "whitener": whitener, "W_vocab": W_vocab,
    }
    print(json.dumps({"event": "derived_hyperparameters",
                      **{k: v for k, v in cal.items()
                         if k not in ("whitener", "W_vocab", "tau_kl")}},
                     sort_keys=True), flush=True)
    return cal


def _save_checkpoint(output_dir, sae, optimizer, scheduler, controller,
                     whitener, W_vocab, cal, config, step, onset_step, D_ema, D_0) -> None:
    ckpt_dir = Path(output_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    torch.save({"sae": sae.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "controller": controller.state_dict()},
               ckpt_dir / "training_state.pt")
    save_file({**whitener.state_dict(), "W_vocab": W_vocab},
              str(ckpt_dir / "calibration.safetensors"))
    with open(ckpt_dir / "metadata.json", "w") as f:
        json.dump({
            "step": step, "onset_step": onset_step,
            "config": OmegaConf.to_container(config, resolve=True),
            "calibration": {k: cal[k] for k in
                            ("d", "V", "F", "L0_target",
                             "tau_faith", "tau_drift", "tau_ortho", "tau_kl")},
            "D_ema": D_ema.item(), "D_0": D_0.item(),
        }, f, indent=2)


def train(config: DictConfig) -> StratifiedSAE:
    """Full SPALF pipeline: seed → calibrate → initialize → MTZF constrained AL loop."""
    set_seed(config.seed)

    store = ActivationStore(
        model_name=config.model_name, dataset_name=config.dataset,
        batch_size=config.batch_size, seq_len=config.seq_len,
        text_column=config.text_column, dataset_split=config.dataset_split,
        dataset_config=config.dataset_config, seed=config.seed,
    )

    cal = _calibrate(config, store)
    whitener, W_vocab = cal["whitener"], cal["W_vocab"]
    buffer = ActivationBuffer(store, buffer_size=cal["buffer_size"])

    sae = initialize_from_calibration(cal, store)
    sae = torch.compile(sae, mode="max-autotune")

    # Initial violations + transition-zone mass D_0 via n_cal_batches warmup.
    accum = torch.zeros(cal["n_primal"], device="cuda")
    D_accum = 0.0
    with torch.no_grad():
        for _ in range(cal["n_cal_batches"]):
            x = buffer.next_batch(config.batch_size)
            x_tilde = whitener.forward(x)
            x_hat, z, _, _, disc = sae(x_tilde)
            D_accum += disc.mean().item()
            v_faith = whitener.compute_mahalanobis_sq(x - x_hat).mean() - cal["tau_faith"]
            v_drift = (sae.W_dec_A - W_vocab).pow(2).sum() - cal["tau_drift"]
            v_ortho = compute_orthogonality_violation(z, sae.W_dec_A, sae.W_dec_B, cal["tau_ortho"])
            accum += torch.stack([v_faith, v_drift, v_ortho]).abs()
    initial_violations = torch.cat([accum / cal["n_cal_batches"],
                                    torch.ones(cal["n_onset"], device="cuda")])
    D_0 = torch.tensor(D_accum / cal["n_cal_batches"], device="cuda")
    D_ema = D_0.clone()

    total_steps = cal["total_steps"]
    rho_0 = cal["l0_scale"] / initial_violations.abs().mean().item()
    controller = DualController(initial_violations, rho_0, n_primal=cal["n_primal"])

    optimizer = torch.optim.Adam(sae.parameters(), lr=cal["lr"], fused=True)

    # AdaGC: per-tensor grad norm clip against EMA of historical norms.
    adagc_beta = optimizer.defaults["betas"][1]
    clip_params = [p for p in sae.parameters() if p.data.ndim > 1]
    ema_norms = [p.data.norm().clone() for p in clip_params]

    # WSD: warmup √T, stable, decay √T.
    warmup = decay = cal["warmup_steps"]
    scheduler = get_wsd_schedule(optimizer, warmup, total_steps - 2 * warmup, decay)

    start_step, onset_step = 0, total_steps
    if config.resume_from_checkpoint:
        ckpt_path = Path(config.resume_from_checkpoint)
        ts = torch.load(ckpt_path / "training_state.pt", map_location="cuda", weights_only=False)
        sae.load_state_dict(ts["sae"])
        optimizer.load_state_dict(ts["optimizer"])
        scheduler.load_state_dict(ts["scheduler"])
        controller.load_state_dict(ts["controller"])
        cal_sd = load_file(str(ckpt_path / "calibration.safetensors"), device="cuda")
        whitener.load_state_dict({k: cal_sd[k] for k in
                                  ("mean", "eigenvalues", "eigenvectors", "reg_eigenvalues",
                                   "n_samples", "total_trace")})
        W_vocab = cal_sd["W_vocab"]
        with open(ckpt_path / "metadata.json") as f:
            meta = json.load(f)
        start_step, onset_step = meta["step"], meta["onset_step"]
        for k in ("tau_faith", "tau_drift", "tau_ortho", "tau_kl"):
            cal[k] = meta["calibration"][k]
        D_0 = torch.tensor(meta["D_0"], device="cuda")
        D_ema = torch.tensor(meta["D_ema"], device="cuda")

    whitener.forward = torch.compile(whitener.forward)
    whitener.compute_mahalanobis_sq = torch.compile(whitener.compute_mahalanobis_sq)

    token_iter = store._token_generator(config.batch_size)
    tau_kl = cal["tau_kl"]
    kl_active = tau_kl is not None
    alpha_floor = (cal["d"] / cal["F"]) ** 2
    kl_sentinel = -torch.tensor([cal["tau_faith"], cal["tau_drift"], cal["tau_ortho"]],
                                device="cuda").sum()

    # Dead threshold: P(0 firings in N) < 1/F  ⇒  N > F·ln(F)/(B·L0).
    dead_threshold = max(
        math.ceil(cal["F"] * math.log(cal["F"]) / (config.batch_size * cal["L0_target"])),
        cal["n_cal_batches"],
    )

    for step in range(start_step, total_steps):
        # ── Forward + violations ─────────────────────────────────
        kl_div = None
        if kl_active:
            with torch.no_grad():
                orig_logits, patched_logits = run_patched_forward(
                    store, sae, whitener, next(token_iter).cuda(),
                )
                kl_div = compute_kl(orig_logits, patched_logits)

        x = buffer.next_batch(config.batch_size)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            x_tilde = whitener.forward(x)
            x_hat, z, gate_mask, l0_probs, disc_penalty = sae(x_tilde)

        v_faith = whitener.compute_mahalanobis_sq(x - x_hat).mean() - cal["tau_faith"]
        v_drift = (sae.W_dec_A - W_vocab).pow(2).sum() - cal["tau_drift"]
        v_ortho = compute_orthogonality_violation(z, sae.W_dec_A, sae.W_dec_B, cal["tau_ortho"])
        v_kl = (kl_div - tau_kl) if kl_active else kl_sentinel
        violations = torch.stack([v_faith, v_drift, v_ortho, v_kl])

        controller.update(violations)
        beta = controller._adaptive_beta()
        D_ema.mul_(beta).add_(disc_penalty.mean(), alpha=1.0 - beta)

        # ── AL objective + step ──────────────────────────────────
        lagrangian = compute_augmented_lagrangian(
            l0_probs.mean() + disc_penalty.mean(), violations,
            controller._lambdas, controller._rhos,
        )
        optimizer.zero_grad(set_to_none=True)
        lagrangian.backward()

        # AdaGC clip (inline).
        for i, p in enumerate(clip_params):
            if p.grad is None:
                continue
            gn = p.grad.data.norm()
            torch.where(gn > ema_norms[i], p.grad.data * (ema_norms[i] / gn),
                        p.grad.data, out=p.grad.data)
            ema_norms[i].mul_(adagc_beta).add_(gn, alpha=1.0 - adagc_beta)

        optimizer.step()
        scheduler.step()
        sae.update_dead_counts(gate_mask)

        # ── Slow-timescale (MTZF γ anneal + KL onset) ────────────
        if controller.should_do_slow_update(step):
            controller.step()

            ratio = min((D_ema / D_0).item(), 1.0)
            with torch.no_grad():
                sae.gamma.copy_(sae.gamma_init * max(ratio, alpha_floor))

            if not kl_active and (controller.v_ema[:controller.n_primal] < 0).all():
                onset_step = step
                with torch.no_grad():
                    orig_logits, patched_logits = run_patched_forward(
                        store, sae, whitener, next(token_iter).cuda(),
                    )
                    tau_kl = compute_kl(orig_logits, patched_logits).item()
                cal["tau_kl"] = tau_kl
                controller.recalibrate(controller.n_primal, tau_kl)
                kl_active = True
                print(json.dumps({"event": "kl_onset", "step": step, "tau_kl": tau_kl},
                                 sort_keys=True), flush=True)

        sae.normalize_free_decoder()

        if kl_active and step % dead_threshold == 0:
            sae.resample_dead_features(x_tilde, x_hat, dead_threshold, cal["L0_target"])

        # ── Checkpointing + logging ──────────────────────────────
        if config.checkpoint_interval > 0 and step % config.checkpoint_interval == 0:
            _save_checkpoint(Path(config.output_dir) / f"checkpoint_step{step}",
                             sae, optimizer, scheduler, controller,
                             whitener, W_vocab, cal, config, step, onset_step, D_ema, D_0)

        if step % config.log_interval == 0:
            diff = x - x_hat
            scalars = torch.stack([gate_mask.sum(dim=1).mean(),
                                   diff.pow(2).sum(dim=1).mean(),
                                   (x - x.mean(dim=0)).pow(2).sum(dim=1).mean()])
            l0_mean, mse, x_var = scalars.tolist()
            metrics = {"event": "train_step", "step": step,
                       "l0": l0_mean, "r2": 1.0 - mse / x_var}
            if kl_active:
                metrics["v_kl"] = controller.v_ema[controller.n_primal].item()
            print(json.dumps(metrics, sort_keys=True), flush=True)

    _save_checkpoint(Path(config.output_dir) / f"checkpoint_step{total_steps}",
                     sae, optimizer, scheduler, controller,
                     whitener, W_vocab, cal, config, total_steps, onset_step, D_ema, D_0)
    return sae
