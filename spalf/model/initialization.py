import torch

from spalf.data.store import ActivationStore
from spalf.optim.constraints import compute_orthogonality_violation
from spalf.model.sae import StratifiedSAE


def initialize_from_calibration(cal: dict, store: ActivationStore) -> StratifiedSAE:
    """Create and initialize the SAE from calibration. Mutates cal['tau_ortho']."""
    d, V, F = cal["d"], cal["V"], cal["F"]
    whitener, W_vocab = cal["whitener"], cal["W_vocab"]
    sae = StratifiedSAE(d, F, V).cuda()

    # Activation sample: F² / L0 tokens (≈F per feature), bounded by buffer.
    n_needed = min(max(F * F // cal["L0_target"], F), cal["buffer_size"])
    samples = []
    while sum(s.shape[0] for s in samples) < n_needed:
        samples.append(store.next_batch())
    x_raw = torch.cat(samples, dim=0)[:n_needed]

    with torch.no_grad():
        sae.W_dec_A.copy_(W_vocab)
        sae.b_dec.copy_(whitener.mean)
        free = torch.randn(d, sae.F_free, device="cuda")
        sae.W_dec_B.copy_(free / free.norm(dim=0, keepdim=True))

        # Anchored encoder = matched filter: w_enc = Σ^{-1/2} w_vocab.
        if whitener.is_low_rank:
            proj = W_vocab.T @ whitener._U_k
            top = (proj * whitener._scale_k) @ whitener._U_k.T
            tail = (W_vocab.T - proj @ whitener._U_k.T) * whitener._scale_tail
            sae.W_enc.data[:V] = top + tail
        else:
            sae.W_enc.data[:V] = (whitener._W_white @ W_vocab).T

        # Free encoder: QR-orthogonalize against anchored rows, unit-norm.
        W_enc_B = torch.randn(sae.F_free, d, device="cuda") / (d ** 0.5)
        n_ortho = min(sae.F_free, d - V)
        if n_ortho > 0:
            Q, _ = torch.linalg.qr(sae.W_enc.data[:V].T)
            W_enc_B[:n_ortho] -= (W_enc_B[:n_ortho] @ Q) @ Q.T
        W_enc_B /= W_enc_B.norm(dim=1, keepdim=True)
        sae.W_enc.data[V:] = W_enc_B

        # JumpReLU thresholds + bandwidth from sample statistics.
        x_tilde = whitener.forward(x_raw)
        pre_act = x_tilde @ sae.W_enc.T + sae.b_enc
        thresholds = torch.quantile(pre_act, 1.0 - cal["L0_target"] / F, dim=0)
        sae.log_threshold.data = thresholds.clamp_min(torch.finfo(thresholds.dtype).eps).log()
        sae.recalibrate_gamma(pre_act)
        sae.gamma_init.copy_(sae.gamma)

        _, z_init, _, _, _ = sae(x_tilde)
        raw_ortho = compute_orthogonality_violation(z_init, sae.W_dec_A, sae.W_dec_B).item()
    cal["tau_ortho"] = max(raw_ortho, 1.0 / d)
    return sae
