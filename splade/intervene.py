"""Surgical intervention API for Lexical-SAE.

Provides verifiable concept removal at the sparse bottleneck level.
Since logit[c] = sum_j s[j] * W_eff[c,j] + b_eff[c], zeroing s[j]
removes token j's contribution to ALL classes with mathematical certainty.

Two mechanisms:
  1. Global suppression via weight surgery (permanent, modifies vocab_projector)
  2. Inference-time suppression via SuppressedCISModel wrapper (reversible)
"""

import torch
import torch.nn as nn

from splade.circuits.core import CircuitState
from splade.utils.cuda import COMPUTE_DTYPE, DEVICE, unwrap_compiled


def get_clean_vocab_mask(tokenizer) -> torch.Tensor:
    """Return a boolean mask selecting whole-word, non-special tokens.

    Filters out special tokens ([CLS], [SEP], [PAD], [UNK], [MASK]),
    subword continuations (## prefix for BERT/DistilBERT), and single
    alphanumeric characters. This ensures surgery demos and top-token
    displays show human-interpretable concepts, not tokenizer artifacts.
    """
    vocab_size = tokenizer.vocab_size
    mask = torch.ones(vocab_size, dtype=torch.bool)

    special_ids = set(tokenizer.all_special_ids)
    for token, tid in tokenizer.get_vocab().items():
        if tid >= vocab_size:
            continue
        if tid in special_ids:
            mask[tid] = False
        elif token.startswith("##"):
            mask[tid] = False
        elif token.startswith("[unused"):
            mask[tid] = False
        elif len(token) == 1 and token.isalnum():
            mask[tid] = False

    return mask


def get_top_tokens(
    model: nn.Module,
    tokenizer,
    class_idx: int,
    centroid_tracker=None,
    top_k: int = 20,
    clean_vocab: bool = True,
) -> list[tuple[str, float]]:
    """Return top-k attributed vocabulary tokens for a class.

    Uses training centroids if available, otherwise requires a dataset pass.
    Returns list of (token_name, attribution_score) sorted by score descending.

    If clean_vocab=True, filters out subwords, special tokens, and single
    characters so results show only human-readable whole-word concepts.
    """
    _model = unwrap_compiled(model)

    if centroid_tracker is not None and centroid_tracker._initialized[class_idx]:
        attr = centroid_tracker.centroids[class_idx].clone()
    else:
        raise ValueError(
            "centroid_tracker required with initialized centroids for class "
            f"{class_idx}. Train the model first."
        )

    if clean_vocab:
        vocab_mask = get_clean_vocab_mask(tokenizer).to(attr.device)
        # Zero out non-clean tokens so they never appear in topk
        attr[:vocab_mask.shape[0]] *= vocab_mask.float()

    scores, indices = attr.topk(top_k)
    tokens = tokenizer.convert_ids_to_tokens(indices.tolist())
    return list(zip(tokens, scores.tolist()))


def suppress_token_globally(model: nn.Module, token_id: int) -> None:
    """Permanently remove a token from the model's vocabulary.

    Zeros the vocab_projector weights for this token, guaranteeing that
    s[token_id] = 0 for all inputs. This is an irreversible, verifiable
    safety guarantee: the concept cannot influence any class.

    Mathematical guarantee: If vocab_projector.weight[token_id, :] = 0
    and vocab_projector.bias[token_id] = 0, then the MLM logit for this
    token is always 0, so after DReLU (which has threshold >= 0),
    s[token_id] = 0 for all inputs.
    """
    _model = unwrap_compiled(model)
    with torch.no_grad():
        _model.vocab_projector.weight[token_id, :] = 0
        _model.vocab_projector.bias[token_id] = 0
        _model.activation.theta[token_id] = 1e6  # ensure DReLU blocks it


def suppress_tokens_by_name(
    model: nn.Module,
    tokenizer,
    token_names: list[str],
) -> list[int]:
    """Suppress multiple tokens by name. Returns list of suppressed token IDs."""
    suppressed = []
    for name in token_names:
        ids = tokenizer.convert_tokens_to_ids([name])
        if ids and ids[0] != tokenizer.unk_token_id:
            suppress_token_globally(model, ids[0])
            suppressed.append(ids[0])
    return suppressed


class SuppressedCISModel(nn.Module):
    """Inference-time wrapper that masks tokens in the sparse vector.

    Unlike global suppression, this is reversible and does not modify
    the underlying model weights. Useful for experimentation.

    The suppression mask zeros specified dimensions of the sparse vector
    before classification. Since logit[c] = sum_j s[j] * W_eff[c,j] + b_eff[c],
    this cleanly removes the masked tokens' contributions while preserving
    the exact DLA identity for remaining tokens.
    """

    def __init__(self, model: nn.Module, suppressed_token_ids: list[int]):
        super().__init__()
        self.model = model
        _orig = unwrap_compiled(model)
        mask = torch.ones(_orig.padded_vocab_size, device=DEVICE)
        for tid in suppressed_token_ids:
            mask[tid] = 0.0
        self.register_buffer("keep_mask", mask)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> CircuitState:
        logits, sparse, W_eff, b_eff = self.model(input_ids, attention_mask)
        clean_sparse = sparse * self.keep_mask
        _orig = unwrap_compiled(self.model)
        new_logits, new_W_eff, new_b_eff = _orig.classifier_forward(clean_sparse)
        return CircuitState(new_logits, clean_sparse, new_W_eff, new_b_eff)

    @property
    def padded_vocab_size(self):
        return unwrap_compiled(self.model).padded_vocab_size

    def classifier_logits_only(self, sparse_vector: torch.Tensor) -> torch.Tensor:
        return unwrap_compiled(self.model).classifier_logits_only(sparse_vector)

    def classifier_parameters(self):
        return unwrap_compiled(self.model).classifier_parameters()


def evaluate_bias(
    model: nn.Module,
    tokenizer,
    texts: list[str],
    labels: list[int],
    identities: list[dict[str, bool]],
    max_length: int,
    batch_size: int = 32,
) -> dict:
    """Compute accuracy and false positive rate broken down by identity group.

    Returns dict with:
        overall_accuracy, overall_fpr,
        per_identity: {name: {accuracy, fpr, count, fpr_gap}}
    """
    from splade.inference import _predict_model

    preds = _predict_model(model, tokenizer, texts, max_length, batch_size, num_labels=2)

    # Overall metrics
    correct = sum(1 for p, l in zip(preds, labels) if p == l)
    overall_acc = correct / len(labels) if labels else 0.0
    neg_count = sum(1 for l in labels if l == 0)
    fp_count = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 0)
    overall_fpr = fp_count / neg_count if neg_count > 0 else 0.0

    # Per-identity metrics
    identity_names = set()
    for ident in identities:
        identity_names.update(k for k, v in ident.items() if v)

    per_identity = {}
    for name in sorted(identity_names):
        group_indices = [i for i, ident in enumerate(identities) if ident.get(name, False)]
        if len(group_indices) < 10:
            continue
        g_preds = [preds[i] for i in group_indices]
        g_labels = [labels[i] for i in group_indices]
        g_correct = sum(1 for p, l in zip(g_preds, g_labels) if p == l)
        g_neg = sum(1 for l in g_labels if l == 0)
        g_fp = sum(1 for p, l in zip(g_preds, g_labels) if p == 1 and l == 0)
        g_fpr = g_fp / g_neg if g_neg > 0 else 0.0
        per_identity[name] = {
            "accuracy": g_correct / len(g_labels),
            "fpr": g_fpr,
            "count": len(group_indices),
            "fpr_gap": g_fpr - overall_fpr,
        }

    return {
        "overall_accuracy": overall_acc,
        "overall_fpr": overall_fpr,
        "per_identity": per_identity,
    }
