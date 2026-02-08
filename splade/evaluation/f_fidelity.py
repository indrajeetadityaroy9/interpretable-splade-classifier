"""F-Fidelity evaluation metric (arXiv:2410.02970).

Trains a surrogate model on randomly masked inputs, then measures whether
attribution-guided masking affects the surrogate's predictions more than
random masking.
"""

import copy

import numpy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from splade.evaluation.constants import (F_FIDELITY_ALPHA_POS,
                                         F_FIDELITY_EPOCHS,
                                         F_FIDELITY_EXPLANATION_SIZE,
                                         F_FIDELITY_MASK_FRACTION,
                                         F_FIDELITY_N_SAMPLES)
from splade.evaluation.faithfulness import _top_k_tokens
from splade.utils.cuda import COMPUTE_DTYPE, DEVICE, unwrap_compiled


def finetune_surrogate_model(
    model: torch.nn.Module,
    tokenizer,
    train_texts: list[str],
    train_labels: list[int],
    max_length: int,
    *,
    mask_fraction: float = F_FIDELITY_MASK_FRACTION,
    epochs: int = F_FIDELITY_EPOCHS,
    batch_size: int = 32,
    seed: int = 42,
) -> torch.nn.Module:
    """Fine-tune a copy of the model on randomly masked inputs.

    This creates the surrogate model f_r that is calibrated for masked inputs,
    as required by the F-Fidelity metric.
    """
    _model = unwrap_compiled(model)
    surrogate = copy.deepcopy(_model)
    surrogate.to(DEVICE)
    surrogate.train()

    encoding = tokenizer(
        train_texts, max_length=max_length, padding="max_length",
        truncation=True, return_tensors="pt",
    )
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    labels_tensor = torch.tensor(train_labels, dtype=torch.long)

    loader = DataLoader(
        TensorDataset(input_ids, attention_mask, labels_tensor),
        batch_size=batch_size, shuffle=True,
    )

    mask_token_id = tokenizer.mask_token_id
    rng = numpy.random.default_rng(seed)
    optimizer = torch.optim.AdamW(surrogate.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    for _epoch in range(epochs):
        for batch_ids, batch_mask, batch_labels in loader:
            batch_ids = batch_ids.to(DEVICE)
            batch_mask = batch_mask.to(DEVICE)
            batch_labels = batch_labels.to(DEVICE)

            # Randomly mask tokens
            mask_prob = torch.tensor(
                rng.random(batch_ids.shape), dtype=torch.float32, device=DEVICE,
            )
            content_mask = batch_mask.bool() & (batch_ids != tokenizer.cls_token_id) & (batch_ids != tokenizer.sep_token_id)
            apply_mask = (mask_prob < mask_fraction) & content_mask
            masked_ids = batch_ids.clone()
            masked_ids[apply_mask] = mask_token_id

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=COMPUTE_DTYPE):
                logits, _ = surrogate(masked_ids, batch_mask)
                loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

    surrogate.eval()
    return surrogate


def compute_f_fidelity(
    surrogate_model: torch.nn.Module,
    tokenizer,
    texts: list[str],
    labels: list[int],
    attributions: list[list[tuple[str, float]]],
    max_length: int,
    *,
    explanation_size: int = F_FIDELITY_EXPLANATION_SIZE,
    alpha_pos: float = F_FIDELITY_ALPHA_POS,
    beta: float = F_FIDELITY_MASK_FRACTION,
    n_samples: int = F_FIDELITY_N_SAMPLES,
    batch_size: int = 32,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute F-Fidelity+ and F-Fidelity- metrics.

    F-Fidelity+ measures accuracy drop when perturbing explanation tokens.
    F-Fidelity- measures accuracy drop when perturbing non-explanation tokens.
    """
    _surrogate = unwrap_compiled(surrogate_model)
    rng = numpy.random.default_rng(seed)

    encoding = tokenizer(
        texts, max_length=max_length, padding="max_length",
        truncation=True, return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)
    labels_tensor = torch.tensor(labels, dtype=torch.long, device=DEVICE)

    # Get baseline accuracy on unmasked inputs
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=COMPUTE_DTYPE):
        base_logits, _ = _surrogate(input_ids, attention_mask)
    base_preds = base_logits.argmax(dim=-1)
    base_correct = (base_preds == labels_tensor).float()

    ffid_pos_scores = []
    ffid_neg_scores = []

    for idx in range(len(texts)):
        text = texts[idx]
        words = text.split()
        attrib = attributions[idx]
        label = labels[idx]

        # Identify explanation tokens (top-s by attribution)
        top_tokens = _top_k_tokens(attrib, explanation_size)
        word_indices = list(range(len(words)))
        expl_indices = [i for i in word_indices
                        if words[i].lower().strip('.,!?;:"\'-') in top_tokens]
        non_expl_indices = [i for i in word_indices if i not in set(expl_indices)]

        s = len(expl_indices)
        td_minus_s = len(non_expl_indices)

        sample_pos_correct = []
        sample_neg_correct = []

        for _ in range(n_samples):
            # F-Fidelity+: remove alpha_pos fraction of explanation tokens + beta of rest
            n_remove_expl = max(1, int(alpha_pos * s)) if s > 0 else 0
            n_remove_rest = max(0, int(beta * td_minus_s)) if td_minus_s > 0 else 0
            remove_expl = set(rng.choice(expl_indices, size=min(n_remove_expl, s), replace=False)) if n_remove_expl > 0 and s > 0 else set()
            remove_rest = set(rng.choice(non_expl_indices, size=min(n_remove_rest, td_minus_s), replace=False)) if n_remove_rest > 0 and td_minus_s > 0 else set()
            remove_pos = remove_expl | remove_rest

            masked_words_pos = [
                tokenizer.mask_token if i in remove_pos else words[i]
                for i in range(len(words))
            ]
            sample_pos_correct.append(" ".join(masked_words_pos))

            # F-Fidelity-: remove alpha_pos fraction of non-explanation tokens + beta of explanation
            n_remove_non = max(1, int(alpha_pos * td_minus_s)) if td_minus_s > 0 else 0
            n_remove_expl_beta = max(0, int(beta * s)) if s > 0 else 0
            remove_non = set(rng.choice(non_expl_indices, size=min(n_remove_non, td_minus_s), replace=False)) if n_remove_non > 0 and td_minus_s > 0 else set()
            remove_expl_b = set(rng.choice(expl_indices, size=min(n_remove_expl_beta, s), replace=False)) if n_remove_expl_beta > 0 and s > 0 else set()
            remove_neg = remove_non | remove_expl_b

            masked_words_neg = [
                tokenizer.mask_token if i in remove_neg else words[i]
                for i in range(len(words))
            ]
            sample_neg_correct.append(" ".join(masked_words_neg))

        # Batch predict all samples for this text
        all_masked = sample_pos_correct + sample_neg_correct
        enc = tokenizer(
            all_masked, max_length=max_length, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=COMPUTE_DTYPE):
            logits_all, _ = _surrogate(
                enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE),
            )
        preds_all = logits_all.argmax(dim=-1)

        pos_preds = preds_all[:n_samples]
        neg_preds = preds_all[n_samples:]

        is_base_correct = float(base_correct[idx].item())
        pos_correct = (pos_preds == label).float().mean().item()
        neg_correct = (neg_preds == label).float().mean().item()

        ffid_pos_scores.append(is_base_correct - pos_correct)
        ffid_neg_scores.append(is_base_correct - neg_correct)

    ffid_pos = sum(ffid_pos_scores) / len(ffid_pos_scores) if ffid_pos_scores else 0.0
    ffid_neg = sum(ffid_neg_scores) / len(ffid_neg_scores) if ffid_neg_scores else 0.0
    return ffid_pos, ffid_neg
