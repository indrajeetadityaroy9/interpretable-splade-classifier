from collections import Counter

import numpy
import torch

from splade.evaluation.token_alignment import _clean_subword


class UnigramSampler:
    def __init__(self, texts: list[str], seed: int):
        counts: Counter[str] = Counter()
        for text in texts:
            for word in text.lower().split():
                normalized_word = word.strip('.,!?;:"\'-')
                if normalized_word:
                    counts[normalized_word] += 1
        total = sum(counts.values())
        self.words = list(counts.keys())
        self.probs = numpy.array([counts[word] / total for word in self.words])
        self.rng = numpy.random.default_rng(seed)

    def sample(self) -> str:
        return str(self.rng.choice(self.words, p=self.probs))


def _top_k_tokens(attrib: list[tuple[str, float]], k: int) -> set[str]:
    seen: set[str] = set()
    result: set[str] = set()
    for token, weight in attrib:
        if weight <= 0:
            continue
        lowered = token.lower().strip('.,!?;:"\'-')
        if lowered not in seen:
            seen.add(lowered)
            result.add(lowered)
            if len(result) >= k:
                break
    return result


def compute_filler_comprehensiveness(
    model,
    texts: list[str],
    attributions: list[list[tuple[str, float]]],
    k_values: tuple[int, ...] | list[int],
    sampler: UnigramSampler,
    original_probs: list[list[float]],
) -> dict[int, float]:
    results = {k: [] for k in k_values}
    original_probabilities = original_probs

    filled_texts = []
    index_map = []
    for text_index, (text, attrib) in enumerate(zip(texts, attributions)):
        for k in k_values:
            top_tokens = _top_k_tokens(attrib, k)
            normalized = {token.strip('.,!?;:"\'-').lower() for token in top_tokens}
            words = text.split()
            filled_words = [
                sampler.sample() if word.lower().strip('.,!?;:"\'-') in normalized else word
                for word in words
            ]
            filled_texts.append(" ".join(filled_words))
            index_map.append((text_index, k))

    all_probabilities = model.predict_proba(filled_texts) if filled_texts else []

    for index, (text_index, k) in enumerate(index_map):
        original_probability = original_probabilities[text_index]
        predicted_class = int(numpy.argmax(original_probability))
        original_confidence = original_probability[predicted_class]
        filled_confidence = all_probabilities[index][predicted_class]
        results[k].append(original_confidence - filled_confidence)

    return {k: float(numpy.mean(scores)) for k, scores in results.items()}


def _build_word_importance_map(text: str, attrib: list[tuple[str, float]]) -> dict[int, float]:
    attrib_dict: dict[str, float] = {}
    for token, weight in attrib:
        key = _clean_subword(token).lower()
        if key not in attrib_dict:
            attrib_dict[key] = abs(weight)

    word_importance: dict[int, float] = {}
    for index, word in enumerate(text.split()):
        clean = word.lower().strip('.,!?;:"\'-')
        if clean in attrib_dict:
            word_importance[index] = attrib_dict[clean]
    return word_importance


def _build_token_importances(
    text: str,
    attrib: list[tuple[str, float]],
    tokenizer,
    max_length: int,
) -> numpy.ndarray:
    word_importance = _build_word_importance_map(text, attrib)
    words = text.split()

    encoding = tokenizer(
        text, max_length=max_length, padding="max_length",
        truncation=True, return_offsets_mapping=True,
    )
    offsets = encoding["offset_mapping"]
    input_ids = encoding["input_ids"]

    char_to_word: dict[int, int] = {}
    pos = 0
    for word_idx, word in enumerate(words):
        start = text.find(word, pos)
        if start == -1:
            continue
        for c in range(start, start + len(word)):
            char_to_word[c] = word_idx
        pos = start + len(word)

    importances = numpy.zeros(len(input_ids), dtype=numpy.float64)
    for tok_idx, (start, end) in enumerate(offsets):
        if start == 0 and end == 0:
            continue
        mid = (start + end) // 2
        word_idx = char_to_word.get(mid)
        if word_idx is not None and word_idx in word_importance:
            importances[tok_idx] = word_importance[word_idx]

    max_imp = importances.max()
    if max_imp > 1e-12:
        importances = importances / max_imp
    return importances


def compute_soft_metrics(
    model,
    texts: list[str],
    attributions: list[list[tuple[str, float]]],
    mask_token: str,
    *,
    n_samples: int = 20,
    seed: int = 42,
    tokenizer=None,
    max_length: int = 128,
    original_probs: list[list[float]],
) -> tuple[float, float]:
    orig_probs = original_probs

    embeddings, attention_masks = model.get_embeddings(texts)
    embed_dim = embeddings.shape[-1]
    zero_emb = torch.zeros_like(embeddings)
    baseline_probs = model.predict_proba_from_embeddings(zero_emb, attention_masks)

    comp_scores = []
    suff_scores = []
    for text_idx in range(len(texts)):
        token_importances = _build_token_importances(
            texts[text_idx], attributions[text_idx], tokenizer, max_length
        )

        predicted_class = int(numpy.argmax(orig_probs[text_idx]))
        orig_conf = orig_probs[text_idx][predicted_class]
        base_conf = baseline_probs[text_idx][predicted_class]

        emb_i = embeddings[text_idx]
        mask_i = attention_masks[text_idx]
        full_L = emb_i.shape[0]

        seq_len = min(len(token_importances), full_L)
        imp_tensor = torch.tensor(
            token_importances[:seq_len], dtype=torch.float32, device=emb_i.device
        ).unsqueeze(1)

        rand_comp = torch.rand(n_samples, seq_len, embed_dim, device=emb_i.device)
        keep_masks = (rand_comp >= imp_tensor.unsqueeze(0)).float()
        if seq_len < full_L:
            keep_masks = torch.cat([keep_masks, torch.ones(n_samples, full_L - seq_len, embed_dim, device=emb_i.device)], dim=1)
        pert_comp = emb_i.unsqueeze(0) * keep_masks
        mask_batch = mask_i.unsqueeze(0).expand(n_samples, -1)

        rand_suff = torch.rand(n_samples, seq_len, embed_dim, device=emb_i.device)
        retain_masks = (rand_suff < imp_tensor.unsqueeze(0)).float()
        if seq_len < full_L:
            retain_masks = torch.cat([retain_masks, torch.zeros(n_samples, full_L - seq_len, embed_dim, device=emb_i.device)], dim=1)
        pert_suff = emb_i.unsqueeze(0) * retain_masks

        comp_probs_t = model.predict_proba_from_embeddings_tensor(pert_comp, mask_batch)
        comp_drop = float((orig_conf - comp_probs_t[:, predicted_class]).mean().item())
        suff_probs_t = model.predict_proba_from_embeddings_tensor(pert_suff, mask_batch)
        suff_drop = float((orig_conf - suff_probs_t[:, predicted_class]).mean().item())

        raw_comp = max(0.0, comp_drop)
        baseline_suff = 1.0 - max(0.0, orig_conf - base_conf)
        denom = 1.0 - baseline_suff
        comp_scores.append(raw_comp / denom if denom > 1e-8 else 0.0)

        raw_suff = 1.0 - max(0.0, suff_drop)
        suff_scores.append(max(0.0, raw_suff - baseline_suff) / denom if denom > 1e-8 else 0.0)

    soft_comp = float(numpy.mean(comp_scores)) if comp_scores else 0.0
    soft_suff = float(numpy.mean(suff_scores)) if suff_scores else 0.0
    return soft_comp, soft_suff
