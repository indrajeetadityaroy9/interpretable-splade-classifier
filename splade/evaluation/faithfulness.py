import random
from collections import Counter

import numpy
import torch
from scipy.stats import kendalltau

from splade.evaluation.token_alignment import _clean_subword, normalize_attributions_to_words


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
    baseline_probs = model.predict_proba_from_embeddings(zero_emb, attention_masks).tolist()

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

        comp_probs_t = model.predict_proba_from_embeddings(pert_comp, mask_batch)
        comp_drop = float((orig_conf - comp_probs_t[:, predicted_class]).mean().item())
        suff_probs_t = model.predict_proba_from_embeddings(pert_suff, mask_batch)
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


def _apply_word_mask(text: str, top_tokens: set[str], mask_token: str, *, keep: bool = False) -> str:
    words = text.split()
    return " ".join(
        mask_token if (word.lower().strip('.,!?;:"\'-') in top_tokens) != keep else word
        for word in words
    )


def compute_eraser_comprehensiveness(
    model,
    texts: list[str],
    attributions: list[list[tuple[str, float]]],
    k_values: tuple[int, ...] | list[int],
    mask_token: str,
    original_probs: list[list[float]],
) -> dict[int, float]:
    """ERASER comprehensiveness: replace top-k tokens with [MASK], measure confidence drop."""
    results = {k: [] for k in k_values}

    masked_texts = []
    index_map = []
    for text_index, (text, attrib) in enumerate(zip(texts, attributions)):
        for k in k_values:
            top_tokens = _top_k_tokens(attrib, k)
            masked_texts.append(_apply_word_mask(text, top_tokens, mask_token))
            index_map.append((text_index, k))

    all_probs = model.predict_proba(masked_texts) if masked_texts else []

    for index, (text_index, k) in enumerate(index_map):
        predicted_class = int(numpy.argmax(original_probs[text_index]))
        orig_conf = original_probs[text_index][predicted_class]
        masked_conf = all_probs[index][predicted_class]
        results[k].append(orig_conf - masked_conf)

    return {k: float(numpy.mean(scores)) for k, scores in results.items()}


def compute_eraser_sufficiency(
    model,
    texts: list[str],
    attributions: list[list[tuple[str, float]]],
    k_values: tuple[int, ...] | list[int],
    mask_token: str,
    original_probs: list[list[float]],
) -> dict[int, float]:
    """ERASER sufficiency: keep only top-k tokens, mask everything else. Lower = more sufficient."""
    results = {k: [] for k in k_values}

    retained_texts = []
    index_map = []
    for text_index, (text, attrib) in enumerate(zip(texts, attributions)):
        for k in k_values:
            top_tokens = _top_k_tokens(attrib, k)
            retained_texts.append(_apply_word_mask(text, top_tokens, mask_token, keep=True))
            index_map.append((text_index, k))

    all_probs = model.predict_proba(retained_texts) if retained_texts else []

    for index, (text_index, k) in enumerate(index_map):
        predicted_class = int(numpy.argmax(original_probs[text_index]))
        orig_conf = original_probs[text_index][predicted_class]
        retained_conf = all_probs[index][predicted_class]
        results[k].append(orig_conf - retained_conf)

    return {k: float(numpy.mean(scores)) for k, scores in results.items()}


def compute_monotonicity(
    model,
    texts: list[str],
    attributions: list[list[tuple[str, float]]],
    mask_token: str,
    original_probs: list[list[float]],
    max_steps: int = 20,
) -> float:
    """Monotonicity: fraction of steps where confidence decreases when removing tokens in rank order."""
    all_monotonic_fractions = []

    for text_index, (text, attrib) in enumerate(zip(texts, attributions)):
        predicted_class = int(numpy.argmax(original_probs[text_index]))
        orig_conf = original_probs[text_index][predicted_class]

        # Get ordered list of unique attributed words
        ordered_words = []
        seen: set[str] = set()
        for token, weight in attrib:
            if weight <= 0:
                continue
            lowered = token.lower().strip('.,!?;:"\'-')
            if lowered not in seen:
                seen.add(lowered)
                ordered_words.append(lowered)
            if len(ordered_words) >= max_steps:
                break

        if len(ordered_words) < 2:
            continue

        # Build progressively masked texts
        masked_variants = []
        cumulative_removed: set[str] = set()
        for word in ordered_words:
            cumulative_removed.add(word)
            masked_variants.append(_apply_word_mask(text, cumulative_removed, mask_token))

        step_probs = model.predict_proba(masked_variants)

        # Check monotonicity: each step should have lower or equal confidence
        prev_conf = orig_conf
        monotonic_count = 0
        for step_prob in step_probs:
            step_conf = step_prob[predicted_class]
            if step_conf <= prev_conf + 1e-9:  # small tolerance for floating point
                monotonic_count += 1
            prev_conf = step_conf

        all_monotonic_fractions.append(monotonic_count / len(masked_variants))

    return float(numpy.mean(all_monotonic_fractions)) if all_monotonic_fractions else 0.0


def compute_aopc(
    model,
    texts: list[str],
    attributions: list[list[tuple[str, float]]],
    mask_token: str,
    original_probs: list[list[float]],
    k_values: tuple[int, ...] | list[int],
) -> dict[int, float]:
    """AOPC: Area Over the Perturbation Curve. Average confidence drop across k values."""
    results = {k: [] for k in k_values}

    masked_texts = []
    index_map = []
    for text_index, (text, attrib) in enumerate(zip(texts, attributions)):
        for k in k_values:
            top_tokens = _top_k_tokens(attrib, k)
            masked_texts.append(_apply_word_mask(text, top_tokens, mask_token))
            index_map.append((text_index, k))

    all_probs = model.predict_proba(masked_texts) if masked_texts else []

    per_text_drops: dict[int, dict[int, float]] = {}
    for index, (text_index, k) in enumerate(index_map):
        predicted_class = int(numpy.argmax(original_probs[text_index]))
        orig_conf = original_probs[text_index][predicted_class]
        masked_conf = all_probs[index][predicted_class]
        drop = orig_conf - masked_conf
        if text_index not in per_text_drops:
            per_text_drops[text_index] = {}
        per_text_drops[text_index][k] = drop

    # AOPC at each k = mean across texts of (1/k * sum of drops up to k)
    for k in k_values:
        k_subset = sorted([kv for kv in k_values if kv <= k])
        aopc_scores = []
        for text_index in per_text_drops:
            drops = [per_text_drops[text_index].get(kv, 0.0) for kv in k_subset]
            aopc_scores.append(float(numpy.mean(drops)))
        results[k] = float(numpy.mean(aopc_scores)) if aopc_scores else 0.0

    return results


def compute_naopc(
    model,
    texts: list[str],
    attributions: list[list[tuple[str, float]]],
    mask_token: str,
    original_probs: list[list[float]],
    k_values: tuple[int, ...] | list[int],
    *,
    beam_size: int = 8,
    seed: int = 42,
) -> dict[int, float]:
    """NAOPC: Normalized AOPC using beam-search upper/lower bounds."""
    rng = numpy.random.default_rng(seed)
    max_k = max(k_values)

    explanation_aopc = compute_aopc(
        model, texts, attributions, mask_token, original_probs, k_values,
    )

    # Compute beam-search upper and lower bounds
    upper_scores = {k: [] for k in k_values}
    lower_scores = {k: [] for k in k_values}

    for text_index, text in enumerate(texts):
        predicted_class = int(numpy.argmax(original_probs[text_index]))
        orig_conf = original_probs[text_index][predicted_class]
        words = text.split()
        unique_words = []
        seen_w: set[str] = set()
        for w in words:
            clean = w.lower().strip('.,!?;:"\'-')
            if clean and clean not in seen_w:
                seen_w.add(clean)
                unique_words.append(clean)

        n_words = min(len(unique_words), max_k)
        if n_words < 1:
            continue

        # Beam search for upper bound (max drop) and lower bound (min drop)
        # Each beam entry is (ordering_so_far, cumulative_drops)
        upper_beams = [([], [])]
        lower_beams = [([], [])]

        remaining_all = set(unique_words[:n_words])

        for step in range(n_words):
            # Expand upper beams
            upper_candidates = []
            expand_texts_upper = []
            expand_info_upper = []
            for ordering, drops in upper_beams:
                remaining = remaining_all - set(ordering)
                for word in remaining:
                    new_ordering = ordering + [word]
                    removed_set = set(new_ordering)
                    expand_texts_upper.append(_apply_word_mask(text, removed_set, mask_token))
                    expand_info_upper.append((new_ordering, drops))

            if expand_texts_upper:
                expand_probs = model.predict_proba(expand_texts_upper)
                for idx, (new_ordering, prev_drops) in enumerate(expand_info_upper):
                    conf = expand_probs[idx][predicted_class]
                    drop = orig_conf - conf
                    new_drops = prev_drops + [drop]
                    upper_candidates.append((new_ordering, new_drops, numpy.mean(new_drops)))
                upper_candidates.sort(key=lambda x: x[2], reverse=True)
                upper_beams = [(o, d) for o, d, _ in upper_candidates[:beam_size]]

            # Expand lower beams
            lower_candidates = []
            expand_texts_lower = []
            expand_info_lower = []
            for ordering, drops in lower_beams:
                remaining = remaining_all - set(ordering)
                for word in remaining:
                    new_ordering = ordering + [word]
                    removed_set = set(new_ordering)
                    expand_texts_lower.append(_apply_word_mask(text, removed_set, mask_token))
                    expand_info_lower.append((new_ordering, drops))

            if expand_texts_lower:
                expand_probs = model.predict_proba(expand_texts_lower)
                for idx, (new_ordering, prev_drops) in enumerate(expand_info_lower):
                    conf = expand_probs[idx][predicted_class]
                    drop = orig_conf - conf
                    new_drops = prev_drops + [drop]
                    lower_candidates.append((new_ordering, new_drops, numpy.mean(new_drops)))
                lower_candidates.sort(key=lambda x: x[2])
                lower_beams = [(o, d) for o, d, _ in lower_candidates[:beam_size]]

            # Record scores at this step count for relevant k_values
            step_count = step + 1
            if step_count in k_values:
                if upper_beams:
                    best_upper = upper_beams[0][1]
                    upper_scores[step_count].append(float(numpy.mean(best_upper)))
                if lower_beams:
                    best_lower = lower_beams[0][1]
                    lower_scores[step_count].append(float(numpy.mean(best_lower)))

    # Normalize: NAOPC = (AOPC_expl - AOPC_lower) / (AOPC_upper - AOPC_lower)
    naopc_results = {}
    for k in k_values:
        upper_mean = float(numpy.mean(upper_scores[k])) if upper_scores[k] else 1.0
        lower_mean = float(numpy.mean(lower_scores[k])) if lower_scores[k] else 0.0
        expl_score = explanation_aopc.get(k, 0.0)
        denom = upper_mean - lower_mean
        if abs(denom) < 1e-8:
            naopc_results[k] = 0.5
        else:
            naopc_results[k] = (expl_score - lower_mean) / denom

    return naopc_results


def _generate_perturbation(text: str, attrib: list[tuple[str, float]], rng: random.Random) -> str | None:
    """Generate a perturbed version of text by swapping adjacent chars in a random word."""
    words = text.split()
    if len(words) < 2:
        return None

    # Prefer low-importance words for perturbation
    low_importance = []
    high_importance_words = set()
    for token, weight in attrib[:len(attrib) // 2]:
        high_importance_words.add(token.lower().strip('.,!?;:"\'-'))
    for idx, word in enumerate(words):
        clean = word.lower().strip('.,!?;:"\'-')
        if clean not in high_importance_words and len(clean) > 2:
            low_importance.append(idx)

    if not low_importance:
        # Fall back to any word with length > 2
        low_importance = [i for i, w in enumerate(words) if len(w) > 2]
    if not low_importance:
        return None

    target_idx = rng.choice(low_importance)
    word = words[target_idx]
    if len(word) < 2:
        return None

    # Swap two adjacent characters
    pos = rng.randint(0, len(word) - 2)
    chars = list(word)
    chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
    new_words = words.copy()
    new_words[target_idx] = "".join(chars)
    return " ".join(new_words)


def compute_adversarial_sensitivity(
    explain_fn,
    texts: list[str],
    attributions: list[list[tuple[str, float]]],
    tokenizer,
    *,
    n_perturbations: int = 5,
    seed: int = 42,
) -> float:
    """Adversarial sensitivity: ranking stability under text perturbations (higher = more stable)."""
    rng = random.Random(seed)
    all_taus = []

    for text_index, (text, orig_attrib) in enumerate(zip(texts, attributions)):
        orig_ranking = [w.lower().strip('.,!?;:"\'-') for w, _ in orig_attrib if _ > 0]
        if len(orig_ranking) < 3:
            continue

        for _ in range(n_perturbations):
            perturbed = _generate_perturbation(text, orig_attrib, rng)
            if perturbed is None:
                continue

            raw_pert_attrib = explain_fn(perturbed, len(orig_ranking))
            pert_attrib = normalize_attributions_to_words(perturbed, raw_pert_attrib, tokenizer)

            pert_ranking = [w.lower().strip('.,!?;:"\'-') for w, _ in pert_attrib if _ > 0]
            if len(pert_ranking) < 3:
                continue

            # Build rank vectors for common words
            common_words = [w for w in orig_ranking if w in set(pert_ranking)]
            if len(common_words) < 3:
                continue

            orig_ranks = [orig_ranking.index(w) for w in common_words]
            pert_ranks = [pert_ranking.index(w) for w in common_words]

            tau, _ = kendalltau(orig_ranks, pert_ranks)
            if not numpy.isnan(tau):
                all_taus.append(tau)

    return float(numpy.mean(all_taus)) if all_taus else 0.0
