"""Word-order sensitivity evaluation.

Tests whether Lexical-SAE produces different representations for sentences
with the same words in different orders (proving it's not a bag-of-words model).

Uses minimal negation pairs where word order changes meaning.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset

from splade.utils.cuda import COMPUTE_DTYPE, DEVICE, unwrap_compiled


NEGATION_PAIRS = [
    ("This movie is not bad at all.", "This movie is bad, not at all what I expected."),
    ("I would not say this is good.", "I would say this is not good."),
    ("The food was not terrible.", "The food was terrible, not what I wanted."),
    ("She is not unhappy with the result.", "She is unhappy, not satisfied with the result."),
    ("This is not the worst I have seen.", "This is the worst, not what I hoped for."),
    ("I can not complain about the service.", "I can complain, not satisfied with the service."),
    ("The performance was not disappointing.", "The performance was disappointing, not great."),
]


def evaluate_word_order_sensitivity(
    model: torch.nn.Module,
    tokenizer,
    max_length: int,
    pairs: list[tuple[str, str]] | None = None,
) -> dict:
    """Evaluate word-order sensitivity using minimal pairs.

    Runs both sentences of each pair through the model, computes L2 distance
    and cosine similarity between sparse representations, and reports
    prediction agreement.

    High divergence + low agreement = model is order-sensitive (good).

    Args:
        model: LexicalSAE model.
        tokenizer: HuggingFace tokenizer.
        max_length: Tokenizer max length.
        pairs: Sentence pairs (defaults to NEGATION_PAIRS).

    Returns:
        {mean_l2_distance, mean_cosine_similarity, prediction_agreement_rate,
         num_pairs, per_pair: [{sent_a, sent_b, l2, cosine, pred_a, pred_b}]}
    """
    if pairs is None:
        pairs = NEGATION_PAIRS

    _model = unwrap_compiled(model)

    all_texts_a = [a for a, b in pairs]
    all_texts_b = [b for a, b in pairs]

    def _encode_and_forward(texts):
        encoding = tokenizer(
            texts, max_length=max_length, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        ids = encoding["input_ids"].to(DEVICE)
        mask = encoding["attention_mask"].to(DEVICE)
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=COMPUTE_DTYPE):
            sparse_seq, *_ = _model(ids, mask)
            sparse_vector = _model.to_pooled(sparse_seq, mask)
            logits = _model.classify(sparse_seq, mask).logits
        preds = logits.argmax(dim=-1).cpu().tolist()
        return sparse_vector.float().cpu(), preds

    sparse_a, preds_a = _encode_and_forward(all_texts_a)
    sparse_b, preds_b = _encode_and_forward(all_texts_b)

    l2_distances = []
    cosine_sims = []
    agreements = []
    per_pair = []

    for i in range(len(pairs)):
        vec_a = sparse_a[i]
        vec_b = sparse_b[i]

        l2 = torch.norm(vec_a - vec_b, p=2).item()
        cos = torch.nn.functional.cosine_similarity(
            vec_a.unsqueeze(0), vec_b.unsqueeze(0),
        ).item()

        l2_distances.append(l2)
        cosine_sims.append(cos)
        agreements.append(int(preds_a[i] == preds_b[i]))

        per_pair.append({
            "sent_a": pairs[i][0],
            "sent_b": pairs[i][1],
            "l2": l2,
            "cosine": cos,
            "pred_a": preds_a[i],
            "pred_b": preds_b[i],
        })

    return {
        "mean_l2_distance": sum(l2_distances) / len(l2_distances) if l2_distances else 0.0,
        "mean_cosine_similarity": sum(cosine_sims) / len(cosine_sims) if cosine_sims else 0.0,
        "prediction_agreement_rate": sum(agreements) / len(agreements) if agreements else 0.0,
        "num_pairs": len(pairs),
        "per_pair": per_pair,
    }
