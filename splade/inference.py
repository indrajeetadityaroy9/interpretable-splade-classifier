import torch
from torch.utils.data import DataLoader, TensorDataset

from splade.evaluation.constants import SPECIAL_TOKENS
from splade.utils.cuda import COMPUTE_DTYPE, DEVICE, unwrap_compiled

def _run_inference_loop(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    extract_sparse: bool = False,
    batch_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    loader = DataLoader(
        TensorDataset(input_ids, attention_mask),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    all_logits = []
    all_sparse = [] if extract_sparse else None

    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=COMPUTE_DTYPE):
        for batch_ids, batch_mask in loader:
            batch_ids = batch_ids.to(DEVICE, non_blocking=True)
            batch_mask = batch_mask.to(DEVICE, non_blocking=True)
            logits, sparse = model(batch_ids, batch_mask)
            all_logits.append(logits)
            if extract_sparse:
                all_sparse.append(sparse)

    logits = torch.cat(all_logits, dim=0)
    sparse = torch.cat(all_sparse, dim=0) if extract_sparse else None
    return logits, sparse

def predict_model(model, tokenizer, texts: list[str], max_length: int, batch_size: int = 32, num_labels: int = 2) -> list[int]:
    encoding = tokenizer(texts, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
    logits, _ = _run_inference_loop(model, encoding["input_ids"], encoding["attention_mask"], batch_size=batch_size)
    if num_labels == 1:
        return (logits.squeeze(-1) > 0).int().tolist()
    return logits.argmax(dim=1).tolist()

def predict_proba_model(model, tokenizer, texts: list[str], max_length: int, batch_size: int = 32, num_labels: int = 2) -> list[list[float]]:
    encoding = tokenizer(texts, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
    logits, _ = _run_inference_loop(model, encoding["input_ids"], encoding["attention_mask"], batch_size=batch_size)
    if num_labels == 1:
        positive = torch.sigmoid(logits).squeeze(-1)
        probabilities = torch.stack([1 - positive, positive], dim=1)
    else:
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
    return probabilities.tolist()

def score_model(model, tokenizer, texts: list[str], labels: list[int], max_length: int, batch_size: int = 32, num_labels: int = 2) -> float:
    preds = predict_model(model, tokenizer, texts, max_length, batch_size, num_labels)
    return sum(p == t for p, t in zip(preds, labels)) / len(labels)


def explain_model(
    model: torch.nn.Module,
    tokenizer,
    text: str,
    max_length: int,
    top_k: int = 10,
    target_class: int | None = None,
    input_only: bool = False,
) -> list[tuple[str, float]]:
    encoding = tokenizer([text], max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
    logits, sparse = _run_inference_loop(
        model, encoding["input_ids"], encoding["attention_mask"],
        extract_sparse=True, batch_size=1,
    )
    sparse_vector = sparse[0]

    if target_class is None:
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        target_class = int(probabilities[0].argmax().item())

    with torch.inference_mode():
        _model = unwrap_compiled(model)
        W_eff, _ = _model.compute_effective_weights(sparse_vector.unsqueeze(0))
        weights = W_eff[0, target_class]

    contributions = sparse_vector * weights
    nonzero_indices = contributions.nonzero(as_tuple=True)[0]

    if input_only:
        input_ids = set(tokenizer.encode(text, add_special_tokens=False))
        mask = torch.tensor([i.item() in input_ids for i in nonzero_indices], dtype=torch.bool, device=DEVICE)
        nonzero_indices = nonzero_indices[mask]

    contrib_vals = contributions[nonzero_indices]
    positive_mask = contrib_vals > 0
    positive_indices = nonzero_indices[positive_mask]
    negative_indices = nonzero_indices[~positive_mask]

    pos_order = torch.argsort(contributions[positive_indices], descending=True)
    positive_indices = positive_indices[pos_order]
    neg_order = torch.argsort(contributions[negative_indices].abs(), descending=True)
    negative_indices = negative_indices[neg_order]

    ranked_indices = torch.cat([positive_indices, negative_indices])

    explanations = []
    for index in ranked_indices:
        idx_int = index.item()
        token = tokenizer.convert_ids_to_tokens(idx_int)
        if token not in SPECIAL_TOKENS:
            explanations.append((token, float(contributions[idx_int].item())))
        if len(explanations) >= top_k:
            break

    return explanations


def explain_model_batch(
    model: torch.nn.Module,
    tokenizer,
    texts: list[str],
    max_length: int,
    top_k: int = 10,
    input_only: bool = False,
    batch_size: int = 32,
) -> list[list[tuple[str, float]]]:
    encoding = tokenizer(texts, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
    logits, sparse = _run_inference_loop(
        model, encoding["input_ids"], encoding["attention_mask"],
        extract_sparse=True, batch_size=batch_size,
    )
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    target_classes = probabilities.argmax(dim=-1).tolist()

    with torch.inference_mode():
        _model = unwrap_compiled(model)
        W_eff, _ = _model.compute_effective_weights(sparse)

    input_id_sets = None
    if input_only:
        input_id_sets = [
            set(tokenizer.encode(text, add_special_tokens=False)) for text in texts
        ]

    all_explanations = []
    for idx in range(len(texts)):
        target_class = target_classes[idx]
        weights = W_eff[idx, target_class]
        contributions = sparse[idx] * weights
        nonzero_indices = contributions.nonzero(as_tuple=True)[0]

        if input_only and input_id_sets is not None:
            mask = torch.tensor(
                [i.item() in input_id_sets[idx] for i in nonzero_indices],
                dtype=torch.bool, device=DEVICE,
            )
            nonzero_indices = nonzero_indices[mask]

        if len(nonzero_indices) == 0:
            all_explanations.append([])
            continue

        contrib_vals = contributions[nonzero_indices]
        positive_mask = contrib_vals > 0
        positive_indices = nonzero_indices[positive_mask]
        negative_indices = nonzero_indices[~positive_mask]

        pos_order = torch.argsort(contributions[positive_indices], descending=True)
        positive_indices = positive_indices[pos_order]
        neg_order = torch.argsort(contributions[negative_indices].abs(), descending=True)
        negative_indices = negative_indices[neg_order]

        ranked_indices = torch.cat([positive_indices, negative_indices])

        explanations = []
        for index in ranked_indices:
            idx_int = index.item()
            token = tokenizer.convert_ids_to_tokens(idx_int)
            if token not in SPECIAL_TOKENS:
                explanations.append((token, float(contributions[idx_int].item())))
            if len(explanations) >= top_k:
                break
        all_explanations.append(explanations)

    return all_explanations
