import torch
from torch.utils.data import DataLoader, TensorDataset

from splade.utils.cuda import COMPUTE_DTYPE, DEVICE, unwrap_compiled


def _run_inference_loop(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    extract_sparse: bool = False,
    extract_weff: bool = False,
    batch_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    loader = DataLoader(
        TensorDataset(input_ids, attention_mask),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    all_logits = []
    all_sparse = [] if extract_sparse else None
    all_weff = [] if extract_weff else None
    _orig = unwrap_compiled(model)

    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=COMPUTE_DTYPE):
        for batch_ids, batch_mask in loader:
            batch_ids = batch_ids.to(DEVICE, non_blocking=True)
            batch_mask = batch_mask.to(DEVICE, non_blocking=True)
            sparse_seq, *_ = model(batch_ids, batch_mask)
            logits, sparse, w_eff, _ = _orig.classify(sparse_seq, batch_mask)
            all_logits.append(logits.clone())
            if extract_sparse:
                all_sparse.append(sparse.clone())
            if extract_weff:
                all_weff.append(w_eff.clone())

    logits = torch.cat(all_logits, dim=0)
    sparse = torch.cat(all_sparse, dim=0) if extract_sparse else None
    weff = torch.cat(all_weff, dim=0) if extract_weff else None
    return logits, sparse, weff

def _predict_model(model, tokenizer, texts: list[str], max_length: int, batch_size: int = 64, num_labels: int = 2) -> list[int]:
    encoding = tokenizer(texts, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
    logits, _, _ = _run_inference_loop(model, encoding["input_ids"], encoding["attention_mask"], batch_size=batch_size)
    if num_labels == 1:
        return (logits.squeeze(-1) > 0).int().tolist()
    return logits.argmax(dim=1).tolist()

def score_model(model, tokenizer, texts: list[str], labels: list[int], max_length: int, batch_size: int = 64, num_labels: int = 2) -> float:
    preds = _predict_model(model, tokenizer, texts, max_length, batch_size, num_labels)
    return sum(p == t for p, t in zip(preds, labels)) / len(labels)
