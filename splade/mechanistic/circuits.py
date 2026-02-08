from dataclasses import dataclass, field

import torch

from splade.mechanistic.attribution import compute_attribution_tensor
from splade.utils.cuda import COMPUTE_DTYPE, DEVICE, unwrap_compiled


@dataclass
class VocabularyCircuit:
    class_idx: int
    token_ids: list[int] = field(default_factory=list)
    token_names: list[str] = field(default_factory=list)
    attribution_scores: list[float] = field(default_factory=list)
    completeness_score: float = 0.0
    total_attribution: float = 0.0


def extract_vocabulary_circuit(
    model: torch.nn.Module,
    input_ids_list: list[torch.Tensor],
    attention_mask_list: list[torch.Tensor],
    tokenizer,
    class_idx: int,
    threshold: float = 0.01,
    precomputed_attributions: torch.Tensor | None = None,
) -> VocabularyCircuit:
    """Identify vocabulary tokens with consistent high attribution across examples.

    Tokens are included in the circuit if their mean absolute attribution exceeds
    the threshold fraction of total attribution mass.

    If precomputed_attributions is provided (e.g. from training centroid tracker),
    skip the per-sample forward+W_eff+DLA loop and use it directly.
    """
    _model = unwrap_compiled(model)

    if precomputed_attributions is not None:
        mean_attributions = precomputed_attributions
    else:
        vocab_size = _model.padded_vocab_size
        token_attribution_sums = torch.zeros(vocab_size, device=DEVICE)
        n_examples = 0

        for input_ids, attention_mask in zip(input_ids_list, attention_mask_list):
            with torch.inference_mode(), torch.amp.autocast("cuda", dtype=COMPUTE_DTYPE):
                _, sparse_vector, W_eff, _ = _model(input_ids, attention_mask)

            attr = compute_attribution_tensor(
                sparse_vector, W_eff,
                torch.tensor([class_idx], device=sparse_vector.device),
            )
            token_attribution_sums += attr[0].abs()
            n_examples += 1

        if n_examples == 0:
            return VocabularyCircuit(class_idx=class_idx)

        mean_attributions = token_attribution_sums / n_examples

    total_mass = mean_attributions.sum()
    if total_mass < 1e-12:
        return VocabularyCircuit(class_idx=class_idx)

    normalized = mean_attributions / total_mass
    circuit_mask = normalized >= threshold
    circuit_token_ids = torch.where(circuit_mask)[0]

    sorted_indices = torch.argsort(mean_attributions[circuit_token_ids], descending=True)
    circuit_token_ids = circuit_token_ids[sorted_indices].tolist()

    token_names = tokenizer.convert_ids_to_tokens(circuit_token_ids)
    scores = [float(mean_attributions[tid].item()) for tid in circuit_token_ids]

    return VocabularyCircuit(
        class_idx=class_idx,
        token_ids=circuit_token_ids,
        token_names=token_names,
        attribution_scores=scores,
        total_attribution=float(total_mass.item()),
    )


def measure_circuit_completeness(
    model: torch.nn.Module,
    input_ids_list: list[torch.Tensor],
    attention_mask_list: list[torch.Tensor],
    labels: list[int],
    circuit: VocabularyCircuit,
) -> float:
    """Measure accuracy retention when keeping only circuit tokens (ablating everything else).

    Returns the fraction of examples correctly classified using only circuit tokens.
    """
    _model = unwrap_compiled(model)
    vocab_size = _model.padded_vocab_size
    circuit_set = set(circuit.token_ids)
    non_circuit_ids = [i for i in range(vocab_size) if i not in circuit_set]

    correct = 0
    total = 0

    for input_ids, attention_mask, label in zip(input_ids_list, attention_mask_list, labels):
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=COMPUTE_DTYPE):
            _, original_sparse, _, _ = _model(input_ids, attention_mask)
            patched_sparse = original_sparse.clone()
            patched_sparse[:, non_circuit_ids] = 0.0
            patched_logits = _model.classifier_logits_only(patched_sparse)
        predicted = int(patched_logits.argmax(dim=-1).item())
        if predicted == label:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0


def visualize_circuit(circuit: VocabularyCircuit, max_tokens: int = 20) -> str:
    """Generate a text visualization of a vocabulary circuit."""
    lines = [
        f"Vocabulary Circuit for class {circuit.class_idx}",
        f"  Tokens: {len(circuit.token_ids)}",
        f"  Total attribution: {circuit.total_attribution:.4f}",
        f"  Completeness: {circuit.completeness_score:.4f}",
        "",
        f"  {'Token':<20} {'Attribution':>12}",
        f"  {'-'*20} {'-'*12}",
    ]
    for name, score in zip(
        circuit.token_names[:max_tokens],
        circuit.attribution_scores[:max_tokens],
    ):
        lines.append(f"  {name:<20} {score:>12.6f}")

    if len(circuit.token_ids) > max_tokens:
        lines.append(f"  ... and {len(circuit.token_ids) - max_tokens} more tokens")

    return "\n".join(lines)
