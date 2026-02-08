from dataclasses import dataclass

import torch

from splade.utils.cuda import DEVICE


@dataclass
class VocabularyAttribution:
    token_ids: list[int]
    token_names: list[str]
    attribution_scores: list[float]
    class_idx: int
    logit: float


def compute_direct_logit_attribution(
    sparse_vector: torch.Tensor,
    classifier_weight: torch.Tensor,
    tokenizer,
    class_idx: int,
) -> VocabularyAttribution:
    """Compute Direct Logit Attribution: element-wise sparse_vector * classifier_weight[class_idx].

    The sum of attribution scores exactly equals the logit for the target class
    (verifiable invariant, ignoring classifier bias).

    All computation stays on GPU. Results are converted to Python types at the boundary.
    """
    sparse_vector = sparse_vector.to(DEVICE)
    classifier_weight = classifier_weight.to(DEVICE)

    if classifier_weight.ndim == 2:
        weights = classifier_weight[class_idx]
    else:
        weights = classifier_weight

    attributions = sparse_vector * weights
    nonzero_mask = attributions != 0
    token_ids = torch.where(nonzero_mask)[0]

    sorted_indices = torch.argsort(attributions[token_ids].abs(), descending=True)
    token_ids = token_ids[sorted_indices]
    scores = attributions[token_ids]

    token_ids_list = token_ids.tolist()
    token_names = tokenizer.convert_ids_to_tokens(token_ids_list)

    return VocabularyAttribution(
        token_ids=token_ids_list,
        token_names=token_names,
        attribution_scores=scores.tolist(),
        class_idx=class_idx,
        logit=float(attributions.sum().item()),
    )


