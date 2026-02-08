from dataclasses import dataclass

import numpy
import torch


@dataclass
class VocabularyAttribution:
    token_ids: numpy.ndarray
    token_names: list[str]
    attribution_scores: numpy.ndarray
    class_idx: int
    logit: float


def compute_direct_logit_attribution(
    sparse_vector: torch.Tensor | numpy.ndarray,
    classifier_weight: torch.Tensor | numpy.ndarray,
    tokenizer,
    class_idx: int,
) -> VocabularyAttribution:
    """Compute Direct Logit Attribution: element-wise sparse_vector * classifier_weight[class_idx].

    The sum of attribution scores exactly equals the logit for the target class
    (verifiable invariant, ignoring classifier bias).
    """
    if isinstance(sparse_vector, torch.Tensor):
        sparse_vector = sparse_vector.detach().cpu().numpy()
    if isinstance(classifier_weight, torch.Tensor):
        classifier_weight = classifier_weight.detach().cpu().numpy()

    if classifier_weight.ndim == 2:
        weights = classifier_weight[class_idx]
    else:
        weights = classifier_weight

    attributions = sparse_vector * weights
    nonzero_mask = attributions != 0
    token_ids = numpy.where(nonzero_mask)[0]

    sorted_indices = numpy.argsort(numpy.abs(attributions[token_ids]))[::-1]
    token_ids = token_ids[sorted_indices]
    scores = attributions[token_ids]

    token_names = tokenizer.convert_ids_to_tokens(token_ids.tolist())

    return VocabularyAttribution(
        token_ids=token_ids,
        token_names=token_names,
        attribution_scores=scores,
        class_idx=class_idx,
        logit=float(attributions.sum()),
    )


def compute_contrastive_attribution(
    sparse_vector: torch.Tensor | numpy.ndarray,
    classifier_weight: torch.Tensor | numpy.ndarray,
    tokenizer,
    class_a: int,
    class_b: int,
) -> VocabularyAttribution:
    """Compute contrastive attribution: difference between class_a and class_b attributions.

    Positive scores favor class_a, negative scores favor class_b.
    """
    if isinstance(sparse_vector, torch.Tensor):
        sparse_vector = sparse_vector.detach().cpu().numpy()
    if isinstance(classifier_weight, torch.Tensor):
        classifier_weight = classifier_weight.detach().cpu().numpy()

    diff_weights = classifier_weight[class_a] - classifier_weight[class_b]
    attributions = sparse_vector * diff_weights

    nonzero_mask = attributions != 0
    token_ids = numpy.where(nonzero_mask)[0]

    sorted_indices = numpy.argsort(numpy.abs(attributions[token_ids]))[::-1]
    token_ids = token_ids[sorted_indices]
    scores = attributions[token_ids]

    token_names = tokenizer.convert_ids_to_tokens(token_ids.tolist())

    return VocabularyAttribution(
        token_ids=token_ids,
        token_names=token_names,
        attribution_scores=scores,
        class_idx=class_a,
        logit=float(attributions.sum()),
    )
