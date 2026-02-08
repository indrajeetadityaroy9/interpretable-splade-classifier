from typing import Callable

import torch

from splade.utils.cuda import COMPUTE_DTYPE, unwrap_compiled


def patch_sparse_vector(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    token_indices: list[int],
    patch_value: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Zero-ablate specific vocabulary dimensions in the sparse vector.

    Returns (original_logits, patched_logits, delta).
    """
    _model = unwrap_compiled(model)

    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=COMPUTE_DTYPE):
        original_logits, original_sparse = _model(input_ids, attention_mask)

        patched_sparse = original_sparse.clone()
        for idx in token_indices:
            patched_sparse[:, idx] = patch_value

        patched_logits = _model.classifier_forward(patched_sparse)

    delta = original_logits - patched_logits
    return original_logits, patched_logits, delta


def ablate_vocabulary_tokens(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    token_ids: list[int],
) -> dict[int, float]:
    """Measure the causal effect of ablating each vocabulary token individually.

    Returns a dict mapping token_id -> logit delta for the predicted class.
    """
    _model = unwrap_compiled(model)

    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=COMPUTE_DTYPE):
        original_logits, sparse_vector = _model(input_ids, attention_mask)
        predicted_class = int(original_logits.argmax(dim=-1).item())

    effects = {}
    for token_id in token_ids:
        _, _, delta = patch_sparse_vector(model, input_ids, attention_mask, [token_id])
        effects[token_id] = float(delta[0, predicted_class].item())

    return effects


def patch_activation(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_name: str,
    patch_fn: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Apply a patch function to a named layer's output via forward hook.

    Returns the logits after patching.
    """
    _model = unwrap_compiled(model)

    target_module = None
    for name, module in _model.named_modules():
        if name == layer_name:
            target_module = module
            break
    if target_module is None:
        raise ValueError(f"Layer '{layer_name}' not found in model")

    def hook_fn(_module, _input, output):
        if isinstance(output, tuple):
            return (patch_fn(output[0]),) + output[1:]
        return patch_fn(output)

    handle = target_module.register_forward_hook(hook_fn)
    try:
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=COMPUTE_DTYPE):
            logits, _ = _model(input_ids, attention_mask)
    finally:
        handle.remove()

    return logits
