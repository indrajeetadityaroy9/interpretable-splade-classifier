from dataclasses import dataclass, field

import torch


@dataclass
class ActivationCache:
    activations: dict[str, torch.Tensor] = field(default_factory=dict)

    def __getitem__(self, key: str) -> torch.Tensor:
        return self.activations[key]

    def __contains__(self, key: str) -> bool:
        return key in self.activations

    def keys(self) -> list[str]:
        return list(self.activations.keys())


class HookManager:
    def __init__(self):
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        self._cache = ActivationCache()

    def register(self, module: torch.nn.Module, name: str) -> None:
        def hook_fn(_module, _input, output, _name=name):
            if isinstance(output, tuple):
                self._cache.activations[_name] = output[0].detach()
            else:
                self._cache.activations[_name] = output.detach()

        handle = module.register_forward_hook(hook_fn)
        self._hooks.append(handle)

    def clear(self) -> None:
        self._cache = ActivationCache()

    def remove_all(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self.clear()

    @property
    def cache(self) -> ActivationCache:
        return self._cache

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.remove_all()


def capture_activations(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Run forward pass with return_intermediates=True, returning (logits, sparse_vector, intermediates)."""
    with torch.inference_mode():
        result = model(input_ids, attention_mask, return_intermediates=True)
    return result[0], result[1], result[2]
