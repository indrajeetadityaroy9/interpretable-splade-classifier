from collections.abc import Iterator
from contextlib import contextmanager
from itertools import chain

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Target dataset — The Pile (uncopyrighted), train stream, flat "text" schema.
# Hard-coded because SPALF is deliberately specialized to this corpus; see
# README for justification of the single-dataset regime.
DATASET_NAME: str = "monology/pile-uncopyrighted"
DATASET_SPLIT: str = "train"
TEXT_COLUMN: str = "text"


class ActivationStore:
    def __init__(self, model_name: str, batch_size: int,
                 seq_len: int = 128, seed: int = 42) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.seed = seed

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa",
        )
        self.model.eval()

        # Midpoint transformer layer (auto-derived from layer-name convention).
        n = self.model.config.num_hidden_layers
        mid_name = next(name for name, _ in self.model.named_modules()
                        if name.split(".")[-1] == str(n // 2))
        self.hook_point = mid_name
        self.mid_module = dict(self.model.named_modules())[mid_name]

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self._token_iter: Iterator[torch.Tensor] = self._token_generator()

    @contextmanager
    def capture(self):
        """Context-managed hook that records the mid-layer output in `captured[0]`."""
        captured: list[torch.Tensor] = []
        handle = self.mid_module.register_forward_hook(
            lambda _m, _i, out: captured.append(out[0].detach())
        )
        try:
            yield captured
        finally:
            handle.remove()

    @contextmanager
    def patch(self, replacement: torch.Tensor):
        """Context-managed hook that replaces the mid-layer output with `replacement`."""
        handle = self.mid_module.register_forward_hook(
            lambda _m, _i, out: (replacement,) + out[1:]
        )
        try:
            yield
        finally:
            handle.remove()

    def _token_generator(self, batch_size: int | None = None) -> Iterator[torch.Tensor]:
        """Yield [batch_size, seq_len] token batches from the Pile streaming split.

        add_special_tokens=False because Pile docs are concatenated; no inter-doc BOS/EOS.
        """
        bs = batch_size or self.batch_size
        seq_len = self.seq_len

        def group(examples: dict) -> dict:
            concat = list(chain.from_iterable(examples["input_ids"]))
            total = (len(concat) // seq_len) * seq_len
            return {"input_ids": [concat[i:i + seq_len] for i in range(0, total, seq_len)]}

        epoch = 0
        while True:
            ds = (load_dataset(DATASET_NAME, split=DATASET_SPLIT, streaming=True)
                  .select_columns([TEXT_COLUMN])
                  .shuffle(seed=self.seed, buffer_size=self.batch_size * seq_len))
            ds.set_epoch(epoch)
            tokenized = ds.map(
                lambda ex: self.tokenizer(ex[TEXT_COLUMN], add_special_tokens=False),
                batched=True, remove_columns=[TEXT_COLUMN],
            )
            chunked = tokenized.map(group, batched=True, batch_size=bs * 4)
            for batch in chunked.iter(batch_size=bs, drop_last_batch=True):
                yield torch.tensor(batch["input_ids"], dtype=torch.long)
            epoch += 1

    @torch.no_grad()
    def next_batch(self) -> torch.Tensor:
        """Run model forward and return flattened mid-layer activations [N, d]."""
        with self.capture() as cap:
            self.model(next(self._token_iter).cuda())
        acts = cap[0]
        return acts.reshape(-1, acts.shape[-1]).float()

    def get_unembedding_matrix(self) -> torch.Tensor:
        return self.model.get_output_embeddings().weight.T.float()
