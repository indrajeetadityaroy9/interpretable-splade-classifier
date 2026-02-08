"""Shared training pipeline and predictor interface.

Consolidates the ~30-line training setup duplicated across entry scripts
into a single function, and provides PredictorWrapper for evaluation.
"""

from dataclasses import dataclass

import torch
from transformers import AutoTokenizer

from splade.config.schema import Config
from splade.training.circuit_losses import AttributionCentroidTracker
from splade.data.loader import infer_max_length, load_dataset_by_name
from splade.inference import score_model
from splade.models.splade import SpladeModel
from splade.training.loop import train_model
from splade.training.optim import _infer_batch_size
from splade.utils.cuda import COMPUTE_DTYPE, DEVICE, set_seed, unwrap_compiled


@dataclass
class TrainedExperiment:
    model: torch.nn.Module
    tokenizer: AutoTokenizer
    train_texts: list[str]
    train_labels: list[int]
    test_texts: list[str]
    test_labels: list[int]
    num_labels: int
    max_length: int
    batch_size: int
    accuracy: float
    seed: int
    centroid_tracker: AttributionCentroidTracker | None = None


def setup_and_train(config: Config, seed: int) -> TrainedExperiment:
    """Load data, create model, train with CIS, and evaluate accuracy."""
    set_seed(seed)

    train_texts, train_labels, test_texts, test_labels, num_labels = load_dataset_by_name(
        config.data.dataset_name,
        config.data.train_samples,
        config.data.test_samples,
        seed=seed,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model.name)
    max_length = infer_max_length(train_texts, tokenizer)
    batch_size = _infer_batch_size(config.model.name, max_length)

    val_size = min(200, len(train_texts) // 5)
    val_texts = train_texts[-val_size:]
    val_labels = train_labels[-val_size:]
    train_texts_split = train_texts[:-val_size]
    train_labels_split = train_labels[:-val_size]

    model = SpladeModel(config.model.name, num_labels).to(DEVICE)
    model = torch.compile(model, mode="max-autotune")

    centroid_tracker = train_model(
        model, tokenizer, train_texts_split, train_labels_split,
        model_name=config.model.name, num_labels=num_labels,
        val_texts=val_texts, val_labels=val_labels,
        max_length=max_length, batch_size=batch_size,
    )

    accuracy = score_model(
        model, tokenizer, test_texts, test_labels,
        max_length, batch_size, num_labels,
    )

    return TrainedExperiment(
        model=model,
        tokenizer=tokenizer,
        train_texts=train_texts,
        train_labels=train_labels,
        test_texts=test_texts,
        test_labels=test_labels,
        num_labels=num_labels,
        max_length=max_length,
        batch_size=batch_size,
        accuracy=accuracy,
        seed=seed,
        centroid_tracker=centroid_tracker,
    )


class PredictorWrapper:
    """Unified predictor interface for evaluation metrics."""

    def __init__(self, model, tokenizer, max_length, batch_size):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size

    def predict_proba(self, texts):
        from splade.inference import predict_proba_model
        return predict_proba_model(self.model, self.tokenizer, texts, self.max_length, self.batch_size)

    def get_embeddings(self, texts):
        encoding = self.tokenizer(
            texts, max_length=self.max_length,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(DEVICE)
        attention_mask = encoding["attention_mask"].to(DEVICE)
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=COMPUTE_DTYPE):
            embeddings = unwrap_compiled(self.model).get_embeddings(input_ids)
        return embeddings, attention_mask

    def predict_proba_from_embeddings(self, embeddings, attention_mask):
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=COMPUTE_DTYPE):
            logits, _, _, _ = unwrap_compiled(self.model).forward_from_embeddings(embeddings, attention_mask)
        return torch.nn.functional.softmax(logits, dim=-1)


def prepare_mechanistic_inputs(
    tokenizer,
    texts: list[str],
    max_length: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Tokenize texts into per-sample tensor lists for mechanistic evaluation."""
    encoding = tokenizer(
        texts, max_length=max_length, padding="max_length",
        truncation=True, return_tensors="pt",
    )
    input_ids_list = [
        encoding["input_ids"][i:i+1].to(DEVICE) for i in range(len(texts))
    ]
    attention_mask_list = [
        encoding["attention_mask"][i:i+1].to(DEVICE) for i in range(len(texts))
    ]
    return input_ids_list, attention_mask_list
