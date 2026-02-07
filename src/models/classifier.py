"""SPLADE classifier with sparse vocabulary features."""

import numpy
import torch
import torch.nn
import torch.nn.functional
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer, DistilBertForMaskedLM

from src.models.components import splade_aggregate
from src.training.losses import DFFlopsRegFunction, DocumentFrequencyTracker
from src.training.optim import (
    _adaptive_gradient_clip,
    _AdaptiveLambdaSchedule,
    _compute_base_lr,
    _compute_warmup_steps,
    _EarlyStopping,
    _LRScheduler,
)
from src.utils.cuda import AUTOCAST_DEVICE_TYPE, AUTOCAST_ENABLED, COMPUTE_DTYPE, DEVICE

SPECIAL_TOKENS = {"[CLS]", "[SEP]", "[UNK]", "[MASK]", "[PAD]"}


class _SpladeEncoder(torch.nn.Module):
    """Encoder and classifier head for SPLADE features."""

    def __init__(self, model_name: str, num_labels: int):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name)
        self.vocab_size = config.vocab_size

        self.bert = AutoModel.from_pretrained(model_name, attn_implementation="sdpa")
        self.vocab_transform = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.vocab_layer_norm = torch.nn.LayerNorm(config.hidden_size)
        self.vocab_projector = torch.nn.Linear(config.hidden_size, self.vocab_size)

        masked_language_model = DistilBertForMaskedLM.from_pretrained(model_name)
        self.vocab_transform.load_state_dict(masked_language_model.vocab_transform.state_dict())
        self.vocab_layer_norm.load_state_dict(masked_language_model.vocab_layer_norm.state_dict())
        self.vocab_projector.load_state_dict(masked_language_model.vocab_projector.state_dict())
        del masked_language_model

        self.classifier = torch.nn.Linear(self.vocab_size, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        transformed = self.vocab_layer_norm(torch.nn.functional.gelu(self.vocab_transform(hidden)))
        mlm_logits = self.vocab_projector(transformed)

        sparse_vector = splade_aggregate(mlm_logits, attention_mask)
        return self.classifier(sparse_vector), sparse_vector


class SPLADEClassifier:
    """Sklearn-style classifier using SPLADE activations."""

    def __init__(
        self,
        num_labels: int = 2,
        model_name: str = "distilbert-base-uncased",
        batch_size: int = 32,
        max_length: int = 128,
        df_alpha: float = 0.1,
        df_beta: float = 5.0,
    ):
        self.num_labels = num_labels
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.df_alpha = df_alpha
        self.df_beta = df_beta
        self.base_lr = _compute_base_lr(model_name)

        self.model = _SpladeEncoder(model_name, num_labels).to(DEVICE)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")

    @staticmethod
    def _validate_texts(texts: list[str], name: str = "X") -> None:
        if not texts:
            raise ValueError(f"{name} must be non-empty")
        if not all(isinstance(text, str) for text in texts):
            raise TypeError(f"{name} must contain strings")

    def _tokenize(self, texts: list[str]) -> dict[str, torch.Tensor]:
        return self.tokenizer(
            texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    def _run_inference_loop(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        extract_sparse: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        loader = DataLoader(
            TensorDataset(input_ids, attention_mask),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=DEVICE.type == "cuda",
        )
        all_logits = []
        all_sparse = [] if extract_sparse else None

        with torch.inference_mode():
            for batch_ids, batch_mask in loader:
                batch_ids = batch_ids.to(DEVICE, non_blocking=True)
                batch_mask = batch_mask.to(DEVICE, non_blocking=True)
                logits, sparse = self.model(batch_ids, batch_mask)
                all_logits.append(logits.cpu())
                if extract_sparse:
                    all_sparse.append(sparse.cpu())

        logits = torch.cat(all_logits, dim=0)
        sparse = torch.cat(all_sparse, dim=0) if extract_sparse else None
        return logits, sparse

    def fit(
        self,
        texts: list[str],
        labels,
        epochs: int | None = None,
        max_epochs: int = 20,
    ) -> "SPLADEClassifier":
        self._validate_texts(texts)
        if len(texts) != len(labels):
            raise ValueError("X and y must have the same length")

        epoch_count = max_epochs if epochs is None else epochs
        steps_per_epoch = -(-len(texts) // self.batch_size)
        total_steps = steps_per_epoch * epoch_count
        warmup_steps = _compute_warmup_steps(total_steps)

        lr_scheduler = _LRScheduler(self.base_lr, total_steps, warmup_steps)
        lambda_schedule = _AdaptiveLambdaSchedule(warmup_steps=warmup_steps)
        df_tracker = DocumentFrequencyTracker(vocab_size=self.model.vocab_size, device=DEVICE)
        early_stopping = _EarlyStopping() if epochs is None else None

        label_tensor = torch.tensor(
            labels,
            dtype=torch.float32 if self.num_labels == 1 else torch.long,
        )
        encoding = self._tokenize(texts)
        loader = DataLoader(
            TensorDataset(encoding["input_ids"], encoding["attention_mask"], label_tensor),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=DEVICE.type == "cuda",
            prefetch_factor=2,
            persistent_workers=True,
        )
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.base_lr)
        criterion = (
            torch.nn.BCEWithLogitsLoss()
            if self.num_labels == 1
            else torch.nn.CrossEntropyLoss()
        )

        self.model.train()
        for epoch_index in range(epoch_count):
            total_loss = 0.0
            batch_count = 0
            for batch_ids, batch_mask, batch_labels in tqdm(loader, desc=f"Epoch {epoch_index + 1}/{epoch_count}"):
                batch_ids = batch_ids.to(DEVICE, non_blocking=True)
                batch_mask = batch_mask.to(DEVICE, non_blocking=True)
                batch_labels = batch_labels.to(DEVICE, non_blocking=True)

                learning_rate = lr_scheduler.step()
                for parameter_group in optimizer.param_groups:
                    parameter_group["lr"] = learning_rate

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(AUTOCAST_DEVICE_TYPE, dtype=COMPUTE_DTYPE, enabled=AUTOCAST_ENABLED):
                    logits, sparse = self.model(batch_ids, batch_mask)
                    classification_loss = (
                        criterion(logits.squeeze(-1), batch_labels)
                        if self.num_labels == 1
                        else criterion(logits, batch_labels.view(-1))
                    )
                    df_tracker.update(sparse)
                    df_weights = df_tracker.get_weights(alpha=self.df_alpha, beta=self.df_beta)
                    regularization_loss = DFFlopsRegFunction.apply(sparse, df_weights)
                    regularization_weight = lambda_schedule.compute_lambda(
                        sparse,
                        classification_loss.item(),
                        regularization_loss.item(),
                    )
                    loss = classification_loss + regularization_weight * regularization_loss

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                _adaptive_gradient_clip(self.model)
                self.scaler.step(optimizer)
                self.scaler.update()

                total_loss += loss.item()
                batch_count += 1

            average_loss = total_loss / batch_count
            sparsity = lambda_schedule._current_sparsity
            stats = df_tracker.get_stats()
            print(
                f"Epoch {epoch_index + 1}: Loss = {average_loss:.4f}, "
                f"Sparsity: {sparsity:.2%}, Top-1 DF: {stats['top1_df_pct']:.1f}%"
            )

            if early_stopping is not None and early_stopping.step(average_loss, self.model):
                self.model.load_state_dict(early_stopping.best_state)
                print(f"Early stopping at epoch {epoch_index + 1}")
                break

        self.model.eval()
        return self

    def predict(self, texts: list[str]) -> list[int]:
        self._validate_texts(texts)
        encoding = self._tokenize(texts)
        logits, _ = self._run_inference_loop(encoding["input_ids"], encoding["attention_mask"])
        if self.num_labels == 1:
            return (logits.squeeze(-1) > 0).int().tolist()
        return logits.argmax(dim=1).tolist()

    def predict_proba(self, texts: list[str]) -> list[list[float]]:
        self._validate_texts(texts)
        encoding = self._tokenize(texts)
        logits, _ = self._run_inference_loop(encoding["input_ids"], encoding["attention_mask"])
        if self.num_labels == 1:
            positive = torch.sigmoid(logits).squeeze(-1)
            probabilities = torch.stack([1 - positive, positive], dim=1)
        else:
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
        return probabilities.tolist()

    def score(self, texts: list[str], labels) -> float:
        self._validate_texts(texts)
        if not labels:
            raise ValueError("y must be non-empty")
        return sum(pred == target for pred, target in zip(self.predict(texts), labels)) / len(labels)

    def transform(self, texts: list[str]) -> numpy.ndarray:
        """Return SPLADE sparse vectors with shape [n_samples, vocab_size]."""
        self._validate_texts(texts)
        encoding = self._tokenize(texts)
        _, sparse = self._run_inference_loop(
            encoding["input_ids"],
            encoding["attention_mask"],
            extract_sparse=True,
        )
        return sparse.numpy()

    def explain(
        self,
        text: str,
        top_k: int = 10,
        target_class: int | None = None,
    ) -> list[tuple[str, float]]:
        """Return top-k token contributions for a target class."""
        if not text or not text.strip():
            raise ValueError("text must be non-empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        sparse_vector = self.transform([text])[0]

        if target_class is None:
            probabilities = self.predict_proba([text])[0]
            target_class = int(numpy.argmax(probabilities))

        with torch.inference_mode():
            weights = self.model.classifier.weight[target_class].cpu().numpy()

        contributions = sparse_vector * weights
        nonzero_indices = numpy.nonzero(contributions)[0]
        ranked_indices = nonzero_indices[numpy.argsort(numpy.abs(contributions[nonzero_indices]))[::-1]]

        explanations = []
        for index in ranked_indices:
            token = self.tokenizer.convert_ids_to_tokens(int(index))
            if token not in SPECIAL_TOKENS:
                explanations.append((token, float(contributions[index])))
            if len(explanations) >= top_k:
                break

        return explanations
