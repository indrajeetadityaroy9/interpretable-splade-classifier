"""
SPLADE models for sparse text classification.

Public API:
- SPLADEClassifier: sklearn-compatible classifier with fit/predict/transform API

Internal (not for direct use):
- _SpladeModule: PyTorch nn.Module used internally by SPLADEClassifier
"""

import os
from typing import List, Tuple, Optional, Dict, Any, Set

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

from src.ops import splade_aggregate, TRITON_AVAILABLE
from src.utils import get_device, set_seed


class _SpladeModule(nn.Module):
    """
    Internal PyTorch module for SPLADE. Use SPLADEClassifier instead.

    Architecture:
        Text -> DistilBERT -> MLM Logits -> log(1 + ReLU(x)) -> Max-pool -> Linear

    Returns both classification logits and sparse document vectors.
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        max_length: int = 128,
        num_labels: int = 1,  # Binary classification
        load_mlm_weights: bool = True,  # Load pretrained MLM head for interpretability
    ):
        super().__init__()

        self.model_name = model_name
        self.max_length = max_length
        self.vocab_size = 30522  # DistilBERT vocab size
        self.mlm_pretrained = load_mlm_weights  # Track for save/load

        self.bert = AutoModel.from_pretrained(model_name)

        # MLM projection layers (DistilBERT hidden size = 768)
        self.vocab_transform = nn.Linear(768, 768)
        self.vocab_layer_norm = nn.LayerNorm(768)
        self.vocab_projector = nn.Linear(768, self.vocab_size)

        # Initialize MLM head from pretrained weights for interpretable explanations
        if load_mlm_weights:
            self._load_pretrained_mlm_head(model_name)

        # Classification head (on top of sparse vectors)
        self.classifier = nn.Linear(self.vocab_size, num_labels)

    def _load_pretrained_mlm_head(self, model_name: str):
        """
        Load pretrained MLM head weights from DistilBertForMaskedLM.

        This is critical for interpretability - the pretrained MLM head
        maps hidden states to vocabulary tokens in a semantically meaningful way.
        Without this, the vocabulary projections are random and explanations
        will show nonsensical terms.
        """
        from transformers import DistilBertForMaskedLM

        mlm_model = DistilBertForMaskedLM.from_pretrained(model_name)

        self.vocab_transform.load_state_dict(mlm_model.vocab_transform.state_dict())
        self.vocab_layer_norm.load_state_dict(mlm_model.vocab_layer_norm.state_dict())
        self.vocab_projector.load_state_dict(mlm_model.vocab_projector.state_dict())

        del mlm_model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass producing logits and sparse vectors.

        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]

        Returns:
            logits: Classification logits [batch, 1]
            sparse_vec: Sparse document vectors [batch, vocab_size]
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # [batch, seq_len, 768]

        transformed = self.vocab_transform(hidden_states)
        transformed = F.gelu(transformed)
        transformed = self.vocab_layer_norm(transformed)
        mlm_logits = self.vocab_projector(transformed)  # [batch, seq_len, vocab_size]

        # SPLADE aggregation: log(1 + ReLU(x)) + max-pool
        # Use Triton if available and in inference mode
        use_triton = (TRITON_AVAILABLE and
                     mlm_logits.is_cuda and
                     not self.training)

        if use_triton:
            sparse_vec = splade_aggregate(mlm_logits, attention_mask, backend="triton")
        else:
            activated = torch.log1p(F.relu(mlm_logits))
            mask_expanded = attention_mask.unsqueeze(-1).float()
            activated = activated * mask_expanded
            sparse_vec = activated.max(dim=1).values  # [batch, vocab_size]

        logits = self.classifier(sparse_vec)  # [batch, 1]

        return logits, sparse_vec


class SPLADEClassifier:
    """
    sklearn-compatible wrapper around DistilBERTSparseClassifier.

    Provides familiar fit/predict/transform interface for easy integration
    with existing ML pipelines.

    Supports both binary and multi-class classification:
    - Binary (num_labels=1): Uses BCEWithLogitsLoss, sigmoid probabilities
    - Multi-class (num_labels>1): Uses CrossEntropyLoss, softmax probabilities

    Example:
        # Binary classification
        clf = SPLADEClassifier()
        clf.fit(train_texts, train_labels, epochs=5)

        # Multi-class classification (e.g., 4 classes)
        clf = SPLADEClassifier(num_labels=4, class_names=["World", "Sports", "Business", "Tech"])
        clf.fit(train_texts, train_labels, epochs=5)
        predictions = clf.predict(test_texts)
        clf.print_explanation("Breaking news about the stock market...")
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        max_length: int = 128,
        batch_size: int = 32,
        learning_rate: float = 2e-5,
        flops_lambda: float = 1e-4,
        random_state: int = 42,
        verbose: bool = True,
        num_labels: int = 1,
        class_names: Optional[List[str]] = None,
        load_mlm_weights: bool = True,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.flops_lambda = flops_lambda
        self.random_state = random_state
        self.verbose = verbose
        self.num_labels = num_labels
        self.class_names = class_names
        self.load_mlm_weights = load_mlm_weights

        set_seed(random_state)
        self.device = get_device()
        self.model = _SpladeModule(
            model_name, max_length,
            num_labels=num_labels,
            load_mlm_weights=load_mlm_weights,
        )
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def _tokenize(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Tokenize list of texts."""
        encoding = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return encoding

    def _batch_inference(
        self,
        X: List[str],
        return_vectors: bool = False,
    ) -> Dict[str, Any]:
        """
        Core batch inference logic - tokenize, batch, forward pass.

        Consolidates common pattern used by predict_proba() and transform().

        Args:
            X: List of text strings
            return_vectors: Whether to collect sparse vectors

        Returns:
            Dict with 'probs' (always), and optionally 'vectors'
        """
        self.model.eval()
        results = {'probs': [], 'vectors': []}

        encoding = self._tokenize(X)
        dataset = TensorDataset(encoding["input_ids"], encoding["attention_mask"])
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        with torch.no_grad():
            for input_ids, attention_mask in loader:
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)

                logits, sparse_vec = self.model(input_ids, attention_mask)

                # Compute probabilities
                if self.num_labels == 1:
                    probs = torch.sigmoid(logits).squeeze(-1)
                    results['probs'].extend(probs.cpu().tolist())
                else:
                    probs = F.softmax(logits, dim=-1)
                    results['probs'].extend(probs.cpu().tolist())

                # Collect vectors if requested
                if return_vectors:
                    results['vectors'].append(sparse_vec.cpu())

        # Concatenate vectors if collected
        if results['vectors']:
            results['vectors'] = torch.cat(results['vectors'], dim=0)

        return results

    def fit(
        self,
        X: List[str],
        y,
        epochs: int = 5,
    ) -> "SPLADEClassifier":
        """
        Train the classifier on texts and labels.

        Args:
            X: List of text strings
            y: Labels (array-like, integers for multi-class)
            epochs: Number of training epochs

        Returns:
            self
        """
        from src.ops import flops_reg as flops_regularization

        # Convert labels to tensor
        if self.num_labels == 1:
            # Binary classification
            y_tensor = torch.tensor(y, dtype=torch.float32)
        else:
            # Multi-class classification
            y_tensor = torch.tensor(y, dtype=torch.long)

        # Create dataset
        encoding = self._tokenize(X)
        dataset = TensorDataset(
            encoding["input_ids"],
            encoding["attention_mask"],
            y_tensor
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Optimizer and loss
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        if self.num_labels == 1:
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()

        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0

            iterator = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}") if self.verbose else loader
            for input_ids, attention_mask, labels in iterator:
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()

                logits, sparse_vec = self.model(input_ids, attention_mask)

                # Combined loss
                if self.num_labels == 1:
                    labels = labels.unsqueeze(1)
                    cls_loss = criterion(logits, labels)
                else:
                    # logits: [batch, num_labels], labels: [batch]
                    cls_loss = criterion(logits, labels)

                flops_loss = self.flops_lambda * flops_regularization(sparse_vec)
                loss = cls_loss + flops_loss

                # Invariant: loss should be finite
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"Loss became non-finite: {loss.item()}. "
                        f"cls_loss={cls_loss.item()}, flops_loss={flops_loss.item()}"
                    )

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            if self.verbose:
                avg_loss = total_loss / num_batches
                print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")

        return self

    def predict(self, X: List[str]) -> List[int]:
        """
        Predict class labels for texts.

        Args:
            X: List of text strings

        Returns:
            List of predicted labels (0 or 1 for binary, 0 to num_labels-1 for multi-class)
        """
        if self.num_labels == 1:
            # Binary classification
            probs = self.predict_proba(X)
            return [1 if p > 0.5 else 0 for p in probs]
        else:
            # Multi-class classification
            probs = self.predict_proba(X)
            return [max(range(len(p)), key=lambda i: p[i]) for p in probs]

    def predict_proba(self, X: List[str]):
        """
        Predict class probabilities for texts.

        Args:
            X: List of text strings

        Returns:
            For binary: List of probabilities for positive class (float)
            For multi-class: List of probability arrays [num_labels]
        """
        return self._batch_inference(X)['probs']

    def transform(self, X: List[str]) -> torch.Tensor:
        """
        Get sparse SPLADE vectors for texts.

        Args:
            X: List of text strings

        Returns:
            Sparse vectors [num_texts, vocab_size]
        """
        return self._batch_inference(X, return_vectors=True)['vectors']

    def score(self, X: List[str], y) -> float:
        """
        Compute accuracy on test data.

        Args:
            X: List of text strings
            y: True labels

        Returns:
            Accuracy score
        """
        predictions = self.predict(X)
        correct = sum(p == t for p, t in zip(predictions, y))
        return correct / len(y)

    def get_sparsity(self, X: List[str], threshold: float = 1e-6) -> float:
        """
        Compute average per-document sparsity of vectors for texts.

        Args:
            X: List of text strings
            threshold: Values with abs < threshold are considered zero

        Returns:
            Sparsity percentage (0-100), averaged across documents
        """
        vectors = self.transform(X)
        # Compute per-document sparsity (proportion of zeros per row)
        per_doc_zeros = (vectors.abs() < threshold).sum(dim=1).float()
        per_doc_sparsity = per_doc_zeros / vectors.shape[1]
        # Average across documents
        sparsity = per_doc_sparsity.mean().item() * 100
        return sparsity

    def _should_filter_token(
        self,
        token_id: int,
        token_str: str,
        filter_special: bool = True,
        filter_stopwords: bool = True,
        filter_subwords: bool = True,
        min_length: int = 2,
        stopwords_set: Optional[Set[str]] = None,
    ) -> bool:
        """
        Determine if a token should be filtered from explanations.

        Args:
            token_id: Vocabulary index of the token
            token_str: Decoded token string
            filter_special: Filter special tokens like [CLS], [unused123]
            filter_stopwords: Filter common stopwords
            filter_subwords: Filter ##-prefixed subword tokens
            min_length: Minimum token length (filters punctuation)
            stopwords_set: Pre-loaded stopwords (computed once if None)

        Returns:
            True if token should be filtered, False otherwise
        """
        # Filter special tokens (IDs 0-999 are [PAD], [UNK], [unused...] etc.)
        if filter_special:
            if token_id < 1000:  # Special token range
                return True
            if token_str.startswith('[') and token_str.endswith(']'):
                return True

        # Filter subword pieces
        if filter_subwords and token_str.startswith('##'):
            return True

        # Filter by length (catches punctuation and single chars)
        if len(token_str) < min_length:
            return True

        # Filter stopwords
        if filter_stopwords and stopwords_set:
            if token_str.lower() in stopwords_set:
                return True

        return False

    def explain(
        self,
        text: str,
        top_k: int = 20,
        filter_special: bool = True,
        filter_stopwords: bool = True,
        filter_subwords: bool = True,
        min_token_length: int = 2,
    ) -> List[Tuple[str, float]]:
        """
        Get top weighted terms for a text.

        Args:
            text: Input text
            top_k: Number of top terms to return
            filter_special: Filter special tokens like [CLS], [unused123]
            filter_stopwords: Filter common English stopwords
            filter_subwords: Filter ##-prefixed subword tokens
            min_token_length: Minimum token length to include

        Returns:
            List of (term, weight) tuples, filtered and sorted by weight
        """
        from src.utils import load_stopwords

        vectors = self.transform([text])
        weights = vectors[0]

        # Load stopwords once if filtering
        stopwords_set = load_stopwords() if filter_stopwords else None

        # Get more candidates than needed to account for filtering
        num_candidates = min(top_k * 5, weights.shape[0])
        values, indices = torch.topk(weights, num_candidates)

        results = []
        for idx, val in zip(indices.tolist(), values.tolist()):
            if val <= 0:
                continue

            token = self.tokenizer.decode([idx]).strip()

            # Apply filtering
            if self._should_filter_token(
                token_id=idx,
                token_str=token,
                filter_special=filter_special,
                filter_stopwords=filter_stopwords,
                filter_subwords=filter_subwords,
                min_length=min_token_length,
                stopwords_set=stopwords_set,
            ):
                continue

            results.append((token, val))

            # Stop once we have enough
            if len(results) >= top_k:
                break

        return results

    def print_explanation(
        self,
        text: str,
        top_k: int = 20,
        filter_special: bool = True,
        filter_stopwords: bool = True,
        filter_subwords: bool = True,
    ):
        """
        Pretty-print explanation for a text.

        Args:
            text: Input text
            top_k: Number of top terms to show
            filter_special: Filter special tokens like [CLS], [unused123]
            filter_stopwords: Filter common English stopwords
            filter_subwords: Filter ##-prefixed subword tokens
        """
        # Get prediction
        prob = self.predict_proba([text])[0]

        if self.num_labels == 1:
            # Binary classification
            pred_label = "Positive" if prob > 0.5 else "Negative"
            confidence = prob if prob > 0.5 else 1 - prob
            print(f"\nText: {text[:100]}...")
            print(f"Prediction: {pred_label} ({confidence:.2%} confidence)")
        else:
            # Multi-class classification
            pred_idx = max(range(len(prob)), key=lambda i: prob[i])
            confidence = prob[pred_idx]
            if self.class_names and pred_idx < len(self.class_names):
                pred_label = self.class_names[pred_idx]
            else:
                pred_label = f"Class {pred_idx}"
            print(f"\nText: {text[:100]}...")
            print(f"Prediction: {pred_label} ({confidence:.2%} confidence)")
            print(f"All probabilities: {[f'{p:.2%}' for p in prob]}")

        print(f"\nTop {top_k} terms driving this prediction:")
        print("-" * 40)

        terms = self.explain(
            text, top_k,
            filter_special=filter_special,
            filter_stopwords=filter_stopwords,
            filter_subwords=filter_subwords,
        )
        for term, weight in terms:
            bar = "█" * int(weight * 10)
            print(f"  {term:<15} {weight:.3f} {bar}")

    def save(self, path: str):
        """Save model to file."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": {
                "model_name": self.model_name,
                "max_length": self.max_length,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "flops_lambda": self.flops_lambda,
                "num_labels": self.num_labels,
                "class_names": self.class_names,
                "mlm_pretrained": self.model.mlm_pretrained,
            },
            "version": 2,
        }, path)

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "SPLADEClassifier":
        """
        Load a saved model from file.

        Args:
            path: Path to saved model file
            device: Device to load model to (auto-detected if None)

        Returns:
            Loaded SPLADEClassifier instance
        """
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        config = checkpoint["config"]
        mlm_pretrained = config.get("mlm_pretrained", False)

        # Create instance without reloading MLM weights (use saved weights)
        clf = cls(
            model_name=config["model_name"],
            max_length=config["max_length"],
            batch_size=config["batch_size"],
            learning_rate=config["learning_rate"],
            flops_lambda=config["flops_lambda"],
            num_labels=config["num_labels"],
            class_names=config.get("class_names"),
            load_mlm_weights=False,
        )

        # Load saved weights
        clf.model.load_state_dict(checkpoint["model_state_dict"])
        clf.model.mlm_pretrained = mlm_pretrained

        if device:
            clf.device = device
        clf.model.to(clf.device)

        return clf

    def diagnose_mlm_head(self) -> Dict[str, Any]:
        """
        Check if the MLM head has pretrained weights.

        Returns:
            Dict with 'likely_pretrained' bool and weight statistics
        """
        vocab_proj_weight = self.model.vocab_projector.weight.data

        stats = {
            "mean": vocab_proj_weight.mean().item(),
            "std": vocab_proj_weight.std().item(),
            "mlm_pretrained_flag": getattr(self.model, 'mlm_pretrained', None),
        }

        # Pretrained has std ~0.047, random has std ~0.021
        stats["likely_pretrained"] = stats["std"] > 0.035 and abs(stats["mean"]) > 0.01

        return stats

