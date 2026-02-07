"""Sklearn-style wrapper for SPLADE classifier."""

from typing import Any, Optional

import numpy
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from transformers import AutoTokenizer

from splade.models.splade import SpladeModel
from splade.inference import (
    predict_model,
    predict_proba_model,
    transform_model,
    explain_model,
    score_model
)
from splade.utils.cuda import DEVICE, set_seed
from splade.training.loop import train_model

class SPLADEClassifier(BaseEstimator, ClassifierMixin):
    """
    SPLADE v3 Classifier with sklearn-compatible API.
    
    Attributes:
        model_name (str): HuggingFace model backbone (e.g. "distilbert-base-uncased").
        num_labels (int): Number of classification labels.
        device (str): Computation device.
    """
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_labels: int = 2,
        max_length: int = 128,
        batch_size: int = 32,
        compile_model: bool = True,
        seed: int = 42
    ):
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.batch_size = batch_size
        self.compile_model = compile_model
        self.seed = seed
        
        self.tokenizer = None
        self.model = None
        
        # Initialize immediately to allow set_params to work, 
        # but heavy loading happens in fit() or manual load().
        set_seed(self.seed)

    def _ensure_initialized(self):
        if self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = SpladeModel(self.model_name, self.num_labels).to(DEVICE)
            if self.compile_model:
                try:
                    self.model = torch.compile(self.model)
                except Exception:
                    # Fallback if compile fails (e.g. MPS)
                    pass

    def fit(
        self, 
        X: list[str], 
        y: list[int], 
        validation_data: Optional[tuple[list[str], list[int]]] = None,
        epochs: int = 3,
        learning_rate: float = 2e-5,
        regularization: str = "df_flops",
        lambda_q: float = 0.1,
        **kwargs
    ):
        """
        Train the SPLADE model.
        
        Args:
            X: List of text documents.
            y: List of integer labels.
            validation_data: Tuple of (X_val, y_val).
            epochs: Number of training epochs.
            learning_rate: Peak learning rate.
            regularization: 'flops' or 'df_flops'.
            lambda_q: Regularization strength.
        """
        self._ensure_initialized()
        
        # Construct config objects on the fly to reuse existing training loop
        # This is a lightweight shim to bridge sklearn API with the config-driven training
        from argparse import Namespace
        
        # Mocking config structures expected by train_model
        training_config = Namespace(
            epochs=epochs,
            batch_size=self.batch_size,
            lr=learning_rate,
            seed=self.seed,
            regularization=regularization,
            lambda_q=lambda_q,
            warmup_steps=kwargs.get("warmup_steps", 100),
            gradient_clipping=1.0,
            lambda_warmup_steps=kwargs.get("lambda_warmup_steps", 0),
            early_stopping_patience=kwargs.get("patience", 3)
        )
        
        model_config = Namespace(
            name=self.model_name,
            compile=self.compile_model
        )
        
        data_config = Namespace(
            max_length=self.max_length,
            num_labels=self.num_labels
        )

        # Handle validation data splitting if not provided
        val_texts, val_labels = ([], [])
        if validation_data:
            val_texts, val_labels = validation_data
        
        train_model(
            self.model, 
            self.tokenizer, 
            X, 
            y, 
            training_config, 
            model_config, 
            data_config,
            test_texts=val_texts,
            test_labels=val_labels
        )
        return self

    def predict(self, X: list[str]) -> list[int]:
        self._ensure_initialized()
        return predict_model(
            self.model, 
            self.tokenizer, 
            X, 
            self.max_length, 
            self.batch_size, 
            self.num_labels
        )

    def predict_proba(self, X: list[str]) -> list[list[float]]:
        self._ensure_initialized()
        return predict_proba_model(
            self.model, 
            self.tokenizer, 
            X, 
            self.max_length, 
            self.batch_size, 
            self.num_labels
        )

    def transform(self, X: list[str]) -> numpy.ndarray:
        """Return the sparse SPLADE embeddings."""
        self._ensure_initialized()
        return transform_model(
            self.model, 
            self.tokenizer, 
            X, 
            self.max_length, 
            self.batch_size
        )

    def score(self, X: list[str], y: list[int], sample_weight=None) -> float:
        self._ensure_initialized()
        return score_model(
            self.model, 
            self.tokenizer, 
            X, 
            y, 
            self.max_length, 
            self.batch_size, 
            self.num_labels
        )

    def explain(self, text: str, top_k: int = 10, target_class: Optional[int] = None) -> list[tuple[str, float]]:
        """Return lexical explanations for a single instance."""
        self._ensure_initialized()
        return explain_model(
            self.model, 
            self.tokenizer, 
            text, 
            self.max_length, 
            top_k, 
            target_class
        )

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self._ensure_initialized()
        self.model.load_state_dict(torch.load(path, map_location=DEVICE))
