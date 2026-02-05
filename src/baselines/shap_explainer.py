"""SHAP-based explainer for text classification."""

import numpy as np
import shap

from src.baselines.base import BaseExplainer


class SHAPExplainer(BaseExplainer):
    """Generate explanations using SHAP."""

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_labels: int = 2,
        max_length: int = 128,
        max_evals: int = 500,
    ):
        super().__init__(model_name, num_labels, max_length)
        self.max_evals = max_evals
        self._shap_explainer = None

    def _predict_fn(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        return np.array(self.predict_proba(list(texts)))

    def fit(self, texts: list[str], labels: list[int], epochs: int = 3,
            batch_size: int = 32, learning_rate: float = 2e-5) -> "SHAPExplainer":
        """Fine-tune model and initialize SHAP partition explainer."""
        super().fit(texts, labels, epochs, batch_size, learning_rate)
        self._shap_explainer = shap.Explainer(
            self._predict_fn, self.tokenizer,
            output_names=[f"class_{i}" for i in range(self.num_labels)],
        )
        return self

    def explain(self, text: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Compute SHAP values for the predicted class."""
        probs = self.predict_proba([text])[0]
        pred_class = int(np.argmax(probs))

        shap_values = self._shap_explainer([text], max_evals=self.max_evals)

        values = shap_values.values[0][:, pred_class]

        tokens = shap_values.data[0]

        explanations = [
            (str(token), float(value))
            for token, value in zip(tokens, values)
            if token and token.strip()
        ]
        explanations.sort(key=lambda x: abs(x[1]), reverse=True)
        return explanations[:top_k]
