"""Baseline explainers that operate on a trained SPLADE classifier."""

import numpy
import torch
from captum.attr import LayerIntegratedGradients
from lime.lime_text import LimeTextExplainer

from src.models.classifier import SPECIAL_TOKENS, SPLADEClassifier
from src.utils.cuda import DEVICE


class SPLADEAttentionExplainer:
    """Attention baseline over the classifier encoder."""

    def __init__(self, clf: SPLADEClassifier):
        self.clf = clf
        self.tokenizer = clf.tokenizer
        self.num_labels = clf.num_labels

    def explain(self, text: str, top_k: int = 10) -> list[tuple[str, float]]:
        encoding = self.tokenizer(
            text,
            max_length=self.clf.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(DEVICE, non_blocking=True)
        attention_mask = encoding["attention_mask"].to(DEVICE, non_blocking=True)

        with torch.inference_mode():
            outputs = self.clf.model.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
            )

        attention = outputs.attentions[-1][0]
        sequence_length = int(attention_mask.sum().item())
        averaged_attention = attention.mean(dim=0)
        token_importance = averaged_attention[:sequence_length, :sequence_length].sum(dim=0)

        weights = token_importance.cpu().numpy()
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0, :sequence_length].cpu().tolist())

        explanations = [
            (token.replace("##", ""), float(weight))
            for token, weight in zip(tokens, weights)
            if token not in SPECIAL_TOKENS
        ]
        explanations.sort(key=lambda pair: pair[1], reverse=True)
        return explanations[:top_k]


class SPLADEIntegratedGradientsExplainer:
    """Integrated Gradients baseline over SPLADE embeddings."""

    def __init__(self, clf: SPLADEClassifier, n_steps: int = 50):
        self.clf = clf
        self.tokenizer = clf.tokenizer
        self.num_labels = clf.num_labels
        self.n_steps = n_steps

        def forward_func(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            logits, _ = self.clf.model(input_ids, attention_mask)
            return logits

        self.integrated_gradients = LayerIntegratedGradients(
            forward_func,
            self.clf.model.bert.embeddings.word_embeddings,
        )

    def explain(self, text: str, top_k: int = 10) -> list[tuple[str, float]]:
        encoding = self.tokenizer(
            text,
            max_length=self.clf.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(DEVICE, non_blocking=True)
        attention_mask = encoding["attention_mask"].to(DEVICE, non_blocking=True)
        baseline_ids = torch.full_like(input_ids, self.tokenizer.pad_token_id)

        with torch.inference_mode():
            logits, _ = self.clf.model(input_ids, attention_mask)
            predicted_class = logits.argmax(dim=-1).item()

        attributions = self.integrated_gradients.attribute(
            inputs=input_ids,
            baselines=baseline_ids,
            additional_forward_args=(attention_mask,),
            target=predicted_class,
            n_steps=self.n_steps,
        )

        token_attributions = attributions.sum(dim=-1).squeeze(0)
        sequence_length = int(attention_mask.sum().item())
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0, :sequence_length].cpu().tolist())
        values = token_attributions[:sequence_length].cpu().detach().numpy()

        explanations = [
            (token.replace("##", ""), float(abs(value)))
            for token, value in zip(tokens, values)
            if token not in SPECIAL_TOKENS
        ]
        explanations.sort(key=lambda pair: pair[1], reverse=True)
        return explanations[:top_k]


class SPLADELIMEExplainer:
    """LIME baseline over SPLADE predictions."""

    def __init__(self, clf: SPLADEClassifier, num_samples: int = 500):
        self.clf = clf
        self.tokenizer = clf.tokenizer
        self.num_labels = clf.num_labels
        self.num_samples = num_samples
        self.lime_explainer = LimeTextExplainer(
            class_names=[f"class_{label}" for label in range(clf.num_labels)],
        )

    def explain(self, text: str, top_k: int = 10) -> list[tuple[str, float]]:
        def predict_fn(texts) -> numpy.ndarray:
            text_list = texts.tolist() if hasattr(texts, "tolist") else list(texts)
            return numpy.array(self.clf.predict_proba(text_list))

        probabilities = self.clf.predict_proba([text])[0]
        predicted_class = int(numpy.argmax(probabilities))

        explanation = self.lime_explainer.explain_instance(
            text,
            predict_fn,
            num_features=top_k,
            num_samples=self.num_samples,
            labels=[predicted_class],
        )
        scores = explanation.as_list(label=predicted_class)
        scores.sort(key=lambda pair: abs(pair[1]), reverse=True)
        return [(word, float(weight)) for word, weight in scores[:top_k]]
