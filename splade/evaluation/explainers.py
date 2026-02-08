"""Baseline explainer implementations for interpretability benchmarking.

Provides 7 explainers: LIME, Integrated Gradients, GradientSHAP,
Attention, Saliency, DeepLIFT, and Random.
"""

import abc

import numpy
import torch
from captum.attr import DeepLift, GradientShap, IntegratedGradients, Saliency
from lime.lime_text import LimeTextExplainer

from splade.evaluation.constants import SPECIAL_TOKENS
from splade.utils.cuda import COMPUTE_DTYPE, DEVICE, unwrap_compiled


class BaseExplainer(abc.ABC):
    @abc.abstractmethod
    def explain(
        self,
        model: torch.nn.Module,
        tokenizer,
        text: str,
        max_length: int,
        top_k: int,
    ) -> list[tuple[str, float]]:
        ...

    def explain_batch(
        self,
        model: torch.nn.Module,
        tokenizer,
        texts: list[str],
        max_length: int,
        top_k: int,
        batch_size: int = 32,
    ) -> list[list[tuple[str, float]]]:
        return [self.explain(model, tokenizer, t, max_length, top_k) for t in texts]


def _embedding_attributions_to_token_list(
    attributions: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tokenizer,
    top_k: int,
) -> list[tuple[str, float]]:
    """Convert per-token embedding attributions to sorted (token, score) list."""
    scores = attributions.norm(dim=-1)
    mask_bool = attention_mask.bool()
    token_scores = []
    for i in range(input_ids.shape[0]):
        if not mask_bool[i]:
            continue
        token = tokenizer.convert_ids_to_tokens(int(input_ids[i].item()))
        if token in SPECIAL_TOKENS:
            continue
        token_scores.append((token, float(scores[i].item())))

    token_scores.sort(key=lambda x: x[1], reverse=True)
    return token_scores[:top_k]


class LIMEExplainer(BaseExplainer):
    """LIME text explainer using lime library."""

    def __init__(self, seed: int = 42, n_samples: int = 500, **_):
        self.seed = seed
        self.n_samples = n_samples

    def explain(self, model, tokenizer, text, max_length, top_k):
        _model = unwrap_compiled(model)

        def predict_fn(texts_list):
            encoding = tokenizer(
                list(texts_list), max_length=max_length,
                padding="max_length", truncation=True, return_tensors="pt",
            )
            with torch.inference_mode(), torch.amp.autocast("cuda", dtype=COMPUTE_DTYPE):
                logits, _, _, _ = _model(
                    encoding["input_ids"].to(DEVICE),
                    encoding["attention_mask"].to(DEVICE),
                )
            # LIME requires numpy arrays at the API boundary
            return torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()

        explainer = LimeTextExplainer(random_state=self.seed)
        probs = predict_fn([text])
        target_class = int(probs[0].argmax())

        explanation = explainer.explain_instance(
            text, predict_fn, labels=(target_class,),
            num_features=top_k, num_samples=self.n_samples,
        )
        word_weights = explanation.as_list(label=target_class)
        return [(w, float(s)) for w, s in word_weights[:top_k]]


class IntegratedGradientsExplainer(BaseExplainer):
    """Integrated Gradients on embedding layer."""

    def __init__(self, n_steps: int = 50, **_):
        self.n_steps = n_steps

    def explain(self, model, tokenizer, text, max_length, top_k):
        _model = unwrap_compiled(model)

        encoding = tokenizer(
            [text], max_length=max_length, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(DEVICE)
        attention_mask = encoding["attention_mask"].to(DEVICE)

        embeddings = _model.get_embeddings(input_ids)
        baseline = torch.zeros_like(embeddings)

        def forward_fn(embeds, mask):
            logits, _, _, _ = _model.forward_from_embeddings(embeds, mask)
            return logits

        with torch.inference_mode():
            logits, _, _, _ = _model(input_ids, attention_mask)
        target = int(logits.argmax(dim=-1).item())

        ig = IntegratedGradients(forward_fn)
        attrs = ig.attribute(
            embeddings, baselines=baseline, target=target,
            additional_forward_args=(attention_mask,), n_steps=self.n_steps,
        )
        return _embedding_attributions_to_token_list(
            attrs[0], input_ids[0], attention_mask[0], tokenizer, top_k,
        )


class GradientShapExplainer(BaseExplainer):
    """GradientSHAP on embedding layer."""

    def __init__(self, n_samples: int = 25, seed: int = 42, **_):
        self.n_samples = n_samples
        self.seed = seed

    def explain(self, model, tokenizer, text, max_length, top_k):
        _model = unwrap_compiled(model)

        encoding = tokenizer(
            [text], max_length=max_length, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(DEVICE)
        attention_mask = encoding["attention_mask"].to(DEVICE)

        embeddings = _model.get_embeddings(input_ids)
        baselines = torch.zeros(self.n_samples, *embeddings.shape[1:], device=DEVICE)

        def forward_fn(embeds, mask):
            logits, _, _, _ = _model.forward_from_embeddings(embeds, mask)
            return logits

        with torch.inference_mode():
            logits, _, _, _ = _model(input_ids, attention_mask)
        target = int(logits.argmax(dim=-1).item())

        gs = GradientShap(forward_fn)
        torch.manual_seed(self.seed)
        attrs = gs.attribute(
            embeddings, baselines=baselines, target=target,
            additional_forward_args=(attention_mask,), n_samples=self.n_samples,
        )
        return _embedding_attributions_to_token_list(
            attrs[0], input_ids[0], attention_mask[0], tokenizer, top_k,
        )


class AttentionExplainer(BaseExplainer):
    """Attention-based explanation: mean attention to CLS from last layer."""

    def __init__(self, **_):
        pass

    def explain(self, model, tokenizer, text, max_length, top_k):
        _model = unwrap_compiled(model)

        encoding = tokenizer(
            [text], max_length=max_length, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(DEVICE)
        attention_mask = encoding["attention_mask"].to(DEVICE)

        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=COMPUTE_DTYPE):
            outputs = _model.bert(
                input_ids=input_ids, attention_mask=attention_mask,
                output_attentions=True,
            )

        last_attention = outputs.attentions[-1]
        cls_attention = last_attention[0, :, 0, :].mean(dim=0)
        ids = input_ids[0]
        mask = attention_mask[0]

        token_scores = []
        for i in range(ids.shape[0]):
            if not mask[i]:
                continue
            token = tokenizer.convert_ids_to_tokens(int(ids[i].item()))
            if token in SPECIAL_TOKENS:
                continue
            token_scores.append((token, float(cls_attention[i].item())))

        token_scores.sort(key=lambda x: x[1], reverse=True)
        return token_scores[:top_k]


class SaliencyExplainer(BaseExplainer):
    """Saliency (gradient magnitude) on embedding layer."""

    def __init__(self, **_):
        pass

    def explain(self, model, tokenizer, text, max_length, top_k):
        _model = unwrap_compiled(model)

        encoding = tokenizer(
            [text], max_length=max_length, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(DEVICE)
        attention_mask = encoding["attention_mask"].to(DEVICE)

        embeddings = _model.get_embeddings(input_ids)

        def forward_fn(embeds, mask):
            logits, _, _, _ = _model.forward_from_embeddings(embeds, mask)
            return logits

        with torch.inference_mode():
            logits, _, _, _ = _model(input_ids, attention_mask)
        target = int(logits.argmax(dim=-1).item())

        saliency = Saliency(forward_fn)
        attrs = saliency.attribute(
            embeddings, target=target,
            additional_forward_args=(attention_mask,),
        )
        return _embedding_attributions_to_token_list(
            attrs[0], input_ids[0], attention_mask[0], tokenizer, top_k,
        )


class DeepLiftExplainer(BaseExplainer):
    """DeepLIFT on embedding layer with zero baseline."""

    def __init__(self, **_):
        pass

    def explain(self, model, tokenizer, text, max_length, top_k):
        _model = unwrap_compiled(model)

        encoding = tokenizer(
            [text], max_length=max_length, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(DEVICE)
        attention_mask = encoding["attention_mask"].to(DEVICE)

        embeddings = _model.get_embeddings(input_ids)
        baseline = torch.zeros_like(embeddings)

        def forward_fn(embeds, mask):
            logits, _, _, _ = _model.forward_from_embeddings(embeds, mask)
            return logits

        with torch.inference_mode():
            logits, _, _, _ = _model(input_ids, attention_mask)
        target = int(logits.argmax(dim=-1).item())

        dl = DeepLift(forward_fn)
        attrs = dl.attribute(
            embeddings, baselines=baseline, target=target,
            additional_forward_args=(attention_mask,),
        )
        return _embedding_attributions_to_token_list(
            attrs[0], input_ids[0], attention_mask[0], tokenizer, top_k,
        )


class RandomExplainer(BaseExplainer):
    """Random baseline: uniform random scores on non-special tokens."""

    def __init__(self, seed: int = 42, **_):
        self.seed = seed

    def explain(self, model, tokenizer, text, max_length, top_k):
        rng = torch.Generator(device=DEVICE).manual_seed(self.seed)

        encoding = tokenizer(
            [text], max_length=max_length, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        ids = encoding["input_ids"][0]
        mask = encoding["attention_mask"][0]

        token_scores = []
        for i in range(ids.shape[0]):
            if not mask[i]:
                continue
            token = tokenizer.convert_ids_to_tokens(int(ids[i].item()))
            if token in SPECIAL_TOKENS:
                continue
            token_scores.append((token, float(torch.rand(1, device=DEVICE, generator=rng).item())))

        token_scores.sort(key=lambda x: x[1], reverse=True)
        return token_scores[:top_k]


EXPLAINER_REGISTRY: dict[str, type[BaseExplainer]] = {
    "lime": LIMEExplainer,
    "ig": IntegratedGradientsExplainer,
    "gradshap": GradientShapExplainer,
    "attention": AttentionExplainer,
    "saliency": SaliencyExplainer,
    "deeplift": DeepLiftExplainer,
    "random": RandomExplainer,
}
