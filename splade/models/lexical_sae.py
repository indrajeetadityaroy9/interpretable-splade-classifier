"""LexicalSAE: sparse autoencoder for interpretable text classification.

Produces per-position sparse representations [B, L, V] from a pretrained MLM
backbone, then pools and classifies via a ReLU MLP head with exact DLA:

    logit[c] = sum_j(s[j] * W_eff[c,j]) + b_eff[c]

Uses AutoModelForMaskedLM as a black-box backbone, delegating architecture
compatibility (BERT, DistilBERT, RoBERTa, ModernBERT, etc.) to HuggingFace.
"""

import inspect

import torch
import torch.nn
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM

from splade.circuits.core import CircuitState
from splade.models.layers.activation import JumpReLUGate
from splade.training.constants import CLASSIFIER_HIDDEN


class AttentionPool(torch.nn.Module):
    """Learned attention-weighted pooling over sequence positions.

    Computes a scalar attention score per position, then weighted-sums
    the sparse vectors. Preserves DLA identity since pooling precedes
    the classifier (W_eff is derived from the pooled vector).
    """

    def __init__(self, dim: int):
        super().__init__()
        self.query = torch.nn.Linear(dim, 1, bias=False)

    def forward(
        self, sparse_seq: torch.Tensor, attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        scores = self.query(sparse_seq).squeeze(-1)  # [B, L]
        scores = scores.masked_fill(~attention_mask.bool(), -1e9)
        weights = F.softmax(scores, dim=1)  # [B, L]
        return (sparse_seq * weights.unsqueeze(-1)).sum(dim=1)  # [B, V]


class LexicalSAE(torch.nn.Module):
    """Lexical Sparse Autoencoder for interpretable classification.

    Args:
        model_name: HuggingFace model name (e.g. "answerdotai/ModernBERT-base").
        num_labels: Number of output classes.
        vpe_config: Optional VPE configuration for polysemy expansion.
        pooling: Pooling strategy ("max" or "attention").
    """

    def __init__(
        self,
        model_name: str,
        num_labels: int,
        vpe_config=None,
        pooling: str = "max",
    ):
        super().__init__()
        self.num_labels = num_labels
        self._pooling_mode = pooling

        # Black-box backbone: encoder + MLM head in one module
        self.backbone = AutoModelForMaskedLM.from_pretrained(
            model_name, attn_implementation="sdpa",
        )
        self.vocab_size = self.backbone.config.vocab_size

        # Discover supported forward params (e.g. ModernBERT lacks token_type_ids)
        self._backbone_params = set(
            inspect.signature(self.backbone.forward).parameters.keys()
        )

        # Virtual Polysemy Expansion
        self.virtual_expander = None
        self._captured_hidden = None
        expanded_dim = self.vocab_size
        if vpe_config and vpe_config.enabled and vpe_config.token_ids:
            from splade.models.layers.virtual_expander import VirtualExpander
            self.virtual_expander = VirtualExpander(
                backbone_hidden_dim=self.backbone.config.hidden_size,
                polysemous_token_ids=vpe_config.token_ids,
                num_senses=vpe_config.num_senses,
            )
            expanded_dim = self.vocab_size + self.virtual_expander.num_virtual_slots
            # Persistent hook to capture hidden states for VPE
            self.backbone.get_output_embeddings().register_forward_pre_hook(
                self._capture_hook
            )

        # JumpReLU gate (exact binary gates for DLA identity)
        self.activation = JumpReLUGate(expanded_dim)

        # Attention-weighted pooling (optional, default: max-pool)
        self.attention_pool = (
            AttentionPool(expanded_dim) if pooling == "attention" else None
        )

        # ReLU MLP classifier head
        self.classifier_fc1 = torch.nn.Linear(expanded_dim, CLASSIFIER_HIDDEN)
        self.classifier_fc2 = torch.nn.Linear(CLASSIFIER_HIDDEN, num_labels)

    @property
    def vocab_size_expanded(self) -> int:
        """Effective sparse vector dimensionality (V + virtual slots if VPE active)."""
        if self.virtual_expander:
            return self.vocab_size + self.virtual_expander.num_virtual_slots
        return self.vocab_size

    @property
    def encoder(self) -> torch.nn.Module:
        """The base encoder (e.g. BertModel, ModernBertModel) within the backbone."""
        return getattr(self.backbone, self.backbone.base_model_prefix)

    def _capture_hook(self, module, args):
        """Persistent hook capturing hidden states before output projection."""
        self._captured_hidden = args[0]

    def _backbone_forward(self, attention_mask, *, input_ids=None, inputs_embeds=None):
        """Run backbone with cleaned kwargs."""
        kwargs = {"attention_mask": attention_mask}
        if input_ids is not None:
            kwargs["input_ids"] = input_ids
        if inputs_embeds is not None:
            kwargs["inputs_embeds"] = inputs_embeds
        kwargs = {k: v for k, v in kwargs.items() if k in self._backbone_params}
        return self.backbone(**kwargs)

    def _compute_sparse_sequence(
        self,
        attention_mask: torch.Tensor,
        *,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """SPLADE head: backbone MLM logits -> [VPE expansion] -> JumpReLU gate.

        Returns:
            sparse_seq: [B, L, V_expanded] per-position sparse representations.
            gate_mask: [B, L, V_expanded] binary gate mask {0,1}.
            l0_probs: [B, L, V_expanded] differentiable P(z > θ) for L0 loss.
        """
        mlm_logits = self._backbone_forward(
            attention_mask, input_ids=input_ids, inputs_embeds=inputs_embeds,
        ).logits  # [B, L, V]

        if self.virtual_expander is not None and self._captured_hidden is not None:
            mlm_logits = self.virtual_expander(self._captured_hidden, mlm_logits)

        activated, gate_mask, l0_probs = self.activation(mlm_logits)

        # Zero out padding positions
        sparse_seq = activated.masked_fill(
            ~attention_mask.unsqueeze(-1).bool(), 0.0
        )
        return sparse_seq, gate_mask, l0_probs

    def classify(
        self,
        sparse_sequence: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> CircuitState:
        """Pool sparse sequence and classify.

        Uses attention-weighted pooling if configured, otherwise max-pool.

        Args:
            sparse_sequence: [B, L, V_expanded] per-position sparse representations.
            attention_mask: [B, L] attention mask.

        Returns:
            CircuitState(logits, sparse_vector, W_eff, b_eff).
        """
        if self.attention_pool is not None:
            sparse_vector = self.attention_pool(sparse_sequence, attention_mask)
        else:
            sparse_vector = self.to_pooled(sparse_sequence, attention_mask)
        logits, W_eff, b_eff = self.classifier_forward(sparse_vector)
        return CircuitState(logits, sparse_vector, W_eff, b_eff)

    def classifier_forward(
        self, sparse_vector: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ReLU MLP classifier returning logits and exact DLA weights.

        Computes fc1 -> ReLU -> fc2 for logits, and derives W_eff from the
        same activation mask (single fc1 computation):
            W_eff(s) = W2 @ diag(D(s)) @ W1
            logit_c = sum_j s_j * W_eff[c,j] + b_eff_c

        Returns:
            logits: [B, C] classification logits.
            W_eff: [B, C, V_expanded] effective weight matrix for exact DLA.
            b_eff: [B, C] effective bias vector.
        """
        pre_relu = self.classifier_fc1(sparse_vector)
        activation_mask = (pre_relu > 0).float()
        hidden = pre_relu * activation_mask
        logits = self.classifier_fc2(hidden)

        W1 = self.classifier_fc1.weight  # [H, V_expanded]
        W2 = self.classifier_fc2.weight  # [C, H]
        b1 = self.classifier_fc1.bias    # [H]
        b2 = self.classifier_fc2.bias    # [C]

        masked_W1 = activation_mask.unsqueeze(-1) * W1.unsqueeze(0)
        W_eff = torch.matmul(W2.unsqueeze(0), masked_W1)
        b_eff = torch.matmul(activation_mask * b1, W2.T) + b2

        return logits, W_eff, b_eff

    def classifier_logits_only(self, sparse_vector: torch.Tensor) -> torch.Tensor:
        """ReLU MLP classifier returning only logits (for masked/patched evaluation)."""
        return self.classifier_fc2(torch.relu(self.classifier_fc1(sparse_vector)))

    def classifier_parameters(self) -> list[torch.nn.Parameter]:
        """Return classifier head parameters (for optimizer param groups)."""
        return list(self.classifier_fc1.parameters()) + list(self.classifier_fc2.parameters())

    @staticmethod
    def to_pooled(
        sparse_sequence: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Max-pool [B, L, V_expanded] to [B, V_expanded]."""
        masked = sparse_sequence.masked_fill(
            ~attention_mask.unsqueeze(-1).bool(), 0.0
        )
        return masked.max(dim=1).values

    def _get_mlm_head_input(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Capture transformed hidden state (input to output projection).

        Architecture-agnostic: registers a pre-hook on get_output_embeddings()
        to capture the intermediate representation that feeds the final
        vocabulary projection. Used by SAE comparison.

        Returns:
            [B, L, H] transformed hidden state before output projection.
        """
        captured = {}

        def _hook(module, args):
            captured["hidden"] = args[0].detach()

        handle = self.backbone.get_output_embeddings().register_forward_pre_hook(_hook)
        try:
            self._backbone_forward(attention_mask, input_ids=input_ids)
        finally:
            handle.remove()
        return captured["hidden"]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode input to per-position sparse representations.

        Returns:
            sparse_seq: [B, L, V_expanded] sparse sequence.
            gate_mask: [B, L, V_expanded] binary gate mask {0,1}.
            l0_probs: [B, L, V_expanded] differentiable P(z > θ) for L0 loss.
        """
        return self._compute_sparse_sequence(attention_mask, input_ids=input_ids)
