"""SPLADE model definition supporting v3 (dReLU/Turbo) on H100."""

import torch
import torch.nn
import torch.nn.functional
import transformer_engine.pytorch as te
from transformers import AutoConfig, AutoModel, DistilBertForMaskedLM

from splade.models.layers.activation import DReLU

class SpladeModel(torch.nn.Module):
    """Encoder and classifier head for SPLADE features using H100 optimizations."""

    def __init__(self, model_name: str, num_labels: int):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name)
        self.vocab_size = config.vocab_size

        self.bert = AutoModel.from_pretrained(model_name, attn_implementation="sdpa")
        
        # H100 Optimized Layers
        self.vocab_transform = te.Linear(config.hidden_size, config.hidden_size)
        self.vocab_projector = te.Linear(config.hidden_size, self.vocab_size)

        self.vocab_layer_norm = torch.nn.LayerNorm(config.hidden_size)
        
        # Turbo Sparse Activation
        self.activation = DReLU(self.vocab_size)

        # Initialize weights from pre-trained MLM head
        masked_language_model = DistilBertForMaskedLM.from_pretrained(model_name)
        
        # Strict loading
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
        
        transformed = self.vocab_transform(hidden)
        transformed = torch.nn.functional.gelu(transformed)
        transformed = self.vocab_layer_norm(transformed)
        
        mlm_logits = self.vocab_projector(transformed)

        # dReLU Activation
        activated = self.activation(mlm_logits)

        # SPLADE Aggregate (Inline)
        log_activations = torch.log1p(activated)
        masked_activations = log_activations.masked_fill(~attention_mask.unsqueeze(-1).bool(), 0.0)
        sparse_vector = masked_activations.max(dim=1).values
        
        return self.classifier(sparse_vector), sparse_vector