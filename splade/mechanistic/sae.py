"""Sparse Autoencoder baseline for comparison with vocabulary-grounded decomposition.

Reference: Cunningham et al. (2023) "Sparse Autoencoders Find Highly Interpretable
Features in Language Models" (arXiv:2309.08600).
"""

import numpy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from splade.utils.cuda import COMPUTE_DTYPE, DEVICE


class SimpleSAE(nn.Module):
    """Overcomplete sparse autoencoder with L1 penalty on hidden activations."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (reconstruction, hidden_activations)."""
        hidden = torch.relu(self.encoder(x))
        reconstruction = self.decoder(hidden)
        return reconstruction, hidden

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.encoder(x))


def train_sae_on_splade(
    model: torch.nn.Module,
    input_ids_list: list[torch.Tensor],
    attention_mask_list: list[torch.Tensor],
    overcompleteness: int = 4,
    l1_coeff: float = 1e-3,
    lr: float = 1e-3,
    epochs: int = 50,
    batch_size: int = 64,
) -> SimpleSAE:
    """Train an SAE on BERT hidden states at the vocabulary projection layer.

    Collects hidden states from the model's vocab_transform output,
    then trains an overcomplete SAE to reconstruct them.
    """
    _model = model._orig_mod if hasattr(model, "_orig_mod") else model

    # Collect hidden states
    all_hidden = []
    for input_ids, attention_mask in zip(input_ids_list, attention_mask_list):
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=COMPUTE_DTYPE):
            bert_output = _model.bert(input_ids=input_ids, attention_mask=attention_mask)
            hidden = bert_output.last_hidden_state
            transformed = _model.vocab_transform(hidden)
            transformed = torch.nn.functional.gelu(transformed)
            transformed = _model.vocab_layer_norm(transformed)
        # Use CLS token representation
        all_hidden.append(transformed[:, 0, :].cpu())

    hidden_states = torch.cat(all_hidden, dim=0)
    input_dim = hidden_states.shape[-1]
    hidden_dim = input_dim * overcompleteness

    sae = SimpleSAE(input_dim, hidden_dim).to(DEVICE)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)

    loader = DataLoader(
        TensorDataset(hidden_states),
        batch_size=batch_size,
        shuffle=True,
    )

    sae.train()
    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        for (batch,) in loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            reconstruction, hidden = sae(batch)
            recon_loss = nn.functional.mse_loss(reconstruction, batch)
            l1_loss = hidden.abs().mean()
            loss = recon_loss + l1_coeff * l1_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

    sae.eval()
    return sae


def compute_sae_attribution(
    sae: SimpleSAE,
    hidden_states: torch.Tensor,
    classifier_weight: numpy.ndarray,
    class_idx: int,
) -> numpy.ndarray:
    """Compute attribution through SAE features.

    Projects hidden states through SAE encoder to get sparse features,
    then computes how each SAE feature contributes to the classifier output.
    """
    with torch.inference_mode():
        sae_features = sae.encode(hidden_states.to(DEVICE))
        decoded = sae.decoder(sae_features)

    # SAE features -> decoder -> vocab space -> classifier
    # Attribution = sae_feature * (decoder_weight @ classifier_weight)
    decoder_weight = sae.decoder.weight.cpu().numpy()  # [input_dim, hidden_dim]
    projection = decoder_weight.T @ classifier_weight[class_idx]  # [hidden_dim]

    sae_features_np = sae_features.cpu().numpy().squeeze()
    attribution = sae_features_np * projection

    return attribution
