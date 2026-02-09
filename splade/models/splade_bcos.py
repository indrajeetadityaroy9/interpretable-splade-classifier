"""SPLADE model with B-cos classifier for exact DLA at arbitrary depth.

Drop-in replacement for SpladeModel — same BERT backbone, same SPLADE head,
but the 2-layer ReLU MLP classifier is replaced with a B-cos classifier
that provides exact W_eff decomposition for any number of layers.

Same CircuitState output, same DLA function, same evaluation pipeline.
"""

import torch

from splade.models.bcos import BcosClassifier
from splade.models.splade import SpladeModel
from splade.training.constants import CLASSIFIER_HIDDEN


class SpladeBcosModel(SpladeModel):
    """SpladeModel with B-cos classifier replacing the ReLU MLP."""

    def __init__(
        self,
        model_name: str,
        num_labels: int,
        num_bcos_layers: int = 2,
        B: int = 2,
    ):
        super().__init__(model_name, num_labels)

        # Remove the ReLU classifier layers
        del self.classifier_fc1
        del self.classifier_fc2

        self.bcos_classifier = BcosClassifier(
            self.padded_vocab_size, CLASSIFIER_HIDDEN, num_labels,
            num_layers=num_bcos_layers, B=B,
        )

    def classifier_forward(
        self, sparse_vector: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.bcos_classifier.classifier_forward(sparse_vector)

    def classifier_logits_only(self, sparse_vector: torch.Tensor) -> torch.Tensor:
        return self.bcos_classifier(sparse_vector)

    def classifier_parameters(self) -> list[torch.nn.Parameter]:
        return list(self.bcos_classifier.parameters())

    @property
    def classifier_fc2(self):
        """Compatibility: circuits/metrics.py uses classifier_fc2.weight.shape[0] for num_classes."""
        return self.bcos_classifier.layers[-1]
