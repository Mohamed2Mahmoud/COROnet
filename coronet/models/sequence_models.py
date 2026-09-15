"""Sequence-model architecture for plaque-type classification.

Refactored from ``06B_Coronet_RNN_Classifier.ipynb``. A variable-length
sequence of per-patch feature vectors along a vessel's centerline is
encoded by a recurrent (LSTM/BiLSTM/GRU) or Transformer-encoder layer,
max-pooled across the sequence dimension ("picking the worst plaque
feature"), and classified into one of four plaque-type classes.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from coronet.exceptions import ConfigurationError
from coronet.logging_config import get_logger

logger = get_logger(__name__)

_SUPPORTED_ARCHITECTURES = ("lstm", "bilstm", "gru", "transformer")


class PlaqueSequenceModel(nn.Module):
    """Classifies a vessel's plaque type from its sequence of patch feature vectors.

    Parameters
    ----------
    input_dim:
        Dimensionality of each feature vector in the input sequence.
    hidden_dim:
        Hidden size of the recurrent layer (unused for the transformer
        architecture, which preserves ``input_dim``).
    num_layers:
        Number of stacked recurrent/transformer-encoder layers.
    num_classes:
        Number of output plaque-type classes.
    architecture:
        One of ``"lstm"``, ``"bilstm"``, ``"gru"``, ``"transformer"``.

    Raises
    ------
    ConfigurationError
        If ``architecture`` is not one of the supported options.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_classes: int,
        architecture: str = "bilstm",
    ) -> None:
        super().__init__()
        self.architecture = architecture.lower()

        if self.architecture == "lstm":
            self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
            encoded_dim = hidden_dim
        elif self.architecture == "bilstm":
            self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
            encoded_dim = hidden_dim * 2
        elif self.architecture == "gru":
            self.encoder = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
            encoded_dim = hidden_dim * 2
        elif self.architecture == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=8, batch_first=True)
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            encoded_dim = input_dim
        else:
            raise ConfigurationError(
                f"Unsupported sequence architecture '{architecture}'. Supported: {_SUPPORTED_ARCHITECTURES}"
            )

        self.classifier = nn.Sequential(
            nn.Linear(encoded_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass.

        Parameters
        ----------
        x:
            Input tensor of shape ``(batch, seq_len, input_dim)``,
            zero-padded to a common sequence length.

        Returns
        -------
        torch.Tensor
            Class logits of shape ``(batch, num_classes)``.
        """
        if self.architecture in ("lstm", "bilstm", "gru"):
            encoded, _ = self.encoder(x)
        else:
            encoded = self.encoder(x)

        pooled, _ = torch.max(encoded, dim=1)
        return self.classifier(pooled)
