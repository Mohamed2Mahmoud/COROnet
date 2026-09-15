"""Training loop for the plaque-type sequence classifier.

Refactored from the training portion of ``run_experiment`` in
``06B_Coronet_RNN_Classifier.ipynb``. Evaluation (metrics, confusion
matrices) is intentionally handled by
:class:`coronet.evaluation.evaluator.SequenceClassificationEvaluator`
rather than this class, to keep training and evaluation concerns
separate.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from coronet.logging_config import get_logger

logger = get_logger(__name__)


class PlaqueSequenceTrainer:
    """Trains a :class:`coronet.models.sequence_models.PlaqueSequenceModel`.

    Parameters
    ----------
    model:
        The sequence classification model to train.
    device:
        Torch device to run computation on.
    class_weights:
        Optional per-class weight tensor passed to
        ``CrossEntropyLoss`` to counteract class imbalance.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        class_weights: Optional[torch.Tensor] = None,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.class_weights = class_weights.to(device) if class_weights is not None else None

    def fit(
        self,
        train_loader: DataLoader,
        epochs: int = 20,
        learning_rate: float = 1e-4,
        log_every_n_epochs: int = 5,
    ) -> List[float]:
        """Train the sequence classifier for a fixed number of epochs.

        Parameters
        ----------
        train_loader:
            DataLoader yielding ``(padded_sequences, labels)`` batches,
            e.g. produced with
            :func:`coronet.data.datasets.sequence_collate_fn`.
        epochs:
            Number of training epochs.
        learning_rate:
            Adam optimizer learning rate.
        log_every_n_epochs:
            Frequency (in epochs) at which the mean training loss is
            logged.

        Returns
        -------
        list of float
            Mean training loss for every epoch.
        """
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss(weight=self.class_weights)

        epoch_losses: List[float] = []

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0

            for sequences, labels in train_loader:
                sequences, labels = sequences.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(sequences)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            mean_loss = running_loss / max(len(train_loader), 1)
            epoch_losses.append(mean_loss)

            if (epoch + 1) % log_every_n_epochs == 0 or epoch == 0:
                logger.info("Epoch [%d/%d] - Loss: %.4f", epoch + 1, epochs, mean_loss)

        return epoch_losses

    @staticmethod
    def compute_balanced_class_weights(label_counts: dict, class_order: List[str], device: torch.device) -> torch.Tensor:
        """Compute inverse-frequency class weights normalized to sum to ``len(class_order)``.

        Parameters
        ----------
        label_counts:
            Mapping from class name to sample count in the training set.
        class_order:
            Ordered list of class names defining label-index alignment.
        device:
            Device the resulting tensor should live on.

        Returns
        -------
        torch.Tensor
            Normalized class weights, one per entry in ``class_order``.
        """
        weights = []
        for class_name in class_order:
            count = label_counts.get(class_name, 0)
            weights.append(1.0 if count == 0 else 1.0 / count)

        weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
        weights_tensor = weights_tensor / weights_tensor.sum() * len(class_order)
        return weights_tensor
