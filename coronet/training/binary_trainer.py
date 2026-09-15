"""Training loop for the binary (normal/abnormal) vessel classifier.

Refactored from the training and hard-negative fine-tuning cells of
``05A_resNet75.ipynb``. Two training modes are supported:

* :meth:`VesselClassifierTrainer.fit` — standard training with
  ``BCEWithLogitsLoss`` (optionally class-weighted), a
  ``ReduceLROnPlateau`` scheduler, macro-F1-based checkpointing, and
  early stopping.
* :meth:`VesselClassifierTrainer.fine_tune` — a lower-learning-rate,
  gradient-accumulated continuation using
  ``torchvision.ops.sigmoid_focal_loss`` to focus on hard negatives,
  intended to be run on top of a checkpoint produced by :meth:`fit`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torchvision.ops as ops
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from coronet.exceptions import ModelCheckpointError
from coronet.logging_config import get_logger

logger = get_logger(__name__)


class VesselClassifierTrainer:
    """Trains and fine-tunes a binary vessel classifier (e.g. :class:`VesselResNet3D`).

    Parameters
    ----------
    model:
        The classification model to train; expected to output a single
        logit per sample.
    device:
        Torch device to run computation on.
    checkpoint_path:
        Path where the best-performing model state dict is saved.
    """

    def __init__(self, model: nn.Module, device: torch.device, checkpoint_path: str | Path) -> None:
        self.model = model.to(device)
        self.device = device
        self.checkpoint_path = Path(checkpoint_path)
        self.best_macro_f1 = 0.0

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 30,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-2,
        positive_class_weight: float = 3.0,
        early_stopping_patience: int = 6,
    ) -> float:
        """Run standard training with early stopping on validation macro-F1.

        Parameters
        ----------
        train_loader, val_loader:
            DataLoaders yielding ``{"image": tensor, "label": tensor}``
            batches.
        epochs:
            Maximum number of training epochs.
        learning_rate:
            Initial optimizer learning rate.
        weight_decay:
            AdamW weight decay coefficient.
        positive_class_weight:
            Positive-class weight passed to ``BCEWithLogitsLoss`` to
            counteract class imbalance toward the abnormal class.
        early_stopping_patience:
            Number of epochs without macro-F1 improvement before
            stopping.

        Returns
        -------
        float
            The best validation macro-F1 achieved during training.
        """
        pos_weight = torch.tensor([positive_class_weight], device=self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

        epochs_without_improvement = 0

        for epoch in range(epochs):
            train_loss = self._train_one_epoch(train_loader, optimizer, criterion)
            accuracy, macro_f1 = self._evaluate(val_loader)

            logger.info(
                "Epoch [%d/%d] | Train Loss: %.4f | Val Acc: %.4f | Macro F1: %.4f",
                epoch + 1, epochs, train_loss, accuracy, macro_f1,
            )

            scheduler.step(macro_f1)

            if macro_f1 > self.best_macro_f1:
                self.best_macro_f1 = macro_f1
                epochs_without_improvement = 0
                self._save_checkpoint()
                logger.info("New best model saved (Macro F1: %.4f)", macro_f1)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= early_stopping_patience:
                    logger.info("Early stopping triggered after %d epochs without improvement.", epochs_without_improvement)
                    break

        return self.best_macro_f1

    def fine_tune(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 20,
        learning_rate: float = 1e-5,
        accumulation_steps: int = 8,
        focal_alpha: float = 0.5,
        focal_gamma: float = 4.0,
        early_stopping_patience: int = 6,
    ) -> float:
        """Continue training with focal loss and gradient accumulation.

        Intended to run on a model whose weights were already loaded
        from a checkpoint produced by :meth:`fit`, to focus additional
        capacity on hard negative examples without catastrophic
        forgetting.

        Parameters
        ----------
        train_loader, val_loader:
            DataLoaders yielding ``{"image": tensor, "label": tensor}``
            batches. ``train_loader`` should typically use a
            class-balancing sampler.
        epochs:
            Maximum number of fine-tuning epochs.
        learning_rate:
            Optimizer learning rate (should be much smaller than the
            initial training rate).
        accumulation_steps:
            Number of physical batches to accumulate gradients over
            before each optimizer step, simulating a larger effective
            batch size under memory constraints.
        focal_alpha, focal_gamma:
            Parameters of ``torchvision.ops.sigmoid_focal_loss``.
        early_stopping_patience:
            Number of epochs without macro-F1 improvement before
            stopping.

        Returns
        -------
        float
            The best validation macro-F1 achieved during fine-tuning.
        """
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

        epochs_without_improvement = 0

        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()

            for step, batch in enumerate(train_loader):
                inputs = batch["image"].to(self.device)
                labels = batch["label"].unsqueeze(1).float().to(self.device)

                outputs = self.model(inputs)
                loss = ops.sigmoid_focal_loss(outputs, labels, alpha=focal_alpha, gamma=focal_gamma, reduction="mean")
                loss = loss / accumulation_steps
                loss.backward()

                if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_loader):
                    optimizer.step()
                    optimizer.zero_grad()

            _, macro_f1 = self._evaluate(val_loader)
            logger.info("Fine-tune epoch [%d/%d] | Macro F1: %.4f", epoch + 1, epochs, macro_f1)
            scheduler.step(macro_f1)

            if macro_f1 > self.best_macro_f1:
                self.best_macro_f1 = macro_f1
                epochs_without_improvement = 0
                self._save_checkpoint()
                logger.info("New fine-tuned champion saved (Macro F1: %.4f)", macro_f1)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= early_stopping_patience:
                    logger.info("Early stopping triggered during fine-tuning.")
                    break

        return self.best_macro_f1

    def _train_one_epoch(
        self, train_loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module
    ) -> float:
        """Run a single standard-training epoch and return the mean loss."""
        self.model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            inputs = batch["image"].to(self.device)
            labels = batch["label"].unsqueeze(1).float().to(self.device)

            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        return epoch_loss / max(len(train_loader), 1)

    def _evaluate(self, data_loader: DataLoader) -> tuple[float, float]:
        """Compute accuracy and macro-F1 on a data loader without tracking gradients."""
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in data_loader:
                inputs = batch["image"].to(self.device)
                labels = batch["label"].numpy()

                logits = self.model(inputs)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float().cpu().numpy()

                all_preds.extend(preds)
                all_targets.extend(labels)

        accuracy = accuracy_score(all_targets, all_preds)
        macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
        return accuracy, macro_f1

    def _save_checkpoint(self) -> None:
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), self.checkpoint_path)

    def load_checkpoint(self, checkpoint_path: Optional[str | Path] = None) -> None:
        """Load model weights from a checkpoint file.

        Parameters
        ----------
        checkpoint_path:
            Path to load from; defaults to ``self.checkpoint_path``.

        Raises
        ------
        ModelCheckpointError
            If the checkpoint cannot be loaded (e.g. missing file or
            architecture mismatch).
        """
        path = Path(checkpoint_path) if checkpoint_path else self.checkpoint_path
        try:
            state_dict = torch.load(path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            logger.info("Loaded checkpoint from %s", path)
        except (FileNotFoundError, RuntimeError) as exc:
            raise ModelCheckpointError(f"Failed to load checkpoint from {path}: {exc}") from exc
