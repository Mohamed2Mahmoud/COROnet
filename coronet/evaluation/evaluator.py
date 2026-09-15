"""Model evaluation: TTA inference, threshold optimization, and metrics reporting.

Refactored from the repeated evaluation cells of ``05A_resNet75.ipynb``
(binary vessel classifier: test-time augmentation, optimal-threshold
search, artery-level vs. patient-level aggregation) and the
``run_experiment`` evaluation portion of
``06B_Coronet_RNN_Classifier.ipynb`` (sequence classifier: confusion
matrix and classification report persistence).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from coronet.logging_config import get_logger

logger = get_logger(__name__)


class ThresholdOptimizer:
    """Finds the classification threshold that maximizes F1 on a precision-recall curve."""

    @staticmethod
    def find_optimal_f1_threshold(targets: np.ndarray, probabilities: np.ndarray) -> Tuple[float, float]:
        """Search the precision-recall curve for the F1-maximizing threshold.

        Parameters
        ----------
        targets:
            Ground-truth binary labels.
        probabilities:
            Predicted probabilities for the positive class.

        Returns
        -------
        tuple of float
            ``(best_threshold, best_f1)``.
        """
        precisions, recalls, thresholds = precision_recall_curve(targets, probabilities)
        f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
        # precision_recall_curve returns one more precision/recall pair than thresholds.
        best_index = int(np.argmax(f1_scores[:-1])) if len(thresholds) > 0 else 0
        if len(thresholds) == 0:
            return 0.5, 0.0
        return float(thresholds[best_index]), float(f1_scores[best_index])


class BinaryVesselEvaluator:
    """Evaluates a binary vessel classifier with optional test-time augmentation.

    Parameters
    ----------
    model:
        A trained model outputting a single logit per sample.
    device:
        Torch device to run inference on.
    use_tta:
        If ``True``, average sigmoid probabilities across the original
        volume and its two spatial flips (matching the "Safe TTA" used
        throughout the original notebooks).
    """

    def __init__(self, model: nn.Module, device: torch.device, use_tta: bool = True) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.use_tta = use_tta

    def _predict_batch_probabilities(self, inputs: torch.Tensor) -> torch.Tensor:
        """Compute sigmoid probabilities for a batch, optionally averaging over TTA views."""
        with torch.no_grad():
            logits = self.model(inputs)
            probs = torch.sigmoid(logits)

            if self.use_tta:
                logits_flip_x = self.model(torch.flip(inputs, dims=[3]))
                logits_flip_y = self.model(torch.flip(inputs, dims=[4]))
                probs = (probs + torch.sigmoid(logits_flip_x) + torch.sigmoid(logits_flip_y)) / 3.0

        return probs

    def predict(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Run inference over an entire data loader.

        Parameters
        ----------
        data_loader:
            DataLoader yielding ``{"image": tensor, "label": tensor,
            "path": str}`` batches (``"path"`` is optional; an empty
            string is used when absent).

        Returns
        -------
        tuple
            ``(probabilities, targets, relative_paths)`` as, respectively,
            a 1D array of predicted probabilities, a 1D array of
            ground-truth labels, and a list of source file paths (used
            later for patient-level aggregation).
        """
        all_probs: List[np.ndarray] = []
        all_targets: List[np.ndarray] = []
        all_paths: List[str] = []

        for batch in data_loader:
            inputs = batch["image"].to(self.device)
            labels = batch["label"].numpy()
            probs = self._predict_batch_probabilities(inputs).cpu().numpy()

            all_probs.extend(probs)
            all_targets.extend(labels)
            all_paths.extend(batch.get("path", [""] * len(labels)))

        return np.array(all_probs).flatten(), np.array(all_targets).flatten(), all_paths

    @staticmethod
    def aggregate_to_patient_level(
        probabilities: np.ndarray, targets: np.ndarray, relative_paths: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Aggregate per-vessel predictions to per-patient predictions.

        A patient is grouped by the parent directory of each vessel's
        relative path (falling back to the leading token of the
        filename when no directory is present). The patient-level
        target is positive if any vessel is abnormal; the patient-level
        probability is the maximum vessel-level probability.

        Parameters
        ----------
        probabilities, targets, relative_paths:
            Outputs of :meth:`predict`.

        Returns
        -------
        tuple
            ``(patient_targets, patient_probabilities, patient_ids)``.
        """
        patient_data: Dict[str, Dict[str, list]] = {}

        for probability, target, relative_path in zip(probabilities, targets, relative_paths):
            folder_name = os.path.dirname(relative_path) or os.path.basename(relative_path).split("_")[0]
            bucket = patient_data.setdefault(folder_name, {"probs": [], "targets": []})
            bucket["probs"].append(probability)
            bucket["targets"].append(target)

        patient_ids = list(patient_data.keys())
        patient_targets = np.array([1.0 if sum(v["targets"]) > 0 else 0.0 for v in patient_data.values()])
        patient_probs = np.array([max(v["probs"]) for v in patient_data.values()])

        return patient_targets, patient_probs, patient_ids

    def evaluate(
        self,
        data_loader: DataLoader,
        threshold: Optional[float] = None,
        patient_level: bool = False,
    ) -> Dict[str, float]:
        """Run inference and compute classification metrics at a given (or optimal) threshold.

        Parameters
        ----------
        data_loader:
            DataLoader to evaluate.
        threshold:
            Decision threshold. If ``None``, the F1-optimal threshold is
            found automatically via :class:`ThresholdOptimizer`.
        patient_level:
            If ``True``, aggregate predictions to patient level before
            scoring.

        Returns
        -------
        dict
            ``{"threshold", "accuracy", "precision", "recall", "f1"}``.
        """
        probabilities, targets, relative_paths = self.predict(data_loader)

        if patient_level:
            targets, probabilities, _ = self.aggregate_to_patient_level(probabilities, targets, relative_paths)

        if threshold is None:
            threshold, _ = ThresholdOptimizer.find_optimal_f1_threshold(targets, probabilities)

        predictions = (probabilities >= threshold).astype(int)

        metrics = {
            "threshold": float(threshold),
            "accuracy": float(accuracy_score(targets, predictions)),
            "precision": float(precision_score(targets, predictions, zero_division=0)),
            "recall": float(recall_score(targets, predictions, zero_division=0)),
            "f1": float((2 * precision_score(targets, predictions, zero_division=0) * recall_score(targets, predictions, zero_division=0))
                        / (precision_score(targets, predictions, zero_division=0) + recall_score(targets, predictions, zero_division=0) + 1e-8)),
        }
        logger.info("Evaluation metrics (%s-level): %s", "patient" if patient_level else "vessel", metrics)
        return metrics

    @staticmethod
    def plot_confusion_matrix(
        targets: np.ndarray,
        predictions: np.ndarray,
        class_names: Tuple[str, str],
        output_path: str | Path,
        title: str = "Confusion Matrix",
    ) -> Path:
        """Render and save a confusion matrix heatmap.

        Parameters
        ----------
        targets, predictions:
            Ground-truth and predicted binary labels.
        class_names:
            ``(negative_name, positive_name)`` display labels.
        output_path:
            Destination PNG path.
        title:
            Plot title.

        Returns
        -------
        pathlib.Path
            Path the figure was saved to.
        """
        cm = confusion_matrix(targets, predictions)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=[f"Predicted {name}" for name in class_names],
            yticklabels=[f"Actual {name}" for name in class_names],
        )
        plt.title(title)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        return output_path


class SequenceClassificationEvaluator:
    """Evaluates a plaque-type sequence classifier and persists metrics artifacts.

    Parameters
    ----------
    model:
        A trained :class:`coronet.models.sequence_models.PlaqueSequenceModel`.
    device:
        Torch device to run inference on.
    class_names:
        Ordered display names for each plaque-type class.
    """

    def __init__(self, model: nn.Module, device: torch.device, class_names: Tuple[str, ...]) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.class_names = class_names

    def predict(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """Run inference over a data loader of padded feature sequences.

        Parameters
        ----------
        data_loader:
            DataLoader yielding ``(padded_sequences, labels)`` batches.

        Returns
        -------
        tuple of numpy.ndarray
            ``(predictions, targets)``.
        """
        all_preds: List[int] = []
        all_targets: List[int] = []

        with torch.no_grad():
            for sequences, labels in data_loader:
                outputs = self.model(sequences.to(self.device))
                all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                all_targets.extend(labels.numpy())

        return np.array(all_preds), np.array(all_targets)

    def evaluate_and_save(self, data_loader: DataLoader, output_dir: str | Path, experiment_name: str) -> Dict[str, float]:
        """Run inference, save a confusion matrix and text report, and return summary metrics.

        Parameters
        ----------
        data_loader:
            DataLoader to evaluate.
        output_dir:
            Directory where ``confusion_matrix.png`` and
            ``metrics_report.txt`` are written.
        experiment_name:
            Human-readable experiment identifier used in report headers
            and filenames.

        Returns
        -------
        dict
            ``{"accuracy", "weighted_f1", "weighted_recall"}``.
        """
        predictions, targets = self.predict(data_loader)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        report_dict = classification_report(
            targets, predictions, target_names=list(self.class_names), output_dict=True, zero_division=0
        )
        cm = confusion_matrix(targets, predictions)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=list(self.class_names), yticklabels=list(self.class_names),
        )
        plt.title(f"CM: {experiment_name}")
        plt.tight_layout()
        plt.savefig(output_dir / "confusion_matrix.png")
        plt.close()

        accuracy = accuracy_score(targets, predictions)
        with (output_dir / "metrics_report.txt").open("w", encoding="utf-8") as handle:
            handle.write(f"Experiment: {experiment_name}\nAccuracy: {accuracy:.4f}\n")
            handle.write(
                classification_report(targets, predictions, target_names=list(self.class_names), zero_division=0)
            )

        summary = {
            "accuracy": float(accuracy),
            "weighted_f1": float(report_dict["weighted avg"]["f1-score"]),
            "weighted_recall": float(report_dict["weighted avg"]["recall"]),
        }
        logger.info("Saved evaluation artifacts for %s to %s: %s", experiment_name, output_dir, summary)
        return summary
