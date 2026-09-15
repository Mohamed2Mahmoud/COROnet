"""PyTorch dataset classes for vessel volumes and extracted feature sequences.

Refactored from the dataset-construction logic in
``05A_resNet75.ipynb`` (volume-level binary classification) and
``06B_Coronet_RNN_Classifier.ipynb`` (feature-sequence multi-class
classification).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from monai.data import Dataset as MonaiDataset
from monai.transforms import Compose
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from coronet.data.preprocessing import DEFAULT_PLAQUE_LABEL_MAP
from coronet.logging_config import get_logger

logger = get_logger(__name__)


class VesselFileListBuilder:
    """Builds MONAI-compatible ``{"image": ..., "label": ...}`` file lists.

    Parameters
    ----------
    nifti_dir:
        Root directory containing per-patient vessel patch volumes.
    """

    def __init__(self, nifti_dir: str | Path) -> None:
        self.nifti_dir = Path(nifti_dir)

    def _resolve_vessel_path(self, folder_name: str, vessel_name: str) -> Optional[Path]:
        for extension in (".nii.gz", ".nii"):
            candidate = self.nifti_dir / folder_name / f"{vessel_name}{extension}"
            if candidate.exists():
                return candidate
        return None

    def from_patient_vessel_split(
        self, split_table: pd.DataFrame, patient_col: str, vessel_status_col: str, split_col: str
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Build train/validation file lists from a ``patient_vessel_id`` style split table.

        Parameters
        ----------
        split_table:
            DataFrame with columns identified by ``patient_col``
            (formatted like ``"ICC_Patient_0001_RCA"``),
            ``vessel_status_col`` (``"Normal"``/``"Abnormal"``), and
            ``split_col`` (``"train"``/other).
        patient_col:
            Name of the composite patient/vessel identifier column.
        vessel_status_col:
            Name of the binary status column.
        split_col:
            Name of the train/validation split column.

        Returns
        -------
        tuple of list of dict
            ``(train_files, val_files)``, each a list of
            ``{"image": path, "label": 0.0/1.0}`` dictionaries.
        """
        train_files: List[Dict[str, Any]] = []
        val_files: List[Dict[str, Any]] = []
        missing_count = 0

        for _, row in split_table.iterrows():
            composite_id = str(row[patient_col]).strip()
            parts = composite_id.split("_")
            if len(parts) < 4:
                missing_count += 1
                continue

            folder_name = f"{parts[1]}_{parts[2]}"
            vessel_name = parts[-1]
            resolved_path = self._resolve_vessel_path(folder_name, vessel_name)

            if resolved_path is None:
                missing_count += 1
                continue

            label = 1.0 if str(row[vessel_status_col]).strip().lower() == "abnormal" else 0.0
            record = {"image": str(resolved_path), "label": label}

            if str(row[split_col]).strip().lower() == "train":
                train_files.append(record)
            else:
                val_files.append(record)

        if missing_count:
            logger.warning("%d rows could not be mapped to an existing NIfTI file.", missing_count)
        logger.info("Loaded %d train / %d validation vessel volumes.", len(train_files), len(val_files))
        return train_files, val_files

    def from_relative_path_split(
        self, split_table: pd.DataFrame, path_col: str, status_col: str, split_col: str, split_value: str
    ) -> List[Dict[str, Any]]:
        """Build a file list from a split table storing relative paths directly.

        Parameters
        ----------
        split_table:
            DataFrame with a column of relative paths under
            ``nifti_dir``.
        path_col:
            Name of the relative-path column.
        status_col:
            Name of the binary status column.
        split_col:
            Name of the split column.
        split_value:
            Value of ``split_col`` to filter on (e.g. ``"test"``).

        Returns
        -------
        list of dict
            ``{"image": path, "label": 0.0/1.0, "path": relative_path}``
            dictionaries for every matching, existing file.
        """
        files: List[Dict[str, Any]] = []
        subset = split_table[split_table[split_col].astype(str).str.strip().str.lower() == split_value.lower()]

        for _, row in subset.iterrows():
            relative_path = str(row[path_col]).strip()
            full_path = self.nifti_dir / relative_path
            if full_path.exists():
                label = 1.0 if str(row[status_col]).strip().lower() == "abnormal" else 0.0
                files.append({"image": str(full_path), "label": label, "path": relative_path})

        return files


class VesselVolumeDataset(MonaiDataset):
    """MONAI dataset of vessel-patch volumes for binary normal/abnormal classification.

    Parameters
    ----------
    file_list:
        List of ``{"image": path, "label": float, ...}`` dictionaries,
        typically produced by :class:`VesselFileListBuilder`.
    transform:
        MONAI transform pipeline (see
        :class:`coronet.data.preprocessing.VesselTransformFactory`).
    """

    def __init__(self, file_list: List[Dict[str, Any]], transform: Compose) -> None:
        super().__init__(data=file_list, transform=transform)


class VesselSequenceDataset(Dataset):
    """Dataset of pre-extracted feature sequences for plaque-type classification.

    Each item is a variable-length sequence of 512-d feature vectors
    (one vector per patch along a vessel's centerline), as produced by
    :class:`coronet.features.feature_extraction_service.FeatureExtractionService`.

    Parameters
    ----------
    split_table:
        DataFrame with ``Patient_ID``, ``Vessel``, and ``Plaque_Type``
        columns.
    feature_dir:
        Directory containing ``<patient>_<vessel>_features.npy`` files
        for a single backbone.
    feature_dim:
        Dimensionality of each feature vector, used to build a
        zero-padded fallback sequence when a feature file is missing.
    label_map:
        Mapping from plaque-type string to integer class id.
    """

    def __init__(
        self,
        split_table: pd.DataFrame,
        feature_dir: str | Path,
        feature_dim: int = 512,
        label_map: Optional[Dict[str, int]] = None,
    ) -> None:
        self.split_table = split_table.reset_index(drop=True)
        self.feature_dir = Path(feature_dir)
        self.feature_dim = feature_dim
        self.label_map = label_map or dict(DEFAULT_PLAQUE_LABEL_MAP)

    def __len__(self) -> int:
        return len(self.split_table)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.split_table.iloc[index]
        full_patient_id = f"ICC_{row['Patient_ID']}"
        vessel = row["Vessel"]
        label = self.label_map[row["Plaque_Type"]]

        feature_path = self.feature_dir / f"{full_patient_id}_{vessel}_features.npy"
        if feature_path.exists():
            sequence = torch.from_numpy(np.load(feature_path)).float()
        else:
            logger.warning("Feature file missing, using zero-padding fallback: %s", feature_path)
            sequence = torch.zeros((1, self.feature_dim))

        return sequence, torch.tensor(label, dtype=torch.long)


def sequence_collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor]]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad a batch of variable-length feature sequences to the same length.

    Parameters
    ----------
    batch:
        List of ``(sequence, label)`` tuples as produced by
        :class:`VesselSequenceDataset`.

    Returns
    -------
    tuple of torch.Tensor
        ``(padded_sequences, labels)`` where ``padded_sequences`` has
        shape ``(batch, max_len, feature_dim)``.
    """
    sequences, labels = zip(*batch)
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    return padded_sequences, torch.stack(labels)
