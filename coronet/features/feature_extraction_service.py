"""Multi-backbone 3D feature extraction service.

Refactored from ``05B_Feature_Extractor.ipynb`` (Phase 5, Step 2). Each
patient/vessel's stacked 4D patch volume (``(32, 32, 32, N)``) is passed
through every configured CNN backbone to produce a per-patch 512-d
feature vector, saved as a ``.npy`` sequence of shape ``(N, 512)``.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Dict, List

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn

from coronet.exceptions import MissingInputError
from coronet.logging_config import get_logger
from coronet.models.architectures import FeatureExtractorFactory
from coronet.utils.memory import clear_gpu_memory

logger = get_logger(__name__)


class VesselRecord:
    """A single patient/vessel unit of work for feature extraction.

    Parameters
    ----------
    patient_id:
        Full clinical patient identifier (e.g. ``"ICC_Patient_0001"``).
    vessel_name:
        Vessel name (``"RCA"``, ``"LAD"``, or ``"LCX"``).
    patch_volume_path:
        Path to the stacked 4D patch NIfTI file for this vessel.
    """

    __slots__ = ("patient_id", "vessel_name", "patch_volume_path")

    def __init__(self, patient_id: str, vessel_name: str, patch_volume_path: Path) -> None:
        self.patient_id = patient_id
        self.vessel_name = vessel_name
        self.patch_volume_path = patch_volume_path


class FeatureExtractionService:
    """Extracts 1D feature sequences for every patient/vessel using multiple CNN backbones.

    Parameters
    ----------
    patches_dir:
        Directory containing per-patient sub-directories (e.g.
        ``Patient_0001/RCA.nii.gz``).
    features_dir:
        Root output directory; one sub-directory is created per
        backbone name.
    device:
        Torch device to run inference on.
    feature_dim:
        Output dimensionality of every backbone.
    batch_size:
        Number of patches processed per forward pass.
    """

    def __init__(
        self,
        patches_dir: str | Path,
        features_dir: str | Path,
        device: torch.device,
        feature_dim: int = 512,
        batch_size: int = 16,
    ) -> None:
        self.patches_dir = Path(patches_dir)
        self.features_dir = Path(features_dir)
        self.device = device
        self.feature_dim = feature_dim
        self.batch_size = batch_size
        self.factory = FeatureExtractorFactory(feature_dim=feature_dim)

    def discover_records(
        self, annotation_patients: Dict, vessel_names: tuple = ("RCA", "LAD", "LCX")
    ) -> List[VesselRecord]:
        """Match patient patch folders against annotated clinical records.

        Parameters
        ----------
        annotation_patients:
            The ``"patients"`` mapping from the annotation JSON, keyed
            by full patient id (e.g. ``"ICC_Patient_0001"``).
        vessel_names:
            Vessel names to look for within each patient folder.

        Returns
        -------
        list of VesselRecord
            One record per existing, annotated vessel patch volume.
        """
        records: List[VesselRecord] = []
        patient_folders = sorted(self.patches_dir.glob("Patient_*"))

        for folder in patient_folders:
            pid_number = folder.name.split("_")[1]
            json_pid = f"ICC_Patient_{pid_number}"

            if json_pid not in annotation_patients:
                continue

            for vessel_name in vessel_names:
                nii_path = folder / f"{vessel_name}.nii.gz"
                if nii_path.exists():
                    records.append(VesselRecord(json_pid, vessel_name, nii_path))

        logger.info("Discovered %d annotated vessel patch volumes.", len(records))
        return records

    def _extract_sequence(self, model: nn.Module, record: VesselRecord) -> np.ndarray:
        """Run a single vessel's stacked patches through one backbone."""
        nii_img = nib.load(str(record.patch_volume_path))
        patches = nii_img.get_fdata()  # (32, 32, 32, N)
        patches = np.transpose(patches, (3, 0, 1, 2))  # (N, 32, 32, 32)

        all_features = []
        with torch.no_grad():
            for start in range(0, len(patches), self.batch_size):
                batch = patches[start:start + self.batch_size]
                batch_tensor = torch.tensor(batch.copy()).float().unsqueeze(1).to(self.device)
                features = model(batch_tensor)
                all_features.append(features.cpu().numpy())

        del patches
        return np.concatenate(all_features, axis=0) if all_features else np.empty((0, self.feature_dim))

    def run_backbone(self, backbone_name: str, records: List[VesselRecord]) -> int:
        """Extract and save feature sequences for every record using one backbone.

        Parameters
        ----------
        backbone_name:
            Name accepted by :class:`FeatureExtractorFactory`.
        records:
            Vessel records to process, typically from
            :meth:`discover_records`.

        Returns
        -------
        int
            Number of newly extracted (non-cached) feature sequences.

        Raises
        ------
        MissingInputError
            If ``records`` is empty.
        """
        if not records:
            raise MissingInputError("No vessel records were provided for feature extraction.")

        model = self.factory.build(backbone_name).to(self.device)
        model.eval()

        model_features_dir = self.features_dir / backbone_name
        model_features_dir.mkdir(parents=True, exist_ok=True)

        extracted_count = 0
        for index, record in enumerate(records, start=1):
            save_path = model_features_dir / f"{record.patient_id}_{record.vessel_name}_features.npy"
            if save_path.exists():
                logger.debug("[%d/%d] %s already exists; skipping.", index, len(records), save_path.name)
                continue

            try:
                sequence = self._extract_sequence(model, record)
                np.save(save_path, sequence)
                extracted_count += 1
                logger.info(
                    "[%d/%d] %s: extracted %s -> %d patches.",
                    index, len(records), backbone_name, record.patch_volume_path.name, len(sequence),
                )
            except (OSError, ValueError) as exc:
                logger.error("Failed to extract features for %s %s: %s", record.patient_id, record.vessel_name, exc)
                continue
            finally:
                gc.collect()

        model.cpu()
        del model
        clear_gpu_memory()

        return extracted_count

    def run_all_backbones(self, records: List[VesselRecord], backbone_names: tuple = None) -> Dict[str, int]:
        """Run feature extraction for every configured backbone, sequentially.

        Parameters
        ----------
        records:
            Vessel records to process.
        backbone_names:
            Backbones to run; defaults to all four supported backbones.

        Returns
        -------
        dict of str to int
            Number of newly extracted sequences per backbone.
        """
        backbone_names = backbone_names or ("CNN", "DenseNet", "ResNet", "EfficientNet")
        results = {}
        for backbone_name in backbone_names:
            logger.info("Starting extraction for backbone: %s", backbone_name)
            results[backbone_name] = self.run_backbone(backbone_name, records)
        return results
