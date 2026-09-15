"""Data preprocessing: stratified dataset splitting and image transform pipelines.

Refactored from the dataset-split cell of ``05B_Feature_Extractor.ipynb``
and the MONAI transform pipelines duplicated across every cell of
``05A_resNet75.ipynb``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from monai.transforms import (
    Compose,
    LoadImaged,
    MapTransform,
    RandGaussianNoised,
    RandRotated,
    RandZoomd,
    Resized,
    ScaleIntensityd,
)
from sklearn.model_selection import train_test_split

from coronet.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_PLAQUE_LABEL_MAP: Dict[str, int] = {"None": 0, "Calcified": 1, "Soft": 2, "Mixed": 3}


class EnsureSingleChannel3D(MapTransform):
    """MONAI transform that squeezes and re-expands a volume to a single leading channel.

    Some source NIfTI patches carry stray singleton or multi-channel
    dimensions; this transform guarantees a clean ``(1, D, H, W)`` shape
    regardless of the original layout.
    """

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        result = dict(data)
        for key in self.keys:
            volume = result[key].squeeze()
            if len(volume.shape) > 3:
                volume = volume[0]
            result[key] = volume.unsqueeze(0)
        return result


class VesselTransformFactory:
    """Builds MONAI transform pipelines for the binary vessel classifier.

    Parameters
    ----------
    spatial_size:
        Target ``(D, H, W)`` size after resizing.
    """

    def __init__(self, spatial_size: Tuple[int, int, int] = (96, 96, 96)) -> None:
        self.spatial_size = spatial_size

    def training_transforms(self) -> Compose:
        """Return the augmented transform pipeline used during training.

        Returns
        -------
        monai.transforms.Compose
            Loads the image, ensures a single channel, scales intensity
            to ``[0, 1]``, resizes, and applies safe 3D medical
            augmentations (rotation, Gaussian noise, zoom).
        """
        return Compose(
            [
                LoadImaged(keys=["image"]),
                EnsureSingleChannel3D(keys=["image"]),
                ScaleIntensityd(keys=["image"]),
                Resized(keys=["image"], spatial_size=self.spatial_size),
                RandRotated(keys=["image"], range_x=0.2, range_y=0.2, range_z=0.2, prob=0.5),
                RandGaussianNoised(keys=["image"], prob=0.2),
                RandZoomd(keys=["image"], min_zoom=0.9, max_zoom=1.1, prob=0.3),
            ]
        )

    def evaluation_transforms(self) -> Compose:
        """Return the deterministic transform pipeline used for validation/testing.

        Returns
        -------
        monai.transforms.Compose
            Same as :meth:`training_transforms` without any random
            augmentation.
        """
        return Compose(
            [
                LoadImaged(keys=["image"]),
                EnsureSingleChannel3D(keys=["image"]),
                ScaleIntensityd(keys=["image"]),
                Resized(keys=["image"], spatial_size=self.spatial_size),
            ]
        )


class DatasetSplitter:
    """Builds a stratified train/test split for the plaque-type classification task.

    Refactored from the dataset-split cell of
    ``05B_Feature_Extractor.ipynb``.

    Parameters
    ----------
    label_map:
        Mapping from plaque-type string to integer class id.
    test_size:
        Fraction of samples held out for testing.
    random_state:
        Seed controlling the stratified split.
    """

    def __init__(
        self,
        label_map: Dict[str, int] = None,
        test_size: float = 0.30,
        random_state: int = 42,
    ) -> None:
        self.label_map = label_map or dict(DEFAULT_PLAQUE_LABEL_MAP)
        self.test_size = test_size
        self.random_state = random_state

    def build_split(self, annotation_patients: Dict[str, Any]) -> pd.DataFrame:
        """Build a stratified train/test split table from annotation metadata.

        Parameters
        ----------
        annotation_patients:
            The ``"patients"`` mapping loaded via
            :meth:`coronet.data.ingestion.DataIngestor.load_annotation_json`.

        Returns
        -------
        pandas.DataFrame
            Columns: ``Patient_ID``, ``Vessel``, ``Plaque_Type``,
            ``Label``, ``Split`` (``"Train"`` or ``"Test"``).
        """
        samples: List[Dict[str, Any]] = []
        labels: List[int] = []

        for patient_id, patient_info in annotation_patients.items():
            clean_id = patient_id.replace("ICC_", "")
            for vessel_name, vessel_info in patient_info.get("vessels", {}).items():
                plaque_type = vessel_info["plaque_type"]
                samples.append(
                    {
                        "Patient_ID": clean_id,
                        "Vessel": vessel_name,
                        "Plaque_Type": plaque_type,
                        "Label": self.label_map[plaque_type],
                    }
                )
                labels.append(self.label_map[plaque_type])

        train_samples, test_samples, _, _ = train_test_split(
            samples, labels, test_size=self.test_size, stratify=labels, random_state=self.random_state
        )

        train_df = pd.DataFrame(train_samples)
        train_df["Split"] = "Train"
        test_df = pd.DataFrame(test_samples)
        test_df["Split"] = "Test"

        combined = pd.concat([train_df, test_df], ignore_index=True)
        logger.info("Built stratified split: %d train, %d test.", len(train_df), len(test_df))
        return combined

    def save_split(self, split_df: pd.DataFrame, output_path: str | Path) -> Path:
        """Persist a split table to CSV.

        Parameters
        ----------
        split_df:
            The split table, as returned by :meth:`build_split`.
        output_path:
            Destination CSV path.

        Returns
        -------
        pathlib.Path
            Path the table was written to.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        split_df.to_csv(output_path, index=False)
        return output_path
