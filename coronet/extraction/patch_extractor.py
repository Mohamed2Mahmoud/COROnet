"""Extraction of fixed-size 3D micro-patches centered on coronary centerline voxels.

Refactored from ``04_Coordinates_and_Extractor.ipynb`` (Phase 4: Master
CNN Micro-Patch Batch Extractor). Every centerline voxel belonging to a
given vessel yields one HU-normalized cubic patch from the raw CT; all
patches for a vessel are stacked into a single 4D NIfTI volume of shape
``(patch, patch, patch, N)``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import nibabel as nib
import numpy as np

from coronet.exceptions import MissingInputError, PatchExtractionError
from coronet.logging_config import get_logger
from coronet.utils.nifti_io import load_volume

logger = get_logger(__name__)

DEFAULT_VESSEL_LABELS: Tuple[Tuple[int, str], ...] = ((1, "RCA"), (2, "LAD"), (3, "LCX"))


class PatchExtractor:
    """Extracts cubic, HU-normalized patches around each vessel's centerline voxels.

    Parameters
    ----------
    patch_size:
        Edge length (in voxels) of the cubic patch.
    hu_clip_range:
        ``(min, max)`` Hounsfield Unit range used to clip intensities
        before min-max normalizing each patch independently.
    vessel_labels:
        Sequence of ``(label_value, folder_safe_name)`` pairs.
    """

    def __init__(
        self,
        patch_size: int = 32,
        hu_clip_range: Tuple[float, float] = (-100.0, 800.0),
        vessel_labels: Tuple[Tuple[int, str], ...] = DEFAULT_VESSEL_LABELS,
    ) -> None:
        self.patch_size = patch_size
        self.half_patch = patch_size // 2
        self.hu_clip_range = hu_clip_range
        self.vessel_labels = vessel_labels

    def _extract_patch(self, x: int, y: int, z: int, raw_volume: np.ndarray) -> Optional[np.ndarray]:
        """Extract and normalize one cubic patch centered at ``(x, y, z)``, or None if out of bounds."""
        half = self.half_patch
        if not (
            half <= x < raw_volume.shape[0] - half
            and half <= y < raw_volume.shape[1] - half
            and half <= z < raw_volume.shape[2] - half
        ):
            return None

        patch = raw_volume[x - half:x + half, y - half:y + half, z - half:z + half]
        low, high = self.hu_clip_range
        patch = np.clip(patch, low, high)
        patch_range = patch.max() - patch.min() + 1e-7
        patch = (patch - patch.min()) / patch_range
        return patch.astype(np.float32)

    def _save_patches(self, coords: np.ndarray, raw_volume: np.ndarray, affine: np.ndarray, output_path: Path) -> int:
        """Extract patches for every coordinate and stack them into a 4D NIfTI file."""
        patches = []
        for x, y, z in coords:
            patch = self._extract_patch(int(x), int(y), int(z), raw_volume)
            if patch is not None:
                patches.append(patch)

        if not patches:
            return 0

        combined = np.stack(patches, axis=-1)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(combined, affine), str(output_path))
        return len(patches)

    def extract_for_patient(
        self,
        classified_tree_path: str | Path,
        raw_ct_path: str | Path,
        patient_output_dir: str | Path,
    ) -> Dict[str, int]:
        """Extract and save patches for every vessel of a single patient.

        Parameters
        ----------
        classified_tree_path:
            Path to the anatomically classified (RCA/LAD/LCx) centerline
            volume.
        raw_ct_path:
            Path to the corresponding (cropped) raw CT volume.
        patient_output_dir:
            Directory where ``RCA.nii.gz``, ``LAD.nii.gz``, and
            ``LCX.nii.gz`` will be written.

        Returns
        -------
        dict of str to int
            Number of patches successfully extracted per vessel name.

        Raises
        ------
        MissingInputError
            If either input volume is missing.
        PatchExtractionError
            If no patches could be extracted for any vessel.
        """
        tree_data, _ = load_volume(classified_tree_path)
        raw_data, raw_affine = load_volume(raw_ct_path)

        patient_output_dir = Path(patient_output_dir)
        patient_output_dir.mkdir(parents=True, exist_ok=True)

        patch_counts: Dict[str, int] = {}
        for label_value, vessel_name in self.vessel_labels:
            coords = np.argwhere(tree_data == label_value)
            output_path = patient_output_dir / f"{vessel_name}.nii.gz"
            count = self._save_patches(coords, raw_data, raw_affine, output_path)
            patch_counts[vessel_name] = count
            logger.info("Extracted %d patches for vessel %s", count, vessel_name)

        if sum(patch_counts.values()) == 0:
            raise PatchExtractionError(
                f"No patches could be extracted for any vessel from {classified_tree_path}"
            )

        return patch_counts
