"""NIfTI file I/O helpers shared by segmentation, mapping, and extraction stages."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple

import nibabel as nib
import numpy as np
from nibabel.nifti1 import Nifti1Image

from coronet.exceptions import MissingInputError
from coronet.logging_config import get_logger

logger = get_logger(__name__)

_PATIENT_ID_PATTERN = re.compile(r"(?:ICC_Patient_|classified_coronary_tree_|full_classified_tree_)(\d+)")


def load_volume(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load a NIfTI volume and return its data array and affine matrix.

    Parameters
    ----------
    path:
        Path to a ``.nii`` or ``.nii.gz`` file.

    Returns
    -------
    tuple of numpy.ndarray
        ``(data, affine)`` where ``data`` is the raw voxel array (as
        returned by ``get_fdata``) and ``affine`` maps voxel indices to
        physical (mm) coordinates.

    Raises
    ------
    MissingInputError
        If ``path`` does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise MissingInputError(f"NIfTI file not found: {path}")

    image = nib.load(str(path))
    return image.get_fdata(), image.affine


def save_volume(data: np.ndarray, affine: np.ndarray, path: str | Path) -> None:
    """Save a voxel array as a NIfTI file, creating parent directories as needed.

    Parameters
    ----------
    data:
        Voxel data to save.
    affine:
        Affine transform mapping voxel indices to physical coordinates.
    path:
        Destination path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Nifti1Image(data, affine)
    nib.save(image, str(path))
    logger.debug("Saved NIfTI volume to %s", path)


def discover_patient_ids(directory: str | Path, glob_pattern: str) -> List[str]:
    """Scan a directory for files matching a glob pattern and extract patient IDs.

    Parameters
    ----------
    directory:
        Directory to scan.
    glob_pattern:
        Glob pattern (e.g. ``"full_classified_tree_*.nii.gz"``) used to
        select candidate files. The numeric patient ID is then parsed
        out of each matching filename.

    Returns
    -------
    list of str
        Sorted, de-duplicated list of patient ID strings.
    """
    directory = Path(directory)
    if not directory.exists():
        logger.warning("Directory does not exist, no patients discovered: %s", directory)
        return []

    patient_ids = set()
    for file_path in directory.glob(glob_pattern):
        match = _PATIENT_ID_PATTERN.search(file_path.name)
        if match:
            patient_ids.add(match.group(1))
    return sorted(patient_ids)


def normalize_patient_filename(filename: str) -> str:
    """Extract the canonical patient ID (e.g. ``"0181"``) from a raw filename.

    Parameters
    ----------
    filename:
        Filename such as ``"ICC_Patient_0181_0000.nii.gz"``.

    Returns
    -------
    str
        The patient ID without prefixes/suffixes, or the stem of the
        filename if no numeric ID could be parsed.
    """
    stem = filename.replace("_0000.nii.gz", "").replace(".nii.gz", "")
    match = re.search(r"(\d+)$", stem)
    return match.group(1) if match else stem
