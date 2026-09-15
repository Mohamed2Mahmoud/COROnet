"""Unit tests for coronet.extraction.patch_extractor using synthetic NIfTI volumes."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from coronet.exceptions import PatchExtractionError
from coronet.extraction.patch_extractor import PatchExtractor


@pytest.fixture
def synthetic_volumes(tmp_path: Path) -> tuple[Path, Path]:
    shape = (64, 64, 64)
    affine = np.eye(4)

    raw_data = np.random.uniform(-200, 1000, size=shape).astype(np.float32)
    raw_path = tmp_path / "raw_ct.nii.gz"
    nib.save(nib.Nifti1Image(raw_data, affine), str(raw_path))

    tree_data = np.zeros(shape, dtype=np.uint8)
    tree_data[32, 32, 32] = 1  # a single RCA centerline voxel, safely inside patch bounds
    tree_data[10, 10, 10] = 2  # a single LAD centerline voxel
    tree_path = tmp_path / "tree.nii.gz"
    nib.save(nib.Nifti1Image(tree_data, affine), str(tree_path))

    return raw_path, tree_path


def test_extract_for_patient_produces_expected_vessel_files(tmp_path: Path, synthetic_volumes) -> None:
    raw_path, tree_path = synthetic_volumes
    output_dir = tmp_path / "patient_0001"

    extractor = PatchExtractor(patch_size=16)
    counts = extractor.extract_for_patient(tree_path, raw_path, output_dir)

    assert counts["RCA"] == 1
    assert counts["LAD"] == 1
    assert counts["LCX"] == 0
    assert (output_dir / "RCA.nii.gz").exists()
    assert (output_dir / "LAD.nii.gz").exists()


def test_extract_for_patient_raises_when_no_patches_found(tmp_path: Path) -> None:
    shape = (64, 64, 64)
    affine = np.eye(4)

    empty_tree = np.zeros(shape, dtype=np.uint8)
    tree_path = tmp_path / "empty_tree.nii.gz"
    nib.save(nib.Nifti1Image(empty_tree, affine), str(tree_path))

    raw_data = np.zeros(shape, dtype=np.float32)
    raw_path = tmp_path / "raw.nii.gz"
    nib.save(nib.Nifti1Image(raw_data, affine), str(raw_path))

    extractor = PatchExtractor(patch_size=16)
    with pytest.raises(PatchExtractionError):
        extractor.extract_for_patient(tree_path, raw_path, tmp_path / "out")
