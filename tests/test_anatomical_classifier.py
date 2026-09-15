"""Unit tests for coronet.mapping.anatomical_classifier using synthetic volumes.

These tests construct small, fully synthetic heart-chamber masks and
coronary skeletons with a known "correct" anatomical answer, so the
distance-transform voting logic in CoronaryTreeClassifier can be
verified without any real patient data.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from coronet.exceptions import AnatomicalMappingError
from coronet.mapping.anatomical_classifier import CoronaryTreeClassifier


def _save(data: np.ndarray, path: Path) -> None:
    nib.save(nib.Nifti1Image(data.astype(np.uint8), np.eye(4)), str(path))


@pytest.fixture
def rca_left_scenario(tmp_path: Path) -> tuple[Path, Path]:
    """Two well-separated branches: one near RA, one near LA/LV."""
    shape = (40, 40, 40)

    chamber = np.zeros(shape, dtype=np.uint8)
    chamber[0:15, :, :] = 3  # RA region (stage-1 convention: RA=3)
    chamber[25:40, :, 0:20] = 1  # LA region
    chamber[25:40, :, 20:40] = 2  # LV region
    chamber_path = tmp_path / "chamber.nii.gz"
    _save(chamber, chamber_path)

    artery = np.zeros(shape, dtype=np.uint8)
    artery[5, 10:16, 10] = 1  # branch A: near RA -> should become RCA (1)
    artery[32, 10:16, 10] = 1  # branch B: near LA/LV -> should become Left System (2)
    artery_path = tmp_path / "artery.nii.gz"
    _save(artery, artery_path)

    return artery_path, chamber_path


def test_classify_rca_vs_left_assigns_branches_by_proximity(tmp_path: Path, rca_left_scenario) -> None:
    artery_path, chamber_path = rca_left_scenario
    output_path = tmp_path / "two_class.nii.gz"

    classifier = CoronaryTreeClassifier()
    classifier.classify_rca_vs_left(artery_path, chamber_path, output_path)

    result = nib.load(str(output_path)).get_fdata()
    assert result[5, 12, 10] == 1  # branch near RA -> RCA
    assert result[32, 12, 10] == 2  # branch near LA/LV -> Left System


def test_classify_rca_vs_left_raises_on_empty_inputs(tmp_path: Path) -> None:
    shape = (10, 10, 10)
    empty_chamber_path = tmp_path / "empty_chamber.nii.gz"
    empty_artery_path = tmp_path / "empty_artery.nii.gz"
    _save(np.zeros(shape, dtype=np.uint8), empty_chamber_path)
    _save(np.zeros(shape, dtype=np.uint8), empty_artery_path)

    classifier = CoronaryTreeClassifier()
    with pytest.raises(AnatomicalMappingError):
        classifier.classify_rca_vs_left(empty_artery_path, empty_chamber_path, tmp_path / "out.nii.gz")


@pytest.fixture
def lad_lcx_scenario(tmp_path: Path) -> tuple[Path, Path]:
    """A 2-class tree with one RCA voxel plus two well-separated left-system clusters."""
    shape = (40, 40, 40)

    # Stage-2 chamber convention: LA=2, RA=4, RV=5.
    chamber = np.zeros(shape, dtype=np.uint8)
    chamber[0:10, :, :] = 2  # LA region
    chamber[30:40, :, :] = 4  # RA/RV (anterior) region
    chamber[30:40, :, 20:40] = 5
    chamber_path = tmp_path / "chamber2.nii.gz"
    _save(chamber, chamber_path)

    two_class = np.zeros(shape, dtype=np.uint8)
    two_class[2, 2, 2] = 1  # a lone RCA voxel that must be preserved untouched
    two_class[5, 5:11, 5] = 2  # cluster near LA -> should become LCx (3)
    two_class[35, 5:11, 5] = 2  # cluster near RA/RV, far from LA -> should become LAD (2)
    two_class_path = tmp_path / "two_class_full.nii.gz"
    _save(two_class, two_class_path)

    return two_class_path, chamber_path


def test_split_lad_lcx_preserves_rca_and_splits_left_system(tmp_path: Path, lad_lcx_scenario) -> None:
    two_class_path, chamber_path = lad_lcx_scenario
    output_path = tmp_path / "three_class.nii.gz"

    classifier = CoronaryTreeClassifier(lcx_bias_mm=5.0, lad_anchor_bias_mm=15.0)
    classifier.split_lad_lcx(two_class_path, chamber_path, output_path)

    result = nib.load(str(output_path)).get_fdata()

    assert result[2, 2, 2] == 1  # RCA voxel untouched
    assert result[5, 7, 5] == 3  # cluster near LA -> LCx
    assert result[35, 7, 5] == 2  # cluster far from LA, near RA/RV -> LAD


def test_split_lad_lcx_raises_when_no_left_system_voxels(tmp_path: Path, lad_lcx_scenario) -> None:
    _, chamber_path = lad_lcx_scenario
    only_rca = np.zeros((40, 40, 40), dtype=np.uint8)
    only_rca[2, 2, 2] = 1
    only_rca_path = tmp_path / "only_rca.nii.gz"
    _save(only_rca, only_rca_path)

    classifier = CoronaryTreeClassifier()
    with pytest.raises(AnatomicalMappingError):
        classifier.split_lad_lcx(only_rca_path, chamber_path, tmp_path / "out.nii.gz")
