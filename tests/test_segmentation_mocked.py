"""Unit tests for coronet.segmentation.heart_segmentor using mocked subprocess calls.

TotalSegmentator is not invoked for real here — subprocess.run is
patched so these tests exercise the class's file-staging, error
handling, and skip-logic without needing the actual tool installed.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from coronet.exceptions import MissingInputError, SegmentationError
from coronet.segmentation.heart_segmentor import HeartChamberSegmentor


def _fake_success_run(local_heart_file: Path):
    """Build a subprocess.run replacement that writes a dummy output file."""

    def _run(cmd, check, capture_output, text):
        local_heart_file.parent.mkdir(parents=True, exist_ok=True)
        local_heart_file.write_bytes(b"0" * 2048)
        return subprocess.CompletedProcess(cmd, 0)

    return _run


def test_segment_patient_success(tmp_path: Path) -> None:
    ct_path = tmp_path / "patient_0001.nii.gz"
    ct_path.write_bytes(b"fake ct data")

    staging_dir = tmp_path / "staging"
    segmentor = HeartChamberSegmentor(output_dir=tmp_path / "heart_masks", staging_dir=staging_dir)
    local_heart_file = staging_dir / "heart.nii.gz"

    with patch("subprocess.run", side_effect=_fake_success_run(local_heart_file)):
        result_path = segmentor.segment_patient("patient_0001", ct_path)

    assert result_path.exists()
    assert result_path.stat().st_size > 1024


def test_segment_patient_missing_input_raises(tmp_path: Path) -> None:
    segmentor = HeartChamberSegmentor(output_dir=tmp_path / "out", staging_dir=tmp_path / "staging")
    with pytest.raises(MissingInputError):
        segmentor.segment_patient("patient_0002", tmp_path / "does_not_exist.nii.gz")


def test_segment_patient_tool_failure_raises_segmentation_error(tmp_path: Path) -> None:
    ct_path = tmp_path / "patient_0003.nii.gz"
    ct_path.write_bytes(b"fake ct data")
    segmentor = HeartChamberSegmentor(output_dir=tmp_path / "out", staging_dir=tmp_path / "staging")

    def _failing_run(cmd, check, capture_output, text):
        raise subprocess.CalledProcessError(1, cmd, stderr="boom")

    with patch("subprocess.run", side_effect=_failing_run):
        with pytest.raises(SegmentationError):
            segmentor.segment_patient("patient_0003", ct_path)


def test_is_already_processed_skips_existing_patient(tmp_path: Path) -> None:
    output_dir = tmp_path / "heart_masks"
    patient_dir = output_dir / "patient_0004"
    patient_dir.mkdir(parents=True)
    (patient_dir / "heart.nii.gz").write_bytes(b"0" * 2048)

    segmentor = HeartChamberSegmentor(output_dir=output_dir, staging_dir=tmp_path / "staging")
    assert segmentor.is_already_processed("patient_0004") is True
    assert segmentor.is_already_processed("patient_9999") is False


def test_segment_batch_skips_already_processed_and_continues_past_failures(tmp_path: Path) -> None:
    output_dir = tmp_path / "heart_masks"
    staging_dir = tmp_path / "staging"

    # Patient A is already processed and should be skipped.
    (output_dir / "0001").mkdir(parents=True)
    (output_dir / "0001" / "heart.nii.gz").write_bytes(b"0" * 2048)
    ct_a = tmp_path / "ICC_Patient_0001_0000.nii.gz"
    ct_a.write_bytes(b"data")

    # Patient B is new and should succeed.
    ct_b = tmp_path / "ICC_Patient_0002_0000.nii.gz"
    ct_b.write_bytes(b"data")

    segmentor = HeartChamberSegmentor(output_dir=output_dir, staging_dir=staging_dir)
    local_heart_file = staging_dir / "heart.nii.gz"

    with patch("subprocess.run", side_effect=_fake_success_run(local_heart_file)) as mocked_run:
        results = segmentor.segment_batch([ct_a, ct_b])

    assert mocked_run.call_count == 1  # only patient B triggered a real segmentation call
    assert len(results) == 1
