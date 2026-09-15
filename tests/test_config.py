"""Unit tests for coronet.config."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from coronet.config import CoronetConfig


def test_path_config_derives_expected_subdirectories(tmp_path: Path) -> None:
    config = CoronetConfig(root=tmp_path)

    assert config.paths.raw_ct_dir == tmp_path / "ICC_Data"
    assert config.paths.patches_dir == tmp_path / "Classification_Input"
    assert config.paths.annotation_json == tmp_path / "annotation.json"


def test_ensure_dirs_creates_output_directories(tmp_path: Path) -> None:
    config = CoronetConfig(root=tmp_path)
    config.paths.ensure_dirs()

    assert config.paths.heart_mask_dir.exists()
    assert config.paths.patches_dir.exists()
    assert config.paths.results_dir.exists()


def test_round_trip_json_serialization(tmp_path: Path) -> None:
    config = CoronetConfig(root=tmp_path)
    config.binary_classifier.epochs = 5
    config.sequence_classifier.architectures = ("bilstm",)

    json_path = tmp_path / "config.json"
    config.to_json(json_path)

    reloaded = CoronetConfig.from_json(json_path)

    assert reloaded.root == str(tmp_path) or Path(reloaded.root) == tmp_path
    assert reloaded.binary_classifier.epochs == 5
    assert reloaded.sequence_classifier.architectures == ["bilstm"] or list(
        reloaded.sequence_classifier.architectures
    ) == ["bilstm"]


def test_from_json_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        CoronetConfig.from_json(tmp_path / "does_not_exist.json")
