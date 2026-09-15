"""Data ingestion: discovering raw inputs and loading clinical metadata.

Consolidates the file-discovery and annotation-loading logic that was
scattered across nearly every notebook cell (``glob.glob`` calls,
``json.load`` for ``annotation.json``, and ``pandas.read_csv`` /
``read_excel`` for split files) into a single, reusable component.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from coronet.exceptions import MissingInputError
from coronet.logging_config import get_logger

logger = get_logger(__name__)


class DataIngestor:
    """Discovers raw CT volumes and loads clinical annotation/split metadata.

    Parameters
    ----------
    root:
        Root data directory (typically ``PathConfig.root``).
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def list_raw_ct_volumes(self, raw_ct_dir: str | Path) -> List[Path]:
        """List all raw CT NIfTI volumes in a directory, sorted by filename.

        Parameters
        ----------
        raw_ct_dir:
            Directory containing ``*.nii.gz`` CT volumes.

        Returns
        -------
        list of pathlib.Path
            Sorted list of matching file paths. Empty if the directory
            does not exist.
        """
        raw_ct_dir = Path(raw_ct_dir)
        if not raw_ct_dir.exists():
            logger.warning("Raw CT directory does not exist: %s", raw_ct_dir)
            return []
        return sorted(raw_ct_dir.glob("*.nii.gz"))

    @staticmethod
    def load_annotation_json(annotation_path: str | Path) -> Dict[str, Any]:
        """Load the clinical annotation file mapping patients to vessel status/plaque type.

        Parameters
        ----------
        annotation_path:
            Path to a JSON file with a top-level ``"patients"`` key.

        Returns
        -------
        dict
            The ``"patients"`` mapping: ``{patient_id: {"vessels": {...}}}``.

        Raises
        ------
        MissingInputError
            If ``annotation_path`` does not exist.
        """
        annotation_path = Path(annotation_path)
        if not annotation_path.exists():
            raise MissingInputError(f"Annotation file not found: {annotation_path}")

        with annotation_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload["patients"]

    @staticmethod
    def load_split_table(split_path: str | Path) -> pd.DataFrame:
        """Load a train/test split table from CSV or Excel.

        Parameters
        ----------
        split_path:
            Path to a ``.csv`` or ``.xlsx``/``.xls`` file describing the
            dataset split.

        Returns
        -------
        pandas.DataFrame
            The loaded table.

        Raises
        ------
        MissingInputError
            If ``split_path`` does not exist.
        ValueError
            If the file extension is not recognized.
        """
        split_path = Path(split_path)
        if not split_path.exists():
            raise MissingInputError(f"Split file not found: {split_path}")

        suffix = split_path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(split_path, keep_default_na=False)
        if suffix in (".xlsx", ".xls"):
            return pd.read_excel(split_path)
        raise ValueError(f"Unsupported split file extension: {suffix}")
