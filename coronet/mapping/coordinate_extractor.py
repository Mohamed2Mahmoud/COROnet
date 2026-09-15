"""Extraction of per-vessel voxel and physical (mm) coordinates.

Refactored from the coordinate-extraction cell of
``04_Coordinates_and_Extractor.ipynb``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from nibabel.affines import apply_affine

from coronet.exceptions import MissingInputError
from coronet.logging_config import get_logger
from coronet.utils.nifti_io import load_volume

logger = get_logger(__name__)

DEFAULT_VESSEL_LABELS: Tuple[Tuple[int, str], ...] = ((1, "RCA"), (2, "LAD"), (3, "LCx"))


class CoordinateExtractor:
    """Extracts voxel and millimeter coordinates for each labelled vessel.

    Parameters
    ----------
    vessel_labels:
        Sequence of ``(label_value, label_name)`` pairs identifying
        which voxel values correspond to which named vessel.
    """

    def __init__(self, vessel_labels: Tuple[Tuple[int, str], ...] = DEFAULT_VESSEL_LABELS) -> None:
        self.vessel_labels = vessel_labels

    def extract(self, classified_tree_path: str | Path, output_json_path: str | Path) -> Path:
        """Extract voxel/mm coordinates for every vessel label and save as JSON.

        Parameters
        ----------
        classified_tree_path:
            Path to a 3-class (or N-class) anatomically classified
            coronary tree.
        output_json_path:
            Destination path for the coordinates JSON file.

        Returns
        -------
        pathlib.Path
            Path to the saved JSON file.

        Raises
        ------
        MissingInputError
            If ``classified_tree_path`` does not exist.
        """
        data, affine = load_volume(classified_tree_path)

        coords_dict: Dict[str, Dict[str, List[List[float]]]] = {}
        for label_value, label_name in self.vessel_labels:
            voxel_coords = np.argwhere(data == label_value)

            if voxel_coords.size > 0:
                mm_coords = apply_affine(affine, voxel_coords)
                coords_dict[label_name] = {
                    "voxel_coordinates": voxel_coords.tolist(),
                    "mm_coordinates": mm_coords.tolist(),
                }
                logger.info("%s: found %d centerline points.", label_name, len(voxel_coords))
            else:
                coords_dict[label_name] = {"voxel_coordinates": [], "mm_coordinates": []}
                logger.info("%s: found 0 centerline points.", label_name)

        output_json_path = Path(output_json_path)
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        with output_json_path.open("w", encoding="utf-8") as handle:
            json.dump(coords_dict, handle)

        return output_json_path
