"""Post-processing steps applied to segmentation masks.

Refactored from cells in ``02_MMBAA_Coronary_Mapping.ipynb``:
heart-ROI cropping of the raw CT (Phase 1.5) and skeletonization /
re-thickening of the raw coronary artery mask.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from scipy.ndimage import binary_dilation
from skimage.morphology import ball, skeletonize

from coronet.exceptions import AnatomicalMappingError, MissingInputError
from coronet.logging_config import get_logger
from coronet.utils.nifti_io import load_volume, save_volume

logger = get_logger(__name__)


class HeartROICropper:
    """Crops raw CT volumes to a padded bounding box around target heart chambers.

    Parameters
    ----------
    target_labels:
        Voxel labels in the heart-chamber mask that define the region
        of interest (e.g. ``(1, 2, 3, 4)`` for LA/LV/RA/RV).
    padding_voxels:
        Isotropic padding applied around the bounding box before
        cropping.
    """

    def __init__(self, target_labels: Sequence[int] = (1, 2, 3, 4), padding_voxels: int = 25) -> None:
        self.target_labels = tuple(target_labels)
        self.padding_voxels = padding_voxels

    def crop(self, raw_ct_path: str | Path, heart_mask_path: str | Path, output_path: str | Path) -> Path:
        """Crop a raw CT volume to the padded bounding box of the target chambers.

        Parameters
        ----------
        raw_ct_path:
            Path to the full-resolution raw CT volume.
        heart_mask_path:
            Path to the corresponding multi-label heart-chamber mask.
        output_path:
            Destination path for the cropped CT volume.

        Returns
        -------
        pathlib.Path
            The path the cropped volume was written to.

        Raises
        ------
        MissingInputError
            If either input file is missing.
        AnatomicalMappingError
            If none of the target chamber labels are present in the mask.
        """
        mask_data, mask_affine = load_volume(heart_mask_path)
        raw_data, raw_affine = load_volume(raw_ct_path)

        chamber_mask = np.isin(mask_data, self.target_labels)
        coords = np.argwhere(chamber_mask)
        if coords.size == 0:
            raise AnatomicalMappingError(
                f"No voxels found for target chamber labels {self.target_labels} in {heart_mask_path}"
            )

        min_bound = coords.min(axis=0)
        max_bound = coords.max(axis=0)
        shape = raw_data.shape

        min_bound = np.maximum(0, min_bound - self.padding_voxels)
        max_bound = np.minimum(shape, max_bound + self.padding_voxels)

        cropped = raw_data[
            min_bound[0]:max_bound[0],
            min_bound[1]:max_bound[1],
            min_bound[2]:max_bound[2],
        ]
        save_volume(cropped, raw_affine, output_path)
        logger.info("Cropped %s to shape %s", Path(raw_ct_path).name, cropped.shape)
        return Path(output_path)


class VesselSkeletonizer:
    """Reduces a binary vessel mask to its 1-voxel centerline and re-thickens it.

    Parameters
    ----------
    thickness_radius:
        Radius (in voxels) of the spherical structuring element used to
        re-thicken the skeleton into a uniform-diameter tube.
    """

    def __init__(self, thickness_radius: int = 6) -> None:
        self.thickness_radius = thickness_radius

    def skeletonize_and_thicken(
        self,
        mask_path: str | Path,
        skeleton_output_path: str | Path,
        thickened_output_path: str | Path,
    ) -> Tuple[Path, Path]:
        """Skeletonize a binary mask and reconstruct a uniform-thickness tube.

        Parameters
        ----------
        mask_path:
            Path to the raw (binary or near-binary) coronary artery
            segmentation mask.
        skeleton_output_path:
            Destination path for the 1-voxel-wide skeleton.
        thickened_output_path:
            Destination path for the re-thickened tube.

        Returns
        -------
        tuple of pathlib.Path
            ``(skeleton_output_path, thickened_output_path)``.
        """
        data, affine = load_volume(mask_path)
        binary_mask = (data > 0).astype(np.uint8)

        skeleton = skeletonize(binary_mask)
        structuring_element = ball(self.thickness_radius)
        thickened = binary_dilation(skeleton, structure=structuring_element)

        save_volume(skeleton.astype(np.uint8), affine, skeleton_output_path)
        save_volume(thickened.astype(np.uint8), affine, thickened_output_path)
        logger.info("Skeletonized and thickened %s", Path(mask_path).name)

        return Path(skeleton_output_path), Path(thickened_output_path)

    @staticmethod
    def skeletonize_only(mask_path: str | Path) -> np.ndarray:
        """Return the binary skeleton of a mask without writing it to disk.

        Parameters
        ----------
        mask_path:
            Path to the binary/near-binary mask to skeletonize.

        Returns
        -------
        numpy.ndarray
            Boolean skeleton array with the same shape as the input.
        """
        data, _ = load_volume(mask_path)
        return skeletonize((data > 0).astype(np.uint8))


class ClassifiedTreeThickener:
    """Re-thickens a 1-voxel, anatomically labelled coronary tree for visualization.

    Each class is dilated independently and then painted onto a shared
    canvas with a fixed priority order (earlier labels win at overlaps),
    matching the "inflate the wire into a pipe" post-processing used in
    the original notebooks for both the 2-class (RCA/Left) and 3-class
    (RCA/LAD/LCx) classified trees.

    Parameters
    ----------
    dilation_iterations:
        Number of binary-dilation iterations applied to each class.
    label_priority:
        Ordered sequence of voxel labels, highest priority first. The
        first label in this sequence always wins where dilated regions
        overlap.
    """

    def __init__(self, dilation_iterations: int = 2, label_priority: Sequence[int] = (1, 2, 3)) -> None:
        self.dilation_iterations = dilation_iterations
        self.label_priority = tuple(label_priority)

    def thicken(self, classified_tree_path: str | Path, output_path: str | Path) -> Path:
        """Dilate each labelled class and composite them by priority.

        Parameters
        ----------
        classified_tree_path:
            Path to a 1-voxel classified coronary tree (integer labels).
        output_path:
            Destination path for the thickened, multi-label volume.

        Returns
        -------
        pathlib.Path
            Path to the saved thickened volume.
        """
        data, affine = load_volume(classified_tree_path)
        thick_canvas = np.zeros_like(data, dtype=np.uint8)
        already_claimed = np.zeros_like(data, dtype=bool)

        for class_label in self.label_priority:
            dilated = binary_dilation(data == class_label, iterations=self.dilation_iterations)
            eligible = dilated & ~already_claimed
            thick_canvas[eligible] = class_label
            already_claimed |= eligible

        save_volume(thick_canvas, affine, output_path)
        logger.info("Thickened classified tree saved to %s", output_path)
        return Path(output_path)
