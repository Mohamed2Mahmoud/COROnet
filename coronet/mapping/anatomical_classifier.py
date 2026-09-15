"""Anatomical labelling of the coronary tree (RCA / LAD / LCx).

Refactored from ``03_Coronary_Classification.ipynb``. The classification
proceeds in two stages, each built around Euclidean distance transforms
of the heart-chamber mask:

1. :meth:`CoronaryTreeClassifier.classify_rca_vs_left` groups the
   coronary skeleton into connected branches and assigns each whole
   branch to either the right coronary artery (RCA) or the "left
   system" (future LAD/LCx) by a per-branch majority vote of
   chamber-proximity at every skeleton voxel.
2. :meth:`CoronaryTreeClassifier.split_lad_lcx` further splits the left
   system into LAD and LCx using a biased provisional per-voxel vote
   followed by a nearest-core anchoring step (k-d tree) that
   consolidates fragmented tips onto the two largest connected
   components.

Voxel labels used throughout: ``1=RCA``, ``2=LAD`` (or "left system" in
the two-class stage), ``3=LCx``. Heart-chamber labels: ``1=LA``,
``2=LV``, ``3=RA``... as produced by TotalSegmentator's
``heartchambers_highres`` task (``2=LA``, ``3=LV``, ``4=RA``, ``5=RV``
in the chamber mask's own labelling used for stage 2).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import numpy.linalg as npl
from nibabel.affines import apply_affine
from scipy.ndimage import distance_transform_edt, label
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

from coronet.exceptions import AnatomicalMappingError
from coronet.logging_config import get_logger
from coronet.utils.nifti_io import load_volume, save_volume

logger = get_logger(__name__)

_CONNECTIVITY_26 = np.ones((3, 3, 3))


class CoronaryTreeClassifier:
    """Assigns anatomical labels (RCA / LAD / LCx) to a coronary artery skeleton.

    Parameters
    ----------
    bounding_box_padding_voxels:
        Padding applied when cropping the heart-chamber volume before
        computing distance transforms, to keep memory usage bounded.
    lcx_bias_mm:
        Millimeter bias favoring the LCx label during the provisional,
        per-voxel left-system split.
    lad_anchor_bias_mm:
        Millimeter bias favoring the LAD label during the final
        nearest-core anchoring step (keeps the left-main bifurcation
        attached to LAD).
    """

    def __init__(
        self,
        bounding_box_padding_voxels: int = 20,
        lcx_bias_mm: float = 5.0,
        lad_anchor_bias_mm: float = 15.0,
    ) -> None:
        self.padding = bounding_box_padding_voxels
        self.lcx_bias_mm = lcx_bias_mm
        self.lad_anchor_bias_mm = lad_anchor_bias_mm

    # ------------------------------------------------------------------
    # Stage 1: RCA vs. Left System
    # ------------------------------------------------------------------
    def classify_rca_vs_left(
        self,
        artery_mask_path: str | Path,
        chamber_mask_path: str | Path,
        output_path: str | Path,
    ) -> Path:
        """Label each coronary branch as RCA (1) or Left System (2).

        Parameters
        ----------
        artery_mask_path:
            Path to the (possibly non-binary) coronary artery
            segmentation mask.
        chamber_mask_path:
            Path to the corresponding TotalSegmentator heart-chamber
            mask, with LA=1, LV=2, RA=3 (index convention used at this
            stage).
        output_path:
            Destination path for the 2-class classified centerline.

        Returns
        -------
        pathlib.Path
            Path to the saved classification result.

        Raises
        ------
        AnatomicalMappingError
            If either the heart mask or the artery skeleton is empty.
        """
        artery_data, artery_affine = load_volume(artery_mask_path)
        canvas_shape = artery_data.shape
        skeleton_data = skeletonize((artery_data > 0).astype(np.uint8))

        chamber_data, chamber_affine = load_volume(chamber_mask_path)
        chamber_inv_affine = npl.inv(chamber_affine)

        skel_coords_artery = np.argwhere(skeleton_data)
        coords_heart = np.argwhere(chamber_data)

        if coords_heart.size == 0 or skel_coords_artery.size == 0:
            raise AnatomicalMappingError("Missing heart-chamber or artery-skeleton voxels; cannot classify.")

        skel_coords_chamber = self._map_to_chamber_space(
            skel_coords_artery, artery_affine, chamber_inv_affine
        )

        chamber_crop, bounds = self._crop_around_joint_bbox(
            chamber_data, coords_heart, skel_coords_chamber
        )

        d_la = self._safe_distance_transform(chamber_crop, label_value=1)
        d_lv = self._safe_distance_transform(chamber_crop, label_value=2)
        d_ra = self._safe_distance_transform(chamber_crop, label_value=3)

        classified_skeleton = np.zeros(canvas_shape, dtype=np.uint8)
        labeled_skeleton, num_branches = label(skeleton_data > 0, structure=_CONNECTIVITY_26)
        logger.info("Found %d skeleton branches; running per-branch majority vote.", num_branches)

        rca_voxel_count = 0
        left_voxel_count = 0

        for branch_id in range(1, num_branches + 1):
            branch_coords_artery = np.argwhere(labeled_skeleton == branch_id)
            branch_coords_chamber = self._map_to_chamber_space(
                branch_coords_artery, artery_affine, chamber_inv_affine
            )

            rca_votes, left_votes = self._vote_branch(
                branch_coords_chamber, bounds, chamber_crop.shape, d_ra, d_la, d_lv
            )
            winner_label = 1 if rca_votes > left_votes else 2

            if winner_label == 1:
                rca_voxel_count += len(branch_coords_artery)
            else:
                left_voxel_count += len(branch_coords_artery)

            for voxel in branch_coords_artery:
                classified_skeleton[tuple(voxel)] = winner_label

        logger.info("RCA voxels: %d | Left-system voxels: %d", rca_voxel_count, left_voxel_count)
        save_volume(classified_skeleton, artery_affine, output_path)
        return Path(output_path)

    @staticmethod
    def _vote_branch(
        branch_coords_chamber: np.ndarray,
        bounds: Tuple[np.ndarray, np.ndarray],
        crop_shape: Tuple[int, int, int],
        d_ra: np.ndarray,
        d_la: np.ndarray,
        d_lv: np.ndarray,
    ) -> Tuple[int, int]:
        """Cast a per-voxel proximity vote for an entire branch; return (rca, left) counts."""
        min_bound, _ = bounds
        rca_votes, left_votes = 0, 0

        local_coords = branch_coords_chamber - min_bound
        in_bounds = np.all((local_coords >= 0) & (local_coords < np.array(crop_shape)), axis=1)

        for cx, cy, cz in local_coords[in_bounds]:
            right_score = d_ra[cx, cy, cz]
            left_score = min(d_la[cx, cy, cz], d_lv[cx, cy, cz])
            if right_score < left_score:
                rca_votes += 1
            else:
                left_votes += 1

        return rca_votes, left_votes

    # ------------------------------------------------------------------
    # Stage 2: LAD vs. LCx split of the Left System
    # ------------------------------------------------------------------
    def split_lad_lcx(
        self,
        two_class_tree_path: str | Path,
        chamber_mask_path: str | Path,
        output_path: str | Path,
    ) -> Path:
        """Split the left coronary system (label 2) into LAD (2) and LCx (3).

        Parameters
        ----------
        two_class_tree_path:
            Path to the 2-class tree produced by
            :meth:`classify_rca_vs_left` (1=RCA, 2=Left System).
        chamber_mask_path:
            Path to the TotalSegmentator heart-chamber mask, with
            LA=2, RA=4, RV=5 (index convention used at this stage).
        output_path:
            Destination path for the final 3-class classified tree.

        Returns
        -------
        pathlib.Path
            Path to the saved 3-class classification result.

        Raises
        ------
        AnatomicalMappingError
            If either the heart mask or the left-system voxels are empty.
        """
        tree_data, tree_affine = load_volume(two_class_tree_path)
        chamber_data, chamber_affine = load_volume(chamber_mask_path)
        chamber_inv_affine = npl.inv(chamber_affine)

        left_coords_artery = np.argwhere(tree_data == 2)
        coords_heart = np.argwhere(chamber_data)

        if coords_heart.size == 0 or left_coords_artery.size == 0:
            raise AnatomicalMappingError("Missing heart-chamber or Left-System voxels; cannot split.")

        left_coords_mm = apply_affine(tree_affine, left_coords_artery)
        left_coords_chamber = np.round(apply_affine(chamber_inv_affine, left_coords_mm)).astype(int)

        chamber_crop, bounds = self._crop_around_joint_bbox(chamber_data, coords_heart, left_coords_chamber)
        min_bound, _ = bounds

        d_la = self._safe_distance_transform(chamber_crop, label_value=2)
        d_ra = self._safe_distance_transform(chamber_crop, label_value=4)
        d_rv = self._safe_distance_transform(chamber_crop, label_value=5)

        provisional = self._provisional_split(
            left_coords_artery, left_coords_chamber, min_bound, chamber_crop.shape, d_la, d_ra, d_rv
        )

        new_tree = np.zeros_like(tree_data, dtype=np.uint8)
        new_tree[tree_data == 1] = 1  # preserve RCA

        lad_count, lcx_count = self._anchor_to_nearest_core(
            provisional, left_coords_artery, left_coords_mm, tree_affine, new_tree
        )
        logger.info("LAD voxels: %d | LCx voxels: %d", lad_count, lcx_count)

        save_volume(new_tree, tree_affine, output_path)
        return Path(output_path)

    def _provisional_split(
        self,
        left_coords_artery: np.ndarray,
        left_coords_chamber: np.ndarray,
        min_bound: np.ndarray,
        crop_shape: Tuple[int, int, int],
        d_la: np.ndarray,
        d_ra: np.ndarray,
        d_rv: np.ndarray,
    ) -> np.ndarray:
        """Assign each left-system voxel a provisional LAD (2) or LCx (3) label."""
        # The provisional canvas must match the artery volume's coordinate space
        # (not the cropped chamber volume) since results are written back
        # using artery-space voxel indices.
        canvas_shape = tuple(int(v) + 1 for v in left_coords_artery.max(axis=0))
        provisional_canvas = np.zeros(canvas_shape, dtype=np.uint8)

        local_coords = left_coords_chamber - min_bound
        in_bounds = np.all((local_coords >= 0) & (local_coords < np.array(crop_shape)), axis=1)

        for i, (ax, ay, az) in enumerate(left_coords_artery):
            if in_bounds[i]:
                cx, cy, cz = local_coords[i]
                la_score = d_la[cx, cy, cz]
                anterior_score = min(d_ra[cx, cy, cz], d_rv[cx, cy, cz])
                label_value = 3 if la_score < (anterior_score - self.lcx_bias_mm) else 2
            else:
                label_value = 2  # fallback: LAD
            provisional_canvas[ax, ay, az] = label_value

        return provisional_canvas

    def _anchor_to_nearest_core(
        self,
        provisional_canvas: np.ndarray,
        left_coords_artery: np.ndarray,
        left_coords_mm: np.ndarray,
        tree_affine: np.ndarray,
        new_tree: np.ndarray,
    ) -> Tuple[int, int]:
        """Reassign fragmented tips to the nearest of the two largest connected cores."""
        lad_labeled, _ = label(provisional_canvas == 2, structure=_CONNECTIVITY_26)
        lcx_labeled, _ = label(provisional_canvas == 3, structure=_CONNECTIVITY_26)

        main_lad_id = self._largest_component_id(lad_labeled)
        main_lcx_id = self._largest_component_id(lcx_labeled)

        main_lad_coords = np.argwhere(lad_labeled == main_lad_id)
        main_lcx_coords = np.argwhere(lcx_labeled == main_lcx_id)

        lad_count, lcx_count = 0, 0

        if len(main_lad_coords) > 0 and len(main_lcx_coords) > 0:
            lad_tree = cKDTree(apply_affine(tree_affine, main_lad_coords))
            lcx_tree = cKDTree(apply_affine(tree_affine, main_lcx_coords))

            for i, point in enumerate(left_coords_mm):
                dist_to_lad, _ = lad_tree.query(point)
                dist_to_lcx, _ = lcx_tree.query(point)

                if dist_to_lcx < (dist_to_lad - self.lad_anchor_bias_mm):
                    final_label = 3
                    lcx_count += 1
                else:
                    final_label = 2
                    lad_count += 1
                new_tree[tuple(left_coords_artery[i])] = final_label
        else:
            logger.warning("A distinct LAD/LCx core could not be found; falling back to provisional labels.")
            for ax, ay, az in left_coords_artery:
                final_label = provisional_canvas[ax, ay, az]
                new_tree[ax, ay, az] = final_label
                if final_label == 2:
                    lad_count += 1
                else:
                    lcx_count += 1

        return lad_count, lcx_count

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _map_to_chamber_space(
        voxel_coords: np.ndarray, source_affine: np.ndarray, target_inv_affine: np.ndarray
    ) -> np.ndarray:
        """Map integer voxel coordinates from one image's grid to another's."""
        mm_coords = apply_affine(source_affine, voxel_coords)
        return np.round(apply_affine(target_inv_affine, mm_coords)).astype(int)

    def _crop_around_joint_bbox(
        self,
        chamber_data: np.ndarray,
        coords_heart: np.ndarray,
        coords_other: np.ndarray,
    ) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Crop the chamber volume to the padded bounding box that covers both point sets."""
        min_bound = np.minimum(coords_heart.min(axis=0), coords_other.min(axis=0))
        max_bound = np.maximum(coords_heart.max(axis=0), coords_other.max(axis=0))

        shape = np.array(chamber_data.shape)
        min_bound = np.maximum(0, min_bound - self.padding)
        max_bound = np.minimum(shape, max_bound + self.padding)

        crop = chamber_data[
            min_bound[0]:max_bound[0],
            min_bound[1]:max_bound[1],
            min_bound[2]:max_bound[2],
        ]
        return crop, (min_bound, max_bound)

    @staticmethod
    def _safe_distance_transform(chamber_crop: np.ndarray, label_value: int) -> np.ndarray:
        """Euclidean distance transform to a chamber label; infinity if the label is absent."""
        if np.any(chamber_crop == label_value):
            return distance_transform_edt(chamber_crop != label_value)
        return np.full(chamber_crop.shape, np.inf)

    @staticmethod
    def _largest_component_id(labeled_volume: np.ndarray) -> int:
        """Return the label id of the largest non-background connected component."""
        counts = np.bincount(labeled_volume.ravel())
        if len(counts) <= 1:
            return 0
        counts[0] = 0
        return int(counts.argmax())
