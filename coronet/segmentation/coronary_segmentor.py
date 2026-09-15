"""Coronary artery segmentation using the UU-Mamba nnU-Net trainer.

Refactored from ``02_MMBAA_Coronary_Mapping.ipynb``. The original
notebook performed environment bootstrapping (installing a native
Python 3.10 interpreter, PyTorch, Mamba kernels, and a patched UU-Mamba
checkout) inline in a Colab cell. That bootstrapping is an
infrastructure/deployment concern, not pipeline logic, so it has been
moved out of this class; see ``README.md`` for environment setup
instructions. This class is responsible only for invoking
``nnUNetv2_predict`` against a prepared local staging directory and
syncing predictions back to persistent storage.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Set

from coronet.exceptions import MissingInputError, SegmentationError
from coronet.logging_config import get_logger

logger = get_logger(__name__)


class CoronaryArterySegmentor:
    """Runs nnU-Net (UU-Mamba trainer) inference for coronary artery segmentation.

    Parameters
    ----------
    output_dir:
        Directory where final coronary artery segmentation masks are
        stored, one file per patient.
    nnunet_raw_dir, nnunet_preprocessed_dir, nnunet_results_dir:
        nnU-Net environment directories (exported as ``nnUNet_raw``,
        ``nnUNet_preprocessed``, ``nnUNet_results`` respectively).
    dataset_id:
        nnU-Net dataset identifier, e.g. ``"Dataset101_ImageCAS"``.
    trainer:
        nnU-Net trainer class name, e.g. ``"nnUNetTrainerUMambaEnc"``.
    checkpoint:
        Checkpoint filename to load for inference.
    step_size:
        Sliding-window step size (lower values increase accuracy at the
        cost of memory and runtime).
    staging_dir:
        Local (fast-disk) directory used to stage inputs/outputs to
        avoid repeatedly reading/writing a slow or network-mounted
        filesystem during inference.
    excluded_patient_ids:
        Patient IDs to skip entirely, e.g. known out-of-memory cases.

    Examples
    --------
    >>> segmentor = CoronaryArterySegmentor(
    ...     output_dir="/data/coronary_masks",
    ...     nnunet_raw_dir="/data/nnUNet_raw",
    ...     nnunet_preprocessed_dir="/data/nnUNet_preprocessed",
    ...     nnunet_results_dir="/data/nnUNet_results",
    ... )
    >>> segmentor.segment_batch(["/data/cropped/patient_0001_0000.nii.gz"])  # doctest: +SKIP
    """

    def __init__(
        self,
        output_dir: str | Path,
        nnunet_raw_dir: str | Path,
        nnunet_preprocessed_dir: str | Path,
        nnunet_results_dir: str | Path,
        dataset_id: str = "Dataset101_ImageCAS",
        trainer: str = "nnUNetTrainerUMambaEnc",
        checkpoint: str = "checkpoint_ImageCAS.pth",
        step_size: float = 0.75,
        staging_dir: str | Path = "/tmp/coronet_coronary_staging",
        excluded_patient_ids: Optional[Set[str]] = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.nnunet_raw_dir = Path(nnunet_raw_dir)
        self.nnunet_preprocessed_dir = Path(nnunet_preprocessed_dir)
        self.nnunet_results_dir = Path(nnunet_results_dir)
        self.dataset_id = dataset_id
        self.trainer = trainer
        self.checkpoint = checkpoint
        self.step_size = step_size
        self.staging_dir = Path(staging_dir)
        self.excluded_patient_ids = excluded_patient_ids or set()

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _expected_output_name(self, input_filename: str) -> str:
        """nnU-Net drops the ``_0000`` modality suffix from output filenames."""
        return input_filename.replace("_0000", "")

    def _queue_pending_patients(self, cropped_files: List[Path], local_inputs: Path) -> int:
        """Copy not-yet-processed patient volumes into the local staging area."""
        queued_count = 0
        for file_path in cropped_files:
            filename = file_path.name

            if any(excluded in filename for excluded in self.excluded_patient_ids):
                logger.info("Manually excluded patient file: %s", filename)
                continue

            expected_output = self.output_dir / self._expected_output_name(filename)
            if expected_output.exists():
                logger.info("Already predicted, skipping: %s", expected_output.name)
                continue

            shutil.copy(file_path, local_inputs / filename)
            queued_count += 1

        return queued_count

    def segment_batch(self, cropped_ct_paths: List[str | Path]) -> List[Path]:
        """Run coronary artery segmentation for a batch of cropped CT volumes.

        Parameters
        ----------
        cropped_ct_paths:
            Paths to heart-ROI-cropped CT volumes named with the
            nnU-Net ``_0000`` modality suffix.

        Returns
        -------
        list of pathlib.Path
            Paths to the final segmentation masks that were newly
            produced (already-processed patients are skipped and not
            included).

        Raises
        ------
        MissingInputError
            If ``cropped_ct_paths`` is empty.
        SegmentationError
            If the underlying ``nnUNetv2_predict`` call fails.
        """
        if not cropped_ct_paths:
            raise MissingInputError("No cropped CT volumes were provided for coronary segmentation.")

        local_inputs = self.staging_dir / "inputs"
        local_preds = self.staging_dir / "predictions"
        for directory in (local_inputs, local_preds):
            if directory.exists():
                shutil.rmtree(directory)
            directory.mkdir(parents=True, exist_ok=True)

        queued_count = self._queue_pending_patients([Path(p) for p in cropped_ct_paths], local_inputs)
        if queued_count == 0:
            logger.info("No new patients to segment; everything is already processed.")
            return []

        logger.info("Launching nnU-Net inference on %d patients.", queued_count)
        env = dict(os.environ)
        env["nnUNet_raw"] = str(self.nnunet_raw_dir)
        env["nnUNet_preprocessed"] = str(self.nnunet_preprocessed_dir)
        env["nnUNet_results"] = str(self.nnunet_results_dir)
        env["MPLBACKEND"] = "agg"

        try:
            subprocess.run(
                [
                    "nnUNetv2_predict",
                    "-i",
                    str(local_inputs),
                    "-o",
                    str(local_preds),
                    "-d",
                    self.dataset_id,
                    "-c",
                    "3d_fullres",
                    "-f",
                    "all",
                    "-tr",
                    self.trainer,
                    "--disable_tta",
                    "-step_size",
                    str(self.step_size),
                    "-chk",
                    self.checkpoint,
                    "-npp",
                    "1",
                    "-nps",
                    "1",
                ],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
        except subprocess.CalledProcessError as exc:
            raise SegmentationError(f"nnUNetv2_predict failed: {exc.stderr}") from exc

        return self._sync_predictions(local_preds)

    def _sync_predictions(self, local_preds: Path) -> List[Path]:
        """Copy newly generated predictions from local staging to the output directory."""
        synced: List[Path] = []
        for prediction_file in local_preds.iterdir():
            destination = self.output_dir / prediction_file.name
            if not destination.exists():
                shutil.copy(prediction_file, destination)
                synced.append(destination)
                logger.info("Synced prediction: %s", prediction_file.name)
        return synced
