"""Heart chamber segmentation using TotalSegmentator.

Refactored from ``01_Totalsegmentor_Heart.ipynb``. The original notebook
mounted Google Drive, shelled out to the ``TotalSegmentator`` CLI for
every patient, and manually staged files through a local temp directory
to avoid Drive I/O latency. This module keeps the local-staging
optimization (it is a genuine performance concern for large volumes)
but expresses it as a reusable, testable class with structured error
handling instead of a linear notebook script.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

from coronet.exceptions import MissingInputError, SegmentationError
from coronet.logging_config import get_logger
from coronet.utils.memory import clear_gpu_memory
from coronet.utils.nifti_io import normalize_patient_filename

logger = get_logger(__name__)


class HeartChamberSegmentor:
    """Segments cardiac chambers from raw CT volumes with TotalSegmentator.

    Parameters
    ----------
    output_dir:
        Root directory under which one sub-directory per patient is
        created, each containing a ``heart.nii.gz`` multi-label mask.
    task:
        TotalSegmentator task name, e.g. ``"heartchambers_highres"``.
    device:
        Compute device passed to TotalSegmentator (``"gpu"`` or
        ``"cpu"``).
    license_key:
        Optional academic/commercial license key required by
        TotalSegmentator V2 for some tasks.
    staging_dir:
        Local (fast-disk) directory used to stage TotalSegmentator's
        output before copying it to ``output_dir``, which may sit on a
        slower or network-mounted filesystem.

    Examples
    --------
    >>> segmentor = HeartChamberSegmentor(output_dir="/data/heart_masks")
    >>> segmentor.segment_patient("patient_0001", "/data/raw/patient_0001.nii.gz")
    PosixPath('/data/heart_masks/patient_0001/heart.nii.gz')
    """

    def __init__(
        self,
        output_dir: str | Path,
        task: str = "heartchambers_highres",
        device: str = "gpu",
        license_key: Optional[str] = None,
        staging_dir: str | Path = "/tmp/coronet_heart_staging",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.task = task
        self.device = device
        self.license_key = license_key
        self.staging_dir = Path(staging_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.license_key:
            self._set_license(self.license_key)

    @staticmethod
    def _set_license(license_key: str) -> None:
        """Register a TotalSegmentator license key via its CLI helper."""
        try:
            subprocess.run(
                ["totalseg_set_license", "-l", license_key],
                check=True,
                capture_output=True,
                text=True,
            )
            logger.info("TotalSegmentator license registered.")
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            raise SegmentationError(f"Failed to register TotalSegmentator license: {exc}") from exc

    def is_already_processed(self, patient_id: str) -> bool:
        """Return ``True`` if a heart mask already exists for this patient.

        Parameters
        ----------
        patient_id:
            Patient identifier used to name the output sub-directory.
        """
        patient_dir = self.output_dir / patient_id
        return patient_dir.exists() and any(patient_dir.glob("*.nii.gz"))

    def segment_patient(self, patient_id: str, ct_path: str | Path) -> Path:
        """Run heart-chamber segmentation for a single patient.

        Parameters
        ----------
        patient_id:
            Identifier used to name the output sub-directory.
        ct_path:
            Path to the patient's raw CT volume.

        Returns
        -------
        pathlib.Path
            Path to the saved ``heart.nii.gz`` multi-label mask.

        Raises
        ------
        MissingInputError
            If ``ct_path`` does not exist.
        SegmentationError
            If TotalSegmentator fails or produces no recognizable output.
        """
        ct_path = Path(ct_path)
        if not ct_path.exists():
            raise MissingInputError(f"CT volume not found for patient {patient_id}: {ct_path}")

        patient_dir = self.output_dir / patient_id
        patient_dir.mkdir(parents=True, exist_ok=True)
        drive_heart_file = patient_dir / "heart.nii.gz"

        if self.staging_dir.exists():
            shutil.rmtree(self.staging_dir)
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        local_heart_file = self.staging_dir / "heart.nii.gz"

        clear_gpu_memory()
        logger.info("Segmenting heart chambers for patient %s", patient_id)
        try:
            subprocess.run(
                [
                    "TotalSegmentator",
                    "-i",
                    str(ct_path),
                    "-o",
                    str(local_heart_file),
                    "-ta",
                    self.task,
                    "--ml",
                    "-d",
                    self.device,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            raise SegmentationError(
                f"TotalSegmentator failed for patient {patient_id}: {exc.stderr}"
            ) from exc

        return self._ship_output(patient_id, local_heart_file, drive_heart_file, patient_dir)

    @staticmethod
    def _ship_output(
        patient_id: str,
        local_heart_file: Path,
        destination_file: Path,
        patient_dir: Path,
    ) -> Path:
        """Move TotalSegmentator's staged output to its final destination."""
        if local_heart_file.is_file():
            shutil.copy(local_heart_file, destination_file)
            if destination_file.exists() and destination_file.stat().st_size > 1024:
                local_heart_file.unlink()
                logger.info("Patient %s processed successfully.", patient_id)
                return destination_file
            raise SegmentationError(f"Copy verification failed for patient {patient_id}.")

        if local_heart_file.is_dir():
            for child in local_heart_file.iterdir():
                shutil.copy(child, patient_dir / child.name)
            logger.info("Patient %s (directory output) processed successfully.", patient_id)
            return destination_file

        raise SegmentationError(f"TotalSegmentator produced no output for patient {patient_id}.")

    def segment_batch(self, ct_paths: List[str | Path]) -> List[Path]:
        """Segment heart chambers for a batch of patients, skipping completed ones.

        Parameters
        ----------
        ct_paths:
            Paths to raw CT volumes, one per patient.

        Returns
        -------
        list of pathlib.Path
            Output ``heart.nii.gz`` paths for every successfully
            processed patient. Patients that fail are logged and
            skipped so that a single bad volume does not abort the
            batch.
        """
        results: List[Path] = []
        for index, ct_path in enumerate(ct_paths, start=1):
            ct_path = Path(ct_path)
            patient_id = normalize_patient_filename(ct_path.name)

            if self.is_already_processed(patient_id):
                logger.info("[%d/%d] Skipping %s (already processed)", index, len(ct_paths), patient_id)
                continue

            try:
                results.append(self.segment_patient(patient_id, ct_path))
            except (MissingInputError, SegmentationError) as exc:
                logger.error("[%d/%d] Failed to segment %s: %s", index, len(ct_paths), patient_id, exc)
                continue

        logger.info("Heart chamber segmentation complete: %d/%d patients succeeded.", len(results), len(ct_paths))
        return results
