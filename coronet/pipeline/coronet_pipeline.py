"""High-level orchestration of the full COROnet pipeline.

:class:`CoronetPipeline` wires together every stage-specific component
(segmentation, post-processing, anatomical mapping, extraction,
modeling, training, evaluation) behind a small number of coarse-grained
methods, so that end users and the CLI do not need to know about the
internal class structure of the package.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import torch

from coronet.config import CoronetConfig
from coronet.data.datasets import (
    VesselFileListBuilder,
    VesselSequenceDataset,
    VesselVolumeDataset,
    sequence_collate_fn,
)
from coronet.data.ingestion import DataIngestor
from coronet.data.preprocessing import DatasetSplitter, VesselTransformFactory
from coronet.evaluation.evaluator import BinaryVesselEvaluator, SequenceClassificationEvaluator
from coronet.exceptions import CoronetError
from coronet.extraction.patch_extractor import PatchExtractor
from coronet.features.feature_extraction_service import FeatureExtractionService
from coronet.logging_config import get_logger
from coronet.mapping.anatomical_classifier import CoronaryTreeClassifier
from coronet.mapping.coordinate_extractor import CoordinateExtractor
from coronet.models.architectures import VesselResNet3D
from coronet.models.sequence_models import PlaqueSequenceModel
from coronet.segmentation.coronary_segmentor import CoronaryArterySegmentor
from coronet.segmentation.heart_segmentor import HeartChamberSegmentor
from coronet.segmentation.postprocessing import ClassifiedTreeThickener, HeartROICropper, VesselSkeletonizer
from coronet.training.binary_trainer import VesselClassifierTrainer
from coronet.training.sequence_trainer import PlaqueSequenceTrainer
from coronet.utils.nifti_io import normalize_patient_filename
from torch.utils.data import DataLoader

logger = get_logger(__name__)


class CoronetPipeline:
    """Orchestrates the end-to-end COROnet coronary-analysis pipeline.

    Parameters
    ----------
    config:
        A fully populated :class:`coronet.config.CoronetConfig` instance.
    device:
        Torch device used for every model-based stage. Defaults to CUDA
        if available, otherwise CPU.

    Examples
    --------
    >>> config = CoronetConfig(root="/data/COROnet_Drive")
    >>> pipeline = CoronetPipeline(config)
    >>> pipeline.run_segmentation_stage()  # doctest: +SKIP
    >>> pipeline.run_mapping_stage()  # doctest: +SKIP
    """

    def __init__(self, config: CoronetConfig, device: Optional[torch.device] = None) -> None:
        self.config = config
        self.config.paths.ensure_dirs()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.ingestor = DataIngestor(config.root)
        self.heart_segmentor = HeartChamberSegmentor(
            output_dir=config.paths.heart_mask_dir,
            task=config.segmentation.totalsegmentator_task,
            device="gpu" if self.device.type == "cuda" else "cpu",
            license_key=config.segmentation.totalsegmentator_license,
        )
        self.roi_cropper = HeartROICropper(
            target_labels=config.segmentation.target_chamber_labels,
            padding_voxels=config.segmentation.crop_padding_voxels,
        )
        self.skeletonizer = VesselSkeletonizer(thickness_radius=config.segmentation.skeleton_thickness_radius)
        self.tree_classifier = CoronaryTreeClassifier(
            bounding_box_padding_voxels=config.anatomical_mapping.bounding_box_padding_voxels,
            lcx_bias_mm=config.anatomical_mapping.lcx_bias_mm,
            lad_anchor_bias_mm=config.anatomical_mapping.lad_anchor_bias_mm,
        )
        self.tree_thickener = ClassifiedTreeThickener(
            dilation_iterations=config.anatomical_mapping.left_system_dilation_iterations
        )
        self.coordinate_extractor = CoordinateExtractor()
        self.patch_extractor = PatchExtractor(
            patch_size=config.patch_extraction.patch_size,
            hu_clip_range=config.patch_extraction.hounsfield_clip_range,
        )
        self.transform_factory = VesselTransformFactory(spatial_size=config.binary_classifier.spatial_size)

    # ------------------------------------------------------------------
    # Stage 1-2: Segmentation
    # ------------------------------------------------------------------
    def run_segmentation_stage(self, nnunet_env: Optional[Dict[str, str]] = None) -> None:
        """Run heart-chamber segmentation, ROI cropping, and coronary artery segmentation.

        Parameters
        ----------
        nnunet_env:
            Optional override for the nnU-Net environment directories,
            with keys ``"raw"``, ``"preprocessed"``, ``"results"``.
            Required to actually invoke coronary segmentation.
        """
        raw_ct_paths = self.ingestor.list_raw_ct_volumes(self.config.paths.raw_ct_dir)
        logger.info("Found %d raw CT volumes for segmentation.", len(raw_ct_paths))

        self.heart_segmentor.segment_batch(raw_ct_paths)

        cropped_paths: List[Path] = []
        for ct_path in raw_ct_paths:
            patient_id = normalize_patient_filename(ct_path.name)
            heart_mask_path = self.config.paths.heart_mask_dir / patient_id / "heart.nii.gz"
            output_path = self.config.paths.cropped_ct_dir / f"{ct_path.stem.split('.')[0]}_0000.nii.gz"

            if output_path.exists() or not heart_mask_path.exists():
                continue
            try:
                cropped_paths.append(self.roi_cropper.crop(ct_path, heart_mask_path, output_path))
            except CoronetError as exc:
                logger.error("Failed to crop patient %s: %s", patient_id, exc)

        if nnunet_env is not None:
            coronary_segmentor = CoronaryArterySegmentor(
                output_dir=self.config.paths.coronary_raw_dir,
                nnunet_raw_dir=nnunet_env["raw"],
                nnunet_preprocessed_dir=nnunet_env["preprocessed"],
                nnunet_results_dir=nnunet_env["results"],
                dataset_id=self.config.segmentation.nnunet_dataset,
                trainer=self.config.segmentation.nnunet_trainer,
                checkpoint=self.config.segmentation.nnunet_checkpoint,
                step_size=self.config.segmentation.nnunet_step_size,
            )
            all_cropped = self.ingestor.list_raw_ct_volumes(self.config.paths.cropped_ct_dir)
            coronary_segmentor.segment_batch(all_cropped)
        else:
            logger.warning("Skipping coronary artery segmentation: no nnU-Net environment provided.")

    # ------------------------------------------------------------------
    # Stage 3: Anatomical mapping
    # ------------------------------------------------------------------
    def run_mapping_stage(self) -> None:
        """Skeletonize, anatomically classify (RCA/LAD/LCx), and extract coordinates for every patient."""
        raw_masks = sorted(self.config.paths.coronary_raw_dir.glob("*.nii.gz"))
        logger.info("Running anatomical mapping for %d coronary masks.", len(raw_masks))

        for raw_mask_path in raw_masks:
            patient_id = normalize_patient_filename(raw_mask_path.name)
            try:
                self._map_single_patient(patient_id, raw_mask_path)
            except CoronetError as exc:
                logger.error("Anatomical mapping failed for patient %s: %s", patient_id, exc)
                continue

    def _map_single_patient(self, patient_id: str, raw_mask_path: Path) -> None:
        """Run the full skeletonize -> classify -> thicken -> extract-coordinates chain for one patient."""
        heart_mask_path = self.config.paths.heart_mask_dir / patient_id / "heart.nii.gz"
        if not heart_mask_path.exists():
            raise CoronetError(f"No heart-chamber mask found for patient {patient_id}")

        skeleton_path = self.config.paths.coronary_refined_dir / f"{patient_id}_skeleton.nii.gz"
        thickened_raw_path = self.config.paths.coronary_refined_dir / f"{patient_id}_thickened_raw.nii.gz"
        self.skeletonizer.skeletonize_and_thicken(raw_mask_path, skeleton_path, thickened_raw_path)

        two_class_path = self.config.paths.classified_centerline_dir / f"{patient_id}_rca_left.nii.gz"
        self.tree_classifier.classify_rca_vs_left(skeleton_path, heart_mask_path, two_class_path)

        three_class_path = self.config.paths.classified_centerline_dir / f"full_classified_tree_{patient_id}.nii.gz"
        self.tree_classifier.split_lad_lcx(two_class_path, heart_mask_path, three_class_path)

        thick_output_path = self.config.paths.classified_thick_dir / f"full_classified_tree_{patient_id}.nii.gz"
        self.tree_thickener.thicken(three_class_path, thick_output_path)

        coordinates_path = self.config.paths.coordinates_dir / f"{patient_id}_coordinates.json"
        self.coordinate_extractor.extract(three_class_path, coordinates_path)

        logger.info("Anatomical mapping complete for patient %s", patient_id)

    # ------------------------------------------------------------------
    # Stage 4: Patch extraction
    # ------------------------------------------------------------------
    def run_extraction_stage(self) -> Dict[str, Dict[str, int]]:
        """Extract fixed-size 3D patches along every patient's classified vessels.

        Returns
        -------
        dict
            Mapping from patient id to its per-vessel patch counts.
        """
        classified_trees = sorted(self.config.paths.classified_thick_dir.glob("full_classified_tree_*.nii.gz"))
        results: Dict[str, Dict[str, int]] = {}

        for tree_path in classified_trees:
            patient_id = normalize_patient_filename(tree_path.name)
            raw_ct_path = self.config.paths.cropped_ct_dir / f"ICC_Patient_{patient_id}_0000.nii.gz"
            patient_output_dir = self.config.paths.patches_dir / f"Patient_{patient_id}"

            if not raw_ct_path.exists():
                logger.warning("No cropped CT found for patient %s, skipping patch extraction.", patient_id)
                continue

            try:
                results[patient_id] = self.patch_extractor.extract_for_patient(
                    tree_path, raw_ct_path, patient_output_dir
                )
            except CoronetError as exc:
                logger.error("Patch extraction failed for patient %s: %s", patient_id, exc)

        return results

    # ------------------------------------------------------------------
    # Stage 5: Feature extraction
    # ------------------------------------------------------------------
    def run_feature_extraction_stage(self) -> Dict[str, int]:
        """Extract multi-backbone feature sequences for every annotated vessel patch.

        Returns
        -------
        dict
            Number of newly extracted sequences per backbone.
        """
        annotation_patients = self.ingestor.load_annotation_json(self.config.paths.annotation_json)
        service = FeatureExtractionService(
            patches_dir=self.config.paths.patches_dir,
            features_dir=self.config.paths.features_dir,
            device=self.device,
            feature_dim=self.config.feature_extraction.feature_dim,
            batch_size=self.config.feature_extraction.batch_size,
        )
        records = service.discover_records(annotation_patients)
        return service.run_all_backbones(records, self.config.feature_extraction.backbones)

    # ------------------------------------------------------------------
    # Stage 6: Binary classifier training
    # ------------------------------------------------------------------
    def train_binary_classifier(
        self, split_table_path: str | Path, patient_col: str, status_col: str, split_col: str
    ) -> float:
        """Train the binary (normal/abnormal) vessel classifier end to end.

        Parameters
        ----------
        split_table_path:
            Path to a CSV/Excel file describing which vessels belong to
            the train vs. validation split.
        patient_col, status_col, split_col:
            Column names within the split table (see
            :class:`coronet.data.datasets.VesselFileListBuilder`).

        Returns
        -------
        float
            Best validation macro-F1 achieved.
        """
        split_table = self.ingestor.load_split_table(split_table_path)
        file_builder = VesselFileListBuilder(self.config.paths.patches_dir)
        train_files, val_files = file_builder.from_patient_vessel_split(
            split_table, patient_col, status_col, split_col
        )

        train_dataset = VesselVolumeDataset(train_files, self.transform_factory.training_transforms())
        val_dataset = VesselVolumeDataset(val_files, self.transform_factory.evaluation_transforms())

        cfg = self.config.binary_classifier
        train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)

        model = VesselResNet3D(pretrained=True, num_classes=1)
        checkpoint_path = self.config.paths.results_dir / "vessel_classifier_best.pt"
        trainer = VesselClassifierTrainer(model, self.device, checkpoint_path)

        best_f1 = trainer.fit(
            train_loader, val_loader, epochs=cfg.epochs, learning_rate=cfg.learning_rate,
            early_stopping_patience=cfg.early_stopping_patience,
        )

        if cfg.fine_tune_epochs > 0:
            trainer.load_checkpoint()
            best_f1 = trainer.fine_tune(
                train_loader, val_loader, epochs=cfg.fine_tune_epochs,
                learning_rate=cfg.fine_tune_learning_rate,
                accumulation_steps=cfg.fine_tune_accumulation_steps,
                early_stopping_patience=cfg.early_stopping_patience,
            )

        return best_f1

    # ------------------------------------------------------------------
    # Stage 7: Sequence classifier training
    # ------------------------------------------------------------------
    def train_sequence_classifier(self, backbone_name: str, architecture: str) -> Dict[str, float]:
        """Train and evaluate the plaque-type sequence classifier for one backbone/architecture pair.

        Parameters
        ----------
        backbone_name:
            Name of the feature-extraction backbone whose outputs to
            use, e.g. ``"ResNet"``.
        architecture:
            Sequence-model architecture, one of ``"lstm"``,
            ``"bilstm"``, ``"gru"``, ``"transformer"``.

        Returns
        -------
        dict
            Test-set metrics: ``{"accuracy", "weighted_f1", "weighted_recall"}``.
        """
        split_table = self.ingestor.load_split_table(self.config.paths.split_csv)
        train_split = split_table[split_table["Split"] == "Train"]
        test_split = split_table[split_table["Split"] == "Test"]

        feature_dir = self.config.paths.features_dir / backbone_name
        cfg = self.config.sequence_classifier

        train_dataset = VesselSequenceDataset(train_split, feature_dir, self.config.feature_extraction.feature_dim)
        test_dataset = VesselSequenceDataset(test_split, feature_dir, self.config.feature_extraction.feature_dim)

        train_loader = DataLoader(
            train_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=sequence_collate_fn
        )
        test_loader = DataLoader(
            test_dataset, batch_size=cfg.batch_size, shuffle=False, collate_fn=sequence_collate_fn
        )

        model = PlaqueSequenceModel(
            input_dim=self.config.feature_extraction.feature_dim,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            num_classes=cfg.num_classes,
            architecture=architecture,
        )
        trainer = PlaqueSequenceTrainer(model, self.device)
        trainer.fit(train_loader, epochs=cfg.epochs, learning_rate=cfg.learning_rate)

        experiment_name = f"{backbone_name}_{architecture}"
        evaluator = SequenceClassificationEvaluator(model, self.device, cfg.class_names)
        return evaluator.evaluate_and_save(
            test_loader, self.config.paths.results_dir / experiment_name, experiment_name
        )
