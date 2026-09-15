"""Configuration management for the COROnet pipeline.

All hyperparameters, file-system paths, and runtime options are defined
as typed :mod:`dataclasses` so that they can be validated early,
serialized to/from JSON or YAML, and passed explicitly between pipeline
components instead of being read from global module-level variables (as
in the original notebooks).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class PathConfig:
    """File-system layout for all pipeline artifacts.

    Attributes
    ----------
    root:
        Root directory that contains (or will contain) every other path.
        Equivalent to ``DRIVE_ROOT`` in the original notebooks, but
        decoupled from Google Colab/Drive.
    raw_ct_dir:
        Directory of raw, patient-level CT volumes (``*_0000.nii.gz`` or
        ``*.nii.gz``).
    heart_mask_dir:
        Output directory for TotalSegmentator heart-chamber masks.
    cropped_ct_dir:
        Output directory for CT volumes cropped to the heart ROI.
    coronary_raw_dir:
        Output directory for raw coronary artery segmentation masks
        (UU-Mamba / nnU-Net predictions).
    coronary_refined_dir:
        Output directory for skeletonized/thickened coronary masks.
    classifier_input_dir:
        Directory holding coronary masks renamed/prepared for anatomical
        classification.
    classified_centerline_dir:
        Output directory for anatomically classified (RCA/Left or
        RCA/LAD/LCx) 1-voxel centerlines.
    classified_thick_dir:
        Output directory for the thickened, anatomically classified
        coronary trees.
    coordinates_dir:
        Output directory for per-vessel voxel/mm coordinate JSON files.
    patches_dir:
        Output directory for per-patient, per-vessel 3D image patches
        (hierarchical ``Patient_<id>/<VESSEL>.nii.gz``).
    features_dir:
        Output directory for extracted 1D feature vectors, one
        sub-directory per CNN backbone.
    results_dir:
        Output directory for metrics, confusion matrices, and trained
        model checkpoints.
    annotation_json:
        Path to the clinical annotation file mapping each patient/vessel
        to ``status`` (Normal/Abnormal) and ``plaque_type``.
    split_csv:
        Path to the CSV describing the train/test split used for the
        plaque-type sequence classifier.
    """

    root: Path
    raw_ct_dir: Path = field(init=False)
    heart_mask_dir: Path = field(init=False)
    cropped_ct_dir: Path = field(init=False)
    coronary_raw_dir: Path = field(init=False)
    coronary_refined_dir: Path = field(init=False)
    classifier_input_dir: Path = field(init=False)
    classified_centerline_dir: Path = field(init=False)
    classified_thick_dir: Path = field(init=False)
    coordinates_dir: Path = field(init=False)
    patches_dir: Path = field(init=False)
    features_dir: Path = field(init=False)
    results_dir: Path = field(init=False)
    annotation_json: Path = field(init=False)
    split_csv: Path = field(init=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.raw_ct_dir = self.root / "ICC_Data"
        self.heart_mask_dir = self.root / "Segmentation" / "Helper" / "Totalsegmentor_Heart"
        self.cropped_ct_dir = self.root / "Segmentation" / "Cropped_Data"
        self.coronary_raw_dir = self.root / "Segmentation" / "Mamba_Coronary_Arteries"
        self.coronary_refined_dir = (
            self.root / "Segmentation" / "Helper" / "Mamba_Coronary_Arteries_Refined"
        )
        self.classifier_input_dir = self.root / "Segmentation" / "Coronary_Mapping_Input"
        self.classified_centerline_dir = self.root / "Segmentation" / "Classified_Trees_Centerline_Full"
        self.classified_thick_dir = self.root / "Segmentation" / "Classified_Trees_Thickened_Full"
        self.coordinates_dir = self.root / "Segmentation" / "Classified_Trees_Coordinates"
        self.patches_dir = self.root / "Classification_Input"
        self.features_dir = self.root / "Classification_Team_A" / "MIL_Features"
        self.results_dir = self.root / "Classification_Team_A" / "Experiment_Results"
        self.annotation_json = self.root / "annotation.json"
        self.split_csv = self.root / "COROnet_Dataset_Split.csv"

    def ensure_dirs(self) -> None:
        """Create every output directory that does not yet exist."""
        for candidate in (
            self.heart_mask_dir,
            self.cropped_ct_dir,
            self.coronary_raw_dir,
            self.coronary_refined_dir,
            self.classifier_input_dir,
            self.classified_centerline_dir,
            self.classified_thick_dir,
            self.coordinates_dir,
            self.patches_dir,
            self.features_dir,
            self.results_dir,
        ):
            candidate.mkdir(parents=True, exist_ok=True)


@dataclass
class SegmentationConfig:
    """Parameters controlling heart and coronary artery segmentation.

    Attributes
    ----------
    totalsegmentator_task:
        TotalSegmentator task name used for heart chamber extraction.
    totalsegmentator_license:
        Optional academic/commercial license key for TotalSegmentator V2.
    device:
        Compute device string passed to TotalSegmentator (``"gpu"`` or
        ``"cpu"``).
    target_chamber_labels:
        Voxel labels (from the heart-chamber mask) that define the
        cardiac ROI used for cropping, e.g. ``(1, 2, 3, 4)`` for
        LA/LV/RA/RV.
    crop_padding_voxels:
        Isotropic padding (in voxels) applied around the bounding box of
        the target chambers before cropping the raw CT.
    nnunet_dataset:
        nnU-Net dataset identifier for the coronary artery segmentation
        model (e.g. ``"Dataset101_ImageCAS"``).
    nnunet_trainer:
        nnU-Net trainer class name (e.g. ``"nnUNetTrainerUMambaEnc"``).
    nnunet_checkpoint:
        Checkpoint filename used at inference time.
    nnunet_step_size:
        Sliding-window step size used by ``nnUNetv2_predict``.
    skeleton_thickness_radius:
        Radius (in voxels) of the spherical structuring element used to
        re-thicken the skeletonized coronary mask.
    """

    totalsegmentator_task: str = "heartchambers_highres"
    totalsegmentator_license: Optional[str] = None
    device: str = "gpu"
    target_chamber_labels: Tuple[int, ...] = (1, 2, 3, 4)
    crop_padding_voxels: int = 25
    nnunet_dataset: str = "Dataset101_ImageCAS"
    nnunet_trainer: str = "nnUNetTrainerUMambaEnc"
    nnunet_checkpoint: str = "checkpoint_ImageCAS.pth"
    nnunet_step_size: float = 0.75
    skeleton_thickness_radius: int = 6


@dataclass
class AnatomicalMappingConfig:
    """Parameters for the RCA / LAD / LCx anatomical classification logic.

    Attributes
    ----------
    bounding_box_padding_voxels:
        Padding applied when cropping the heart-chamber volume for
        distance-map computation.
    lcx_bias_mm:
        Bias (in millimeters) applied in favor of the LCx label during
        the initial, per-voxel provisional split of the left coronary
        system.
    lad_anchor_bias_mm:
        Bias (in millimeters) applied in favor of the LAD label during
        the final nearest-core anchoring step.
    left_system_dilation_iterations:
        Number of binary-dilation iterations used to thicken the final
        classified centerlines for visualization.
    """

    bounding_box_padding_voxels: int = 20
    lcx_bias_mm: float = 5.0
    lad_anchor_bias_mm: float = 15.0
    left_system_dilation_iterations: int = 2


@dataclass
class PatchExtractionConfig:
    """Parameters for extracting fixed-size 3D patches along each vessel.

    Attributes
    ----------
    patch_size:
        Edge length (in voxels) of the cubic patch extracted around each
        centerline point.
    hounsfield_clip_range:
        ``(min, max)`` Hounsfield Unit clipping range applied before
        min-max normalization.
    """

    patch_size: int = 32
    hounsfield_clip_range: Tuple[float, float] = (-100.0, 800.0)

    @property
    def half_patch(self) -> int:
        """Half of :attr:`patch_size`, used for centered cropping."""
        return self.patch_size // 2


@dataclass
class BinaryClassifierConfig:
    """Hyperparameters for the vessel-level normal/abnormal 3D classifier.

    Attributes
    ----------
    spatial_size:
        Target ``(D, H, W)`` volume size after resizing.
    batch_size:
        Mini-batch size used during training/evaluation.
    epochs:
        Number of training epochs for the initial training stage.
    learning_rate:
        Optimizer learning rate for the initial training stage.
    fine_tune_epochs:
        Number of epochs used for the hard-negative fine-tuning stage.
    fine_tune_learning_rate:
        Learning rate used for the hard-negative fine-tuning stage.
    fine_tune_accumulation_steps:
        Gradient accumulation steps used during fine-tuning to simulate
        a larger effective batch size under memory constraints.
    early_stopping_patience:
        Number of epochs without F1 improvement before stopping early.
    test_time_augmentation:
        Whether to average predictions across the original volume and
        its two spatial flips at inference time.
    """

    spatial_size: Tuple[int, int, int] = (96, 96, 96)
    batch_size: int = 4
    epochs: int = 30
    learning_rate: float = 1e-4
    fine_tune_epochs: int = 20
    fine_tune_learning_rate: float = 1e-5
    fine_tune_accumulation_steps: int = 8
    early_stopping_patience: int = 6
    test_time_augmentation: bool = True


@dataclass
class FeatureExtractionConfig:
    """Parameters for the multi-backbone 1D feature extraction stage.

    Attributes
    ----------
    backbones:
        Names of the CNN backbones used to compress each 3D patch into a
        fixed-length feature vector.
    feature_dim:
        Output dimensionality of every backbone's feature vector.
    batch_size:
        Number of patches processed per forward pass.
    """

    backbones: Tuple[str, ...] = ("CNN", "DenseNet", "ResNet", "EfficientNet")
    feature_dim: int = 512
    batch_size: int = 16


@dataclass
class SequenceClassifierConfig:
    """Hyperparameters for the plaque-type sequence classifier.

    Attributes
    ----------
    architectures:
        Sequence-model architectures to evaluate (``"lstm"``,
        ``"bilstm"``, ``"gru"``, ``"transformer"``).
    hidden_dim:
        Hidden dimensionality of the recurrent layer (ignored for the
        transformer, which keeps ``input_dim``).
    num_layers:
        Number of stacked recurrent/transformer-encoder layers.
    num_classes:
        Number of plaque-type classes (Normal, Calcified, Soft, Mixed).
    batch_size:
        Mini-batch size used during training/evaluation.
    epochs:
        Number of training epochs.
    learning_rate:
        Optimizer learning rate.
    class_names:
        Human-readable names for each plaque-type class, indexed by
        label id.
    """

    architectures: Tuple[str, ...] = ("lstm", "bilstm", "gru", "transformer")
    hidden_dim: int = 128
    num_layers: int = 1
    num_classes: int = 4
    batch_size: int = 16
    learning_rate: float = 1e-4
    epochs: int = 20
    class_names: Tuple[str, ...] = ("Normal", "Calcified", "Soft", "Mixed")


@dataclass
class CoronetConfig:
    """Top-level configuration aggregating every pipeline stage.

    Parameters
    ----------
    paths:
        File-system layout, see :class:`PathConfig`.
    segmentation:
        See :class:`SegmentationConfig`.
    anatomical_mapping:
        See :class:`AnatomicalMappingConfig`.
    patch_extraction:
        See :class:`PatchExtractionConfig`.
    binary_classifier:
        See :class:`BinaryClassifierConfig`.
    feature_extraction:
        See :class:`FeatureExtractionConfig`.
    sequence_classifier:
        See :class:`SequenceClassifierConfig`.
    seed:
        Global random seed used for reproducible splits and training.

    Examples
    --------
    >>> config = CoronetConfig(root="/data/COROnet_Drive")
    >>> config.paths.raw_ct_dir
    PosixPath('/data/COROnet_Drive/ICC_Data')
    """

    root: Path
    paths: PathConfig = field(init=False)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    anatomical_mapping: AnatomicalMappingConfig = field(default_factory=AnatomicalMappingConfig)
    patch_extraction: PatchExtractionConfig = field(default_factory=PatchExtractionConfig)
    binary_classifier: BinaryClassifierConfig = field(default_factory=BinaryClassifierConfig)
    feature_extraction: FeatureExtractionConfig = field(default_factory=FeatureExtractionConfig)
    sequence_classifier: SequenceClassifierConfig = field(default_factory=SequenceClassifierConfig)
    seed: int = 42

    def __post_init__(self) -> None:
        self.paths = PathConfig(root=self.root)

    @classmethod
    def from_json(cls, json_path: str | Path) -> "CoronetConfig":
        """Load a configuration from a JSON file.

        Parameters
        ----------
        json_path:
            Path to a JSON document previously produced by
            :meth:`to_json`.

        Returns
        -------
        CoronetConfig
            The reconstructed configuration object.

        Raises
        ------
        FileNotFoundError
            If ``json_path`` does not exist.
        json.JSONDecodeError
            If ``json_path`` does not contain valid JSON.
        """
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"Config file not found: {json_path}")

        with json_path.open("r", encoding="utf-8") as handle:
            payload: Dict[str, Any] = json.load(handle)

        root = payload["root"]
        config = cls(root=root, seed=payload.get("seed", 42))
        for section_name in (
            "segmentation",
            "anatomical_mapping",
            "patch_extraction",
            "binary_classifier",
            "feature_extraction",
            "sequence_classifier",
        ):
            section_payload = payload.get(section_name)
            if section_payload:
                section_obj = getattr(config, section_name)
                for key, value in section_payload.items():
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, value)
        return config

    def to_json(self, json_path: str | Path) -> None:
        """Serialize this configuration to a human-readable JSON file.

        Parameters
        ----------
        json_path:
            Destination path. Parent directories are created if needed.
        """
        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)

        payload: Dict[str, Any] = {
            "root": str(self.root),
            "seed": self.seed,
            "segmentation": asdict(self.segmentation),
            "anatomical_mapping": asdict(self.anatomical_mapping),
            "patch_extraction": asdict(self.patch_extraction),
            "binary_classifier": asdict(self.binary_classifier),
            "feature_extraction": asdict(self.feature_extraction),
            "sequence_classifier": asdict(self.sequence_classifier),
        }
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)
