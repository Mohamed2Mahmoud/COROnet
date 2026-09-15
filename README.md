# COROnet

**COROnet** is an object-oriented pipeline that turns a raw cardiac CT scan into
per-vessel coronary artery plaque classifications. It automates the full
radiology-to-diagnosis workflow: heart chamber segmentation, coronary artery
segmentation, anatomical vessel labelling (RCA / LAD / LCx), 3D patch
extraction, deep-learning-based normal/abnormal vessel classification, and
sequence-model-based plaque-type classification (Normal, Calcified, Soft,
Mixed).

This repository was refactored from a series of exploratory Jupyter notebooks
into a modular, testable, production-oriented Python package under `coronet/`.

## What it does

Given a raw, non-contrast or contrast cardiac CT volume, COROnet:

1. **Segments the heart chambers** with TotalSegmentator and crops the CT to
   the cardiac region of interest.
2. **Segments the coronary arteries** with a UU-Mamba-enhanced nnU-Net model.
3. **Anatomically classifies** the coronary tree into the right coronary
   artery (RCA), left anterior descending (LAD), and left circumflex (LCx)
   branches, using per-branch and per-voxel proximity voting against the
   heart-chamber mask.
4. **Extracts 3D micro-patches** (32³ voxels by default) along each vessel's
   centerline from the raw CT.
5. **Classifies each vessel as normal or abnormal** with a 3D ResNet-18,
   using test-time augmentation and F1-optimal thresholding.
6. **Extracts fixed-length feature vectors** for every patch using one of
   four interchangeable CNN backbones (custom CNN, DenseNet, ResNet,
   EfficientNet).
7. **Classifies the plaque type** of each abnormal vessel (Calcified / Soft /
   Mixed) from its sequence of patch features, using a configurable
   recurrent or Transformer sequence model (LSTM / BiLSTM / GRU /
   Transformer).

## Architecture overview

The package follows a strict separation of concerns: each pipeline stage is
its own class with a narrow, well-typed interface, and every class can be
instantiated, tested, and swapped independently. A single
`CoronetConfig` dataclass carries every hyperparameter and file-system path,
so no module reads global variables.

```
CoronetConfig                     -> hyperparameters & paths (coronet/config.py)
        |
        v
CoronetPipeline                   -> high-level orchestration (coronet/pipeline/)
        |
        +-- HeartChamberSegmentor        (coronet/segmentation/)
        +-- HeartROICropper
        +-- CoronaryArterySegmentor
        +-- VesselSkeletonizer / ClassifiedTreeThickener
        +-- CoronaryTreeClassifier       (coronet/mapping/)   RCA / LAD / LCx
        +-- CoordinateExtractor
        +-- PatchExtractor               (coronet/extraction/)
        +-- FeatureExtractionService     (coronet/features/)
        +-- VesselResNet3D / PlaqueSequenceModel  (coronet/models/)
        +-- VesselClassifierTrainer / PlaqueSequenceTrainer   (coronet/training/)
        +-- BinaryVesselEvaluator / SequenceClassificationEvaluator (coronet/evaluation/)
        +-- DataIngestor / DatasetSplitter / *Dataset classes (coronet/data/)
```

**Design principles applied during the refactor:**

- **Configuration as data**: every hyperparameter and path lives in a typed
  `dataclass` (`coronet/config.py`), not scattered notebook variables.
- **Single-responsibility classes**: segmentation, cropping, skeletonization,
  anatomical classification, patch extraction, model architecture, training,
  and evaluation are all distinct classes in distinct modules.
- **Explicit error handling**: a custom exception hierarchy
  (`coronet/exceptions.py`) replaces bare `try/except: pass` and silent
  notebook failures, so batch jobs can log and skip a single bad patient
  without crashing.
- **Structured logging**: every module logs through
  `coronet.logging_config.get_logger`, never `print`.
- **Full type hints and docstrings**: every public class and method has
  NumPy-style docstrings and type hints, enabling static analysis and
  IDE autocompletion.
- **Testability**: components that don't require a GPU or external tools
  (config, patch extraction) ship with unit tests using synthetic data
  (`tests/`); GPU/tool-dependent stages are structured so they can be
  exercised via dependency injection in integration tests.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Mohamed2Mahmoud/COROnet.git
cd COROnet
```

### 2. Create an environment

COROnet depends on PyTorch, MONAI, TotalSegmentator, and nnU-Net (with the
UU-Mamba trainer). These have interacting native/CUDA dependencies, so a
dedicated virtual environment is strongly recommended.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

### 4. External tool setup (segmentation stages only)

- **TotalSegmentator**: register a license key if your chosen task requires
  one: `totalseg_set_license -l <your_key>`, or pass `license_key=` to
  `HeartChamberSegmentor`.
- **nnU-Net / UU-Mamba**: install the UU-Mamba fork of nnU-Net and set the
  three standard nnU-Net environment variables (`nnUNet_raw`,
  `nnUNet_preprocessed`, `nnUNet_results`) or pass the equivalent directories
  to `CoronaryArterySegmentor`. Place the trained
  `checkpoint_ImageCAS.pth` checkpoint under the corresponding
  `nnUNet_results/Dataset101_ImageCAS/.../nnUNetTrainerUMambaEnc__.../` folder.

## Project structure

```
COROnet/
├── coronet/                          # Installable Python package
│   ├── __init__.py
│   ├── config.py                     # CoronetConfig, PathConfig, and per-stage dataclasses
│   ├── exceptions.py                 # CoronetError hierarchy
│   ├── logging_config.py             # get_logger / configure_logging
│   ├── utils/
│   │   ├── nifti_io.py               # load_volume, save_volume, patient-ID parsing
│   │   └── memory.py                 # clear_gpu_memory
│   ├── segmentation/
│   │   ├── heart_segmentor.py        # HeartChamberSegmentor (TotalSegmentator)
│   │   ├── coronary_segmentor.py     # CoronaryArterySegmentor (nnU-Net / UU-Mamba)
│   │   └── postprocessing.py         # HeartROICropper, VesselSkeletonizer, ClassifiedTreeThickener
│   ├── mapping/
│   │   ├── anatomical_classifier.py  # CoronaryTreeClassifier (RCA / LAD / LCx)
│   │   └── coordinate_extractor.py   # CoordinateExtractor
│   ├── extraction/
│   │   └── patch_extractor.py        # PatchExtractor
│   ├── data/
│   │   ├── ingestion.py              # DataIngestor
│   │   ├── preprocessing.py          # DatasetSplitter, VesselTransformFactory
│   │   └── datasets.py               # VesselVolumeDataset, VesselSequenceDataset
│   ├── models/
│   │   ├── architectures.py          # CustomCNN3D, VesselResNet3D, FeatureExtractorFactory
│   │   └── sequence_models.py        # PlaqueSequenceModel (LSTM/BiLSTM/GRU/Transformer)
│   ├── training/
│   │   ├── binary_trainer.py         # VesselClassifierTrainer
│   │   └── sequence_trainer.py       # PlaqueSequenceTrainer
│   ├── evaluation/
│   │   └── evaluator.py              # BinaryVesselEvaluator, SequenceClassificationEvaluator, ThresholdOptimizer
│   ├── features/
│   │   └── feature_extraction_service.py  # FeatureExtractionService
│   └── pipeline/
│       └── coronet_pipeline.py       # CoronetPipeline (orchestration)
├── scripts/
│   └── cli.py                        # Command-line entry point
├── tests/
│   ├── test_config.py
│   └── test_patch_extractor.py
├── requirements.txt
├── setup.py
├── .gitignore
└── README.md
```

## Usage

### Programmatic usage

```python
from pathlib import Path
from coronet.config import CoronetConfig
from coronet.pipeline.coronet_pipeline import CoronetPipeline

config = CoronetConfig(root=Path("/data/COROnet_Drive"))
pipeline = CoronetPipeline(config)

# Stage 1-2: heart-chamber + coronary artery segmentation
pipeline.run_segmentation_stage(nnunet_env={
    "raw": "/data/nnUNet_raw",
    "preprocessed": "/data/nnUNet_preprocessed",
    "results": "/data/nnUNet_results",
})

# Stage 3: anatomical classification (RCA/LAD/LCx) + coordinate extraction
pipeline.run_mapping_stage()

# Stage 4: 3D patch extraction
pipeline.run_extraction_stage()

# Stage 5: multi-backbone feature extraction
pipeline.run_feature_extraction_stage()

# Stage 6: train the binary (normal/abnormal) vessel classifier
best_f1 = pipeline.train_binary_classifier(
    split_table_path="/data/COROnet_Drive/vessel_split.csv",
    patient_col="patient_vessel_id",
    status_col="status",
    split_col="split",
)

# Stage 7: train the plaque-type sequence classifier
metrics = pipeline.train_sequence_classifier(backbone_name="ResNet", architecture="bilstm")
print(metrics)
```

### Command-line usage

```bash
# Segmentation (heart chambers + coronary arteries)
python scripts/cli.py --root /data/COROnet_Drive segment

# Anatomical mapping (RCA / LAD / LCx classification + coordinates)
python scripts/cli.py --root /data/COROnet_Drive map

# 3D patch extraction
python scripts/cli.py --root /data/COROnet_Drive extract-patches

# Multi-backbone feature extraction
python scripts/cli.py --root /data/COROnet_Drive extract-features

# Train the binary vessel classifier
python scripts/cli.py --root /data/COROnet_Drive train-binary \
    --split-table /data/COROnet_Drive/vessel_split.csv \
    --patient-col patient_vessel_id --status-col status --split-col split

# Train the plaque-type sequence classifier
python scripts/cli.py --root /data/COROnet_Drive train-sequence \
    --backbone ResNet --architecture bilstm
```

### Using individual components directly

Every stage is independently usable, which is useful for debugging a single
patient or writing custom pipelines:

```python
from coronet.mapping.anatomical_classifier import CoronaryTreeClassifier

classifier = CoronaryTreeClassifier(lcx_bias_mm=5.0, lad_anchor_bias_mm=15.0)
classifier.classify_rca_vs_left("skeleton.nii.gz", "heart.nii.gz", "rca_left.nii.gz")
classifier.split_lad_lcx("rca_left.nii.gz", "heart.nii.gz", "full_classified_tree.nii.gz")
```

## Running tests

```bash
pip install pytest
pytest tests/ -v
```

`tests/test_config.py` and `tests/test_patch_extractor.py` use synthetic,
in-memory volumes and require no GPU, TotalSegmentator, or nnU-Net
installation.

## License

MIT — see `LICENSE` for details.
