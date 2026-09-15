"""COROnet: an object-oriented pipeline for coronary artery disease analysis on CT.

The package implements a seven-stage pipeline that turns a raw cardiac CT
volume into a per-vessel plaque classification:

1. Heart chamber segmentation (TotalSegmentator).
2. Coronary artery segmentation (UU-Mamba / nnU-Net) and post-processing.
3. Anatomical labelling of the coronary tree (RCA / LAD / LCx).
4. Coordinate and 3D patch extraction along each labelled vessel.
5. Binary (normal / abnormal) vessel classification with a 3D CNN.
6. Multi-backbone feature extraction (CNN, ResNet, DenseNet, EfficientNet).
7. Sequence-based plaque-type classification (LSTM / BiLSTM / GRU / Transformer).

See ``coronet.pipeline.coronet_pipeline.CoronetPipeline`` for the
high-level orchestration class that strings these stages together, and
``README.md`` at the repository root for usage examples.
"""

from coronet.config import CoronetConfig

__version__ = "1.0.0"
__all__ = ["CoronetConfig", "__version__"]
