"""Custom exception hierarchy for the COROnet pipeline.

Using specific exception types (instead of bare ``Exception`` catches,
as in the original notebooks) allows callers to distinguish between
recoverable, per-patient failures and hard configuration errors.
"""

from __future__ import annotations


class CoronetError(Exception):
    """Base class for all COROnet-specific errors."""


class MissingInputError(CoronetError):
    """Raised when a required input file or directory is missing."""


class SegmentationError(CoronetError):
    """Raised when an external segmentation tool fails or produces no output."""


class AnatomicalMappingError(CoronetError):
    """Raised when anatomical (RCA/LAD/LCx) classification cannot proceed."""


class PatchExtractionError(CoronetError):
    """Raised when 3D patch extraction fails for a patient or vessel."""


class ModelCheckpointError(CoronetError):
    """Raised when a model checkpoint cannot be loaded or is incompatible."""


class ConfigurationError(CoronetError):
    """Raised when pipeline configuration is invalid or incomplete."""
