"""Model evaluation: metrics, threshold optimization, and patient-level aggregation."""

from coronet.evaluation.evaluator import (
    BinaryVesselEvaluator,
    SequenceClassificationEvaluator,
    ThresholdOptimizer,
)

__all__ = ["BinaryVesselEvaluator", "SequenceClassificationEvaluator", "ThresholdOptimizer"]
