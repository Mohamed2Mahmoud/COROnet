"""Training loops for the binary vessel classifier and the plaque sequence classifier."""

from coronet.training.binary_trainer import VesselClassifierTrainer
from coronet.training.sequence_trainer import PlaqueSequenceTrainer

__all__ = ["VesselClassifierTrainer", "PlaqueSequenceTrainer"]
