"""Memory management helpers for long-running batch jobs over many patients."""

from __future__ import annotations

import gc
from typing import Any

from coronet.logging_config import get_logger

logger = get_logger(__name__)

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - torch is a hard runtime dependency in practice
    _TORCH_AVAILABLE = False


def clear_gpu_memory() -> None:
    """Force garbage collection and empty the CUDA cache, if available.

    This mirrors the ``clear_gpu_memory`` helper used throughout the
    original notebooks between heavy per-patient inference calls, but is
    now a reusable, well-documented utility.
    """
    gc.collect()
    if _TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.debug("Cleared CUDA cache and synchronized device.")


def release(*objects: Any) -> None:
    """Delete references to large objects and trigger garbage collection.

    Parameters
    ----------
    *objects:
        Objects that should be dereferenced (e.g. large NumPy arrays)
        before the next heavy operation. Passing an already-deleted or
        undefined name is safe; each release is attempted independently.
    """
    del objects
    gc.collect()
