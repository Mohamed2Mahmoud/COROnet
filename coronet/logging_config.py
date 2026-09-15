"""Centralized logging configuration for the COROnet pipeline.

Every module in the package obtains its logger through
:func:`get_logger` instead of using bare ``print`` statements, which
makes log verbosity configurable and log output redirectable to a file
in production deployments.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

_CONFIGURED = False


def configure_logging(
    level: int = logging.INFO,
    log_file: Optional[str | Path] = None,
) -> None:
    """Configure the root ``coronet`` logger once per process.

    Parameters
    ----------
    level:
        Minimum severity level to emit, e.g. ``logging.DEBUG``.
    log_file:
        Optional path to a file that should also receive log records.
        Parent directories are created automatically.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    root_logger = logging.getLogger("coronet")
    root_logger.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger under the ``coronet`` hierarchy.

    Parameters
    ----------
    name:
        Usually ``__name__`` of the calling module.

    Returns
    -------
    logging.Logger
        A configured logger instance. If :func:`configure_logging` has
        not been called yet, it is invoked with default settings.
    """
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(f"coronet.{name}")
