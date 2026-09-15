#!/usr/bin/env python3
"""Command-line interface for the COROnet pipeline.

Examples
--------
Run heart/coronary segmentation for every raw CT in the configured root::

    python scripts/cli.py segment --root /data/COROnet_Drive

Run anatomical mapping (RCA/LAD/LCx classification + coordinates)::

    python scripts/cli.py map --root /data/COROnet_Drive

Extract 3D patches along every classified vessel::

    python scripts/cli.py extract-patches --root /data/COROnet_Drive

Extract multi-backbone feature sequences::

    python scripts/cli.py extract-features --root /data/COROnet_Drive

Train the binary vessel classifier::

    python scripts/cli.py train-binary --root /data/COROnet_Drive \\
        --split-table /data/COROnet_Drive/vessel_split.csv \\
        --patient-col patient_vessel_id --status-col status --split-col split

Train the plaque-type sequence classifier::

    python scripts/cli.py train-sequence --root /data/COROnet_Drive \\
        --backbone ResNet --architecture bilstm
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from coronet.config import CoronetConfig
from coronet.logging_config import configure_logging, get_logger
from coronet.pipeline.coronet_pipeline import CoronetPipeline

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser with one sub-command per pipeline stage."""
    parser = argparse.ArgumentParser(description="COROnet coronary artery analysis pipeline.")
    parser.add_argument("--root", required=True, help="Root data directory (see PathConfig).")
    parser.add_argument("--config-json", default=None, help="Optional path to a saved CoronetConfig JSON file.")
    parser.add_argument("--log-file", default=None, help="Optional path to also write logs to a file.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("segment", help="Run heart-chamber and coronary artery segmentation.")
    subparsers.add_parser("map", help="Run anatomical (RCA/LAD/LCx) classification and coordinate extraction.")
    subparsers.add_parser("extract-patches", help="Extract 3D patches along every classified vessel.")
    subparsers.add_parser("extract-features", help="Extract multi-backbone feature sequences.")

    train_binary_parser = subparsers.add_parser("train-binary", help="Train the binary vessel classifier.")
    train_binary_parser.add_argument("--split-table", required=True, help="Path to the train/val split CSV/Excel file.")
    train_binary_parser.add_argument("--patient-col", default="patient_vessel_id")
    train_binary_parser.add_argument("--status-col", default="status")
    train_binary_parser.add_argument("--split-col", default="split")

    train_sequence_parser = subparsers.add_parser("train-sequence", help="Train the plaque-type sequence classifier.")
    train_sequence_parser.add_argument("--backbone", required=True, choices=["CNN", "DenseNet", "ResNet", "EfficientNet"])
    train_sequence_parser.add_argument("--architecture", required=True, choices=["lstm", "bilstm", "gru", "transformer"])

    subparsers.add_parser("run-all", help="Run every stage sequentially (segmentation through patch extraction).")

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Parameters
    ----------
    argv:
        Argument list to parse; defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit code (``0`` on success, ``1`` on failure).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    configure_logging(log_file=args.log_file)

    config = CoronetConfig.from_json(args.config_json) if args.config_json else CoronetConfig(root=Path(args.root))
    pipeline = CoronetPipeline(config)

    try:
        if args.command == "segment":
            pipeline.run_segmentation_stage()
        elif args.command == "map":
            pipeline.run_mapping_stage()
        elif args.command == "extract-patches":
            pipeline.run_extraction_stage()
        elif args.command == "extract-features":
            pipeline.run_feature_extraction_stage()
        elif args.command == "train-binary":
            best_f1 = pipeline.train_binary_classifier(
                args.split_table, args.patient_col, args.status_col, args.split_col
            )
            logger.info("Training complete. Best validation macro-F1: %.4f", best_f1)
        elif args.command == "train-sequence":
            metrics = pipeline.train_sequence_classifier(args.backbone, args.architecture)
            logger.info("Training complete. Test metrics: %s", metrics)
        elif args.command == "run-all":
            pipeline.run_segmentation_stage()
            pipeline.run_mapping_stage()
            pipeline.run_extraction_stage()
    except Exception:  # noqa: BLE001 - top-level CLI boundary, log and exit non-zero
        logger.exception("Pipeline command '%s' failed.", args.command)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
