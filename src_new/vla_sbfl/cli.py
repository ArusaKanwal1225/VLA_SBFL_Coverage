"""Shared command-line helpers."""

import argparse
from pathlib import Path

from .config import DatasetSelection


def add_input_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--input", action="append", required=True, type=Path,
        help="Run, algorithm, or model directory. Repeat to combine roots.")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="Optional split manifest containing episode_id")
    parser.add_argument("--split", default=None,
                        help="Manifest split value, normally train or test")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed to select when the manifest contains repeated splits")
    parser.add_argument("--algorithm", default=None,
                        help="Optional algorithm value to select")
    parser.add_argument("--chunk-size", type=int, default=250_000,
                        help="Trajectory CSV rows processed per chunk")


def selection_from_args(args: argparse.Namespace, default_split: str = None) -> DatasetSelection:
    split = args.split
    if args.manifest is not None and split is None:
        split = default_split
    return DatasetSelection(
        manifest=args.manifest,
        split=split,
        seed=args.seed,
        algorithm=args.algorithm,
    )

