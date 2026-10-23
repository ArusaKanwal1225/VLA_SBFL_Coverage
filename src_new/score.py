#!/usr/bin/env python3
"""Score new trajectories with a frozen SBFL artifact; labels are optional."""

import argparse
import json
from pathlib import Path

from vla_sbfl.cli import add_input_arguments, selection_from_args
from vla_sbfl.dataset import load_trajectory_dataset
from vla_sbfl.model import SBFLModel
from vla_sbfl.scoring import TrajectoryScorer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_input_arguments(parser)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path,
                        help="Prediction CSV; use .csv.gz for gzip compression")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    model = SBFLModel.load(args.artifact)
    dataset = load_trajectory_dataset(
        args.input,
        grid=model.grid,
        selection=selection_from_args(args, default_split="test"),
        require_labels=False,
        chunk_size=args.chunk_size,
    )
    predictions = TrajectoryScorer(model).score(dataset.visits, dataset.episodes)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    compression = "gzip" if output.name.endswith(".gz") else None
    predictions.to_csv(output, index=False, encoding="utf-8", compression=compression)
    summary = {
        "output": str(output),
        "episodes": len(predictions),
        "formulas": len(model.formula_ids),
        "no_training_support": int((predictions["scoring_status"] == "no_training_support").sum()),
        "mean_known_fraction": float(predictions["known_fraction"].mean()),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

