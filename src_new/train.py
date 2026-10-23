#!/usr/bin/env python3
"""Fit voxel suspiciousness maps from labelled training trajectories."""

import argparse
import json
from pathlib import Path

from vla_sbfl.cli import add_input_arguments, selection_from_args
from vla_sbfl.dataset import load_trajectory_dataset
from vla_sbfl.formulas import formula_ids_for_set
from vla_sbfl.model import SBFLModel
from vla_sbfl.voxelizer import VoxelGrid


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_input_arguments(parser)
    parser.add_argument("--artifact", required=True, type=Path,
                        help="Directory in which to save the fitted model")
    parser.add_argument("--formula-set", choices=("table4", "legacy14", "all"),
                        default="table4", help="Formula catalogue to fit")
    parser.add_argument("--voxel-size", type=float, default=0.02)
    parser.add_argument("--origin", type=float, nargs=3, metavar=("X", "Y", "Z"),
                        default=(-1.0, -1.0, 0.0))
    parser.add_argument("--coordinate-frame", default="robot_base")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    selection = selection_from_args(args, default_split="train")
    grid = VoxelGrid(
        voxel_size=args.voxel_size,
        origin=tuple(args.origin),
        coordinate_frame=args.coordinate_frame,
    )
    dataset = load_trajectory_dataset(
        args.input,
        grid=grid,
        selection=selection,
        require_labels=True,
        chunk_size=args.chunk_size,
    )
    formula_ids = formula_ids_for_set(args.formula_set)
    selection_metadata = {
        "input_roots": [str(path.expanduser().resolve()) for path in args.input],
        "manifest": str(args.manifest.expanduser().resolve()) if args.manifest else None,
        "split": selection.split,
        "seed": selection.seed,
        "algorithm": selection.algorithm,
        "formula_set": args.formula_set,
    }
    model = SBFLModel.fit(
        visits=dataset.visits,
        labels=dataset.labels,
        grid=grid,
        formula_ids=formula_ids,
        metadata=selection_metadata,
    )
    model.save(args.artifact)
    summary = {
        "artifact": str(args.artifact.expanduser().resolve()),
        "episodes": len(dataset.episodes),
        "failures": int(dataset.episodes["failure"].sum()),
        "passes": int(len(dataset.episodes)-dataset.episodes["failure"].sum()),
        "voxels": len(model.voxels),
        "formulas": len(model.formula_ids),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

