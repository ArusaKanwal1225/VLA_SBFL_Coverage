#!/usr/bin/env python3
"""Evaluate frozen SBFL trajectory scores with held-out pass/fail labels."""

import argparse
import json
from pathlib import Path

from vla_sbfl.cli import add_input_arguments, selection_from_args
from vla_sbfl.dataset import load_trajectory_dataset
from vla_sbfl.metrics import evaluate_predictions
from vla_sbfl.model import SBFLModel
from vla_sbfl.scoring import TrajectoryScorer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_input_arguments(parser)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--allow-training-data", action="store_true",
        help="Allow evaluation episodes that overlap the fitted training set")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    model = SBFLModel.load(args.artifact)
    dataset = load_trajectory_dataset(
        args.input,
        grid=model.grid,
        selection=selection_from_args(args, default_split="test"),
        require_labels=True,
        chunk_size=args.chunk_size,
    )
    evaluation_ids = set(dataset.episodes["episode_id"].astype(str))
    overlap = evaluation_ids & set(model.training_episode_ids)
    if overlap and not args.allow_training_data:
        raise ValueError(
            "evaluation overlaps training by {} episodes; use held-out data or pass "
            "--allow-training-data explicitly".format(len(overlap)))
    predictions = TrajectoryScorer(model).score(dataset.visits, dataset.episodes)
    metrics = evaluate_predictions(predictions, model.formula_ids)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output / "predictions.csv.gz", index=False,
                       compression="gzip", encoding="utf-8")
    metrics.to_csv(output / "metrics.csv", index=False, encoding="utf-8")
    report = {
        "episodes": len(predictions),
        "failures": int(predictions["failure"].sum()),
        "passes": int(len(predictions)-predictions["failure"].sum()),
        "failure_prevalence": float(predictions["failure"].mean()),
        "formulas": len(model.formula_ids),
        "best_roc_auc_formula": str(metrics.iloc[0]["formula_id"]),
        "best_roc_auc": float(metrics.iloc[0]["roc_auc"]),
        "no_training_support": int((predictions["scoring_status"] == "no_training_support").sum()),
        "mean_known_fraction": float(predictions["known_fraction"].mean()),
        "training_episode_overlap": len(overlap),
        "test_labels_used_for_scoring": False,
        "test_labels_used_only_for_metrics": True,
    }
    (output / "evaluation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
