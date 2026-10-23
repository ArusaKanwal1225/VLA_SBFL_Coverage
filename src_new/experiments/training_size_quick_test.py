#!/usr/bin/env python3
"""Quick nested training-size experiment on a fixed held-out test sample."""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from vla_sbfl.config import DatasetSelection
from vla_sbfl.dataset import TrajectoryDataset, load_trajectory_dataset
from vla_sbfl.formulas import FORMULA_REGISTRY, TABLE4_FORMULA_IDS
from vla_sbfl.metrics import evaluate_predictions
from vla_sbfl.model import SBFLModel
from vla_sbfl.scoring import TrajectoryScorer
from vla_sbfl.voxelizer import VoxelGrid

REQUESTED_TRAIN_SIZES = (1, 5, 50, 100, 200)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("input/Pi0/DH_200scenes"))
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("output/RQ1/pi0_200/step03_grouped_splits/split_manifest.csv"))
    parser.add_argument("--algorithm", default="DH_200scenes")
    parser.add_argument("--split-seed", type=int, default=11)
    parser.add_argument("--sampling-seed", type=int, default=20261023)
    parser.add_argument("--test-size", type=int, default=100)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("output_new/training_size_quick_test"))
    parser.add_argument("--voxel-size", type=float, default=0.02)
    parser.add_argument("--origin", type=float, nargs=3, default=(-1.0, -1.0, 0.0))
    return parser.parse_args()


def class_allocation(size: int, failure_fraction: float) -> Tuple[int, int]:
    if size < 1:
        raise ValueError("sample size must be positive")
    if size == 1:
        failures = int(failure_fraction >= 0.5)
        return failures, 1-failures
    failures = int(round(size*failure_fraction))
    failures = min(size-1, max(1, failures))
    return failures, size-failures


def shuffled_class_ids(episodes: pd.DataFrame, rng: np.random.Generator):
    failed = episodes.loc[episodes["failure"] == 1, "episode_id"].astype(str).to_numpy()
    passed = episodes.loc[episodes["failure"] == 0, "episode_id"].astype(str).to_numpy()
    return rng.permutation(failed), rng.permutation(passed)


def select_nested_ids(
    failed_ids: np.ndarray,
    passed_ids: np.ndarray,
    sizes: Iterable[int],
    failure_fraction: float,
) -> Dict[int, Tuple[str, ...]]:
    selected = {}
    for size in sorted(set(sizes)):
        n_failed, n_passed = class_allocation(size, failure_fraction)
        if n_failed > len(failed_ids) or n_passed > len(passed_ids):
            raise ValueError("requested sample exceeds the available class counts")
        selected[size] = tuple(sorted(
            [str(value) for value in failed_ids[:n_failed]]
            + [str(value) for value in passed_ids[:n_passed]]
        ))
    return selected


def subset(dataset: TrajectoryDataset, episode_ids: Sequence[str]) -> TrajectoryDataset:
    wanted = set(episode_ids)
    episodes = dataset.episodes[dataset.episodes["episode_id"].isin(wanted)].copy()
    if len(episodes) != len(wanted):
        raise ValueError("one or more selected episode IDs are absent")
    visits = {episode_id: dataset.visits[episode_id] for episode_id in wanted}
    return TrajectoryDataset(episodes=episodes.reset_index(drop=True), visits=visits)


def main() -> int:
    args = parse_args()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    grid = VoxelGrid(args.voxel_size, tuple(args.origin), "robot_base")
    common = dict(
        input_paths=[args.input], grid=grid, require_labels=True,
        chunk_size=250_000)
    train_pool = load_trajectory_dataset(
        selection=DatasetSelection(
            manifest=args.manifest, split="train", seed=args.split_seed,
            algorithm=args.algorithm),
        **common)
    test_pool = load_trajectory_dataset(
        selection=DatasetSelection(
            manifest=args.manifest, split="test", seed=args.split_seed,
            algorithm=args.algorithm),
        **common)

    rng = np.random.default_rng(args.sampling_seed)
    train_failed, train_passed = shuffled_class_ids(train_pool.episodes, rng)
    test_failed, test_passed = shuffled_class_ids(test_pool.episodes, rng)
    experiment_sizes = REQUESTED_TRAIN_SIZES
    train_fraction = float(train_pool.episodes["failure"].mean())
    nested = select_nested_ids(
        train_failed, train_passed, experiment_sizes, train_fraction)
    test_selected = select_nested_ids(
        test_failed, test_passed, (args.test_size,),
        float(test_pool.episodes["failure"].mean()))[args.test_size]
    held_out = subset(test_pool, test_selected)

    selected_rows = []
    for size, episode_ids in nested.items():
        labels = train_pool.episodes.set_index("episode_id").loc[list(episode_ids), "failure"]
        for episode_id, failure in labels.items():
            selected_rows.append({
                "train_size": size,
                "episode_id": episode_id,
                "failure": int(failure),
                "requested_configuration": int(size in REQUESTED_TRAIN_SIZES),
            })
    pd.DataFrame(selected_rows).to_csv(output / "selected_training_episodes.csv", index=False)
    held_out.episodes.to_csv(output / "selected_test_episodes.csv", index=False)

    status_rows: List[dict] = []
    metric_frames: List[pd.DataFrame] = []
    prediction_frames: List[pd.DataFrame] = []
    for size in experiment_sizes:
        training = subset(train_pool, nested[size])
        failures = int(training.episodes["failure"].sum())
        passes = len(training.episodes)-failures
        try:
            model = SBFLModel.fit(
                visits=training.visits,
                labels=training.labels,
                grid=grid,
                formula_ids=TABLE4_FORMULA_IDS,
                metadata={
                    "experiment": "training_size_quick_test",
                    "train_size": size,
                    "sampling_seed": args.sampling_seed,
                    "split_seed": args.split_seed,
                    "algorithm": args.algorithm,
                },
            )
        except ValueError as error:
            status_rows.append({
                "train_size": size,
                "failures": failures,
                "passes": passes,
                "status": "not_trainable",
                "reason": str(error),
                "requested_configuration": int(size in REQUESTED_TRAIN_SIZES),
            })
            continue

        artifact = output / "models" / ("n_"+str(size))
        model.save(artifact)
        predictions = TrajectoryScorer(model).score(held_out.visits, held_out.episodes)
        metrics = evaluate_predictions(predictions, model.formula_ids)
        metrics.insert(0, "train_size", size)
        metrics["train_failures"] = failures
        metrics["train_passes"] = passes
        metrics["mean_known_fraction"] = float(predictions["known_fraction"].mean())
        metrics["no_training_support"] = int(
            (predictions["scoring_status"] == "no_training_support").sum())
        metric_frames.append(metrics)
        predictions.insert(0, "train_size", size)
        prediction_frames.append(predictions)
        status_rows.append({
            "train_size": size,
            "failures": failures,
            "passes": passes,
            "status": "trained",
            "reason": "",
            "requested_configuration": int(size in REQUESTED_TRAIN_SIZES),
        })

    statuses = pd.DataFrame(status_rows).sort_values("train_size")
    metrics = pd.concat(metric_frames, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    names = pd.DataFrame([
        {"formula_id": formula_id, "technique": FORMULA_REGISTRY[formula_id].name}
        for formula_id in TABLE4_FORMULA_IDS
    ])
    metrics = metrics.merge(names, on="formula_id", how="left", validate="many_to_one")
    metrics["roc_auc_rank"] = metrics.groupby("train_size")["roc_auc"].rank(
        method="min", ascending=False).astype(int)
    metrics["ap_rank"] = metrics.groupby("train_size")["average_precision"].rank(
        method="min", ascending=False).astype(int)
    metrics = metrics.sort_values(["train_size", "roc_auc_rank", "formula_id"])
    top = metrics[metrics["roc_auc_rank"] <= 5].copy()
    statuses.to_csv(output / "training_status.csv", index=False)
    metrics.to_csv(output / "metrics.csv", index=False)
    top.to_csv(output / "top_methods_by_size.csv", index=False)
    predictions.to_csv(output / "predictions.csv.gz", index=False, compression="gzip")

    config = {
        "input": str(args.input.resolve()),
        "manifest": str(args.manifest.resolve()),
        "algorithm": args.algorithm,
        "split_seed": args.split_seed,
        "sampling_seed": args.sampling_seed,
        "requested_train_sizes": list(REQUESTED_TRAIN_SIZES),
        "test_size": args.test_size,
        "test_failures": int(held_out.episodes["failure"].sum()),
        "test_passes": int(len(held_out.episodes)-held_out.episodes["failure"].sum()),
        "nested_training_samples": True,
        "same_test_episodes_for_every_model": True,
        "formula_set": "table4",
        "formula_count": len(TABLE4_FORMULA_IDS),
        "one_trajectory_limitation": (
            "A single trajectory has one outcome class. Formulas whose denominator "
            "needs the absent class use the configured undefined-neutral policy, so "
            "the n=1 model is finite but statistically underdetermined."
        ),
    }
    (output / "experiment.json").write_text(
        json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "training_status": status_rows,
        "test_episodes": len(held_out.episodes),
        "metrics_rows": len(metrics),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
