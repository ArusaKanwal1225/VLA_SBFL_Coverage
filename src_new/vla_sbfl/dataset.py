"""Loading and validation for the repository's episode/trajectory CSV schema."""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Set

import numpy as np
import pandas as pd

from .config import DatasetSelection
from .voxelizer import Voxel, VoxelGrid


@dataclass
class TrajectoryDataset:
    episodes: pd.DataFrame
    visits: Mapping[str, FrozenSet[Voxel]]

    @property
    def labels(self) -> Dict[str, int]:
        if "failure" not in self.episodes.columns:
            raise ValueError("dataset does not contain pass/fail labels")
        return {
            str(row.episode_id): int(row.failure)
            for row in self.episodes[["episode_id", "failure"]].itertuples(index=False)
        }


def discover_run_directories(paths: Iterable[Path]) -> List[Path]:
    """Find directories containing both required CSV files."""

    found: Set[Path] = set()
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        if path.is_file():
            if path.name not in {"episodes.csv", "trajectories.csv"}:
                raise ValueError("input files must be episodes.csv or trajectories.csv")
            path = path.parent
        if (path / "episodes.csv").is_file() and (path / "trajectories.csv").is_file():
            found.add(path)
            continue
        for episode_file in path.rglob("episodes.csv"):
            run_dir = episode_file.parent
            if (run_dir / "trajectories.csv").is_file():
                found.add(run_dir.resolve())
    if not found:
        raise ValueError("no run directories containing episodes.csv and trajectories.csv were found")
    return sorted(found)


def _load_episode_tables(run_directories: Sequence[Path], require_labels: bool) -> pd.DataFrame:
    frames = []
    for run_dir in run_directories:
        frame = pd.read_csv(run_dir / "episodes.csv", dtype={"episode_id": str}, encoding="utf-8-sig")
        required = {"episode_id"}
        if require_labels:
            required.add("failure")
        missing = required - set(frame.columns)
        if missing:
            raise ValueError("missing episode columns in {}: {}".format(
                run_dir / "episodes.csv", ", ".join(sorted(missing))))
        if frame.empty or frame["episode_id"].isna().any() or (frame["episode_id"].str.len() == 0).any():
            raise ValueError("episode_id must be non-empty in " + str(run_dir / "episodes.csv"))
        frame = frame.copy()
        frame["_run_directory"] = str(run_dir)
        frames.append(frame)
    episodes = pd.concat(frames, ignore_index=True)
    duplicated = episodes.loc[episodes["episode_id"].duplicated(), "episode_id"]
    if not duplicated.empty:
        raise ValueError("episode_id values must be globally unique; duplicate: " + str(duplicated.iloc[0]))
    if "failure" in episodes.columns:
        numeric = pd.to_numeric(episodes["failure"], errors="coerce")
        if numeric.isna().any() or not numeric.isin([0, 1]).all():
            raise ValueError("failure labels must be binary 0/1")
        episodes["failure"] = numeric.astype(np.int8)
    return episodes


def _apply_selection(episodes: pd.DataFrame, selection: DatasetSelection) -> pd.DataFrame:
    selected = episodes
    if selection.algorithm is not None:
        if "algorithm" not in selected.columns:
            raise ValueError("algorithm selection requested but episodes.csv has no algorithm column")
        selected = selected[selected["algorithm"].astype(str) == selection.algorithm]

    if selection.manifest is not None:
        manifest_path = selection.manifest.expanduser().resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest = pd.read_csv(manifest_path, dtype={"episode_id": str}, encoding="utf-8-sig")
        if "episode_id" not in manifest.columns:
            raise ValueError("split manifest is missing episode_id")
        if selection.seed is not None:
            if "seed" not in manifest.columns:
                raise ValueError("seed selection requested but manifest has no seed column")
            manifest = manifest[pd.to_numeric(manifest["seed"], errors="coerce") == selection.seed]
        elif "seed" in manifest.columns and manifest["seed"].nunique() > 1:
            raise ValueError("manifest contains multiple seeds; select one with --seed")
        if selection.split is not None:
            if "split" not in manifest.columns:
                raise ValueError("split selection requested but manifest has no split column")
            manifest = manifest[manifest["split"].astype(str) == selection.split]
        if selection.algorithm is not None and "algorithm" in manifest.columns:
            manifest = manifest[manifest["algorithm"].astype(str) == selection.algorithm]
        if manifest.empty:
            raise ValueError("the manifest selection contains no episodes")
        if manifest["episode_id"].duplicated().any():
            raise ValueError("the selected manifest contains duplicate episode_id rows")
        if "failure" in manifest.columns and "failure" in selected.columns:
            check = selected[["episode_id", "failure"]].merge(
                manifest[["episode_id", "failure"]], on="episode_id", how="inner",
                suffixes=("_input", "_manifest"), validate="one_to_one")
            manifest_failure = pd.to_numeric(check["failure_manifest"], errors="coerce")
            if not np.array_equal(check["failure_input"].astype(int).to_numpy(),
                                  manifest_failure.astype(int).to_numpy()):
                raise ValueError("manifest and episodes.csv contain different failure labels")
        selected = selected[selected["episode_id"].isin(set(manifest["episode_id"]))]

    if selected.empty:
        raise ValueError("dataset selection contains no episodes")
    return selected.reset_index(drop=True)


def load_trajectory_dataset(
    input_paths: Iterable[Path],
    grid: VoxelGrid,
    selection: Optional[DatasetSelection] = None,
    require_labels: bool = False,
    chunk_size: int = 250_000,
) -> TrajectoryDataset:
    """Load selected episodes and build binary voxel coverage incrementally."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    run_directories = discover_run_directories(input_paths)
    episodes = _load_episode_tables(run_directories, require_labels=require_labels)
    selected = _apply_selection(episodes, selection or DatasetSelection())
    selected_ids = set(selected["episode_id"].astype(str))
    visits: Dict[str, Set[Voxel]] = {episode_id: set() for episode_id in selected_ids}
    sample_counts: Counter = Counter()
    minimum_step: Dict[str, int] = {}
    maximum_step: Dict[str, int] = {}

    for run_dir in run_directories:
        run_ids = set(episodes.loc[episodes["_run_directory"] == str(run_dir), "episode_id"])
        wanted = selected_ids & run_ids
        if not wanted:
            continue
        trajectory_path = run_dir / "trajectories.csv"
        reader = pd.read_csv(
            trajectory_path,
            dtype={"episode_id": str},
            usecols=["episode_id", "step", "x", "y", "z"],
            encoding="utf-8-sig",
            chunksize=chunk_size,
        )
        for chunk in reader:
            chunk = chunk[chunk["episode_id"].isin(wanted)]
            if chunk.empty:
                continue
            xyz = chunk[["x", "y", "z"]].to_numpy(dtype=np.float64)
            indices = grid.transform_points(xyz)
            steps = pd.to_numeric(chunk["step"], errors="coerce").to_numpy()
            if not np.isfinite(steps).all() or (steps < 0).any() or not np.equal(steps, np.floor(steps)).all():
                raise ValueError("trajectory steps must be non-negative integers in " + str(trajectory_path))
            for episode_id, step, index in zip(chunk["episode_id"], steps.astype(np.int64), indices):
                voxel = tuple(int(value) for value in index)
                visits[episode_id].add(voxel)
                sample_counts[episode_id] += 1
                minimum_step[episode_id] = min(minimum_step.get(episode_id, int(step)), int(step))
                maximum_step[episode_id] = max(maximum_step.get(episode_id, int(step)), int(step))

    missing = sorted(episode_id for episode_id, covered in visits.items() if not covered)
    if missing:
        raise ValueError("selected episodes without trajectory data: " + ", ".join(missing[:5]))
    if "num_steps" in selected.columns:
        expected = pd.to_numeric(selected["num_steps"], errors="coerce")
        if expected.isna().any() or (expected <= 0).any():
            raise ValueError("num_steps must contain positive integers")
        for episode_id, expected_count in zip(selected["episode_id"], expected.astype(int)):
            actual = sample_counts[episode_id]
            if actual != expected_count or minimum_step[episode_id] != 0 or maximum_step[episode_id] != expected_count-1:
                raise ValueError("trajectory steps do not match num_steps for " + episode_id)

    frozen = {episode_id: frozenset(covered) for episode_id, covered in visits.items()}
    return TrajectoryDataset(episodes=selected, visits=frozen)

