"""Apply a fitted SBFL model to new trajectories without using outcomes."""

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence

import numpy as np
import pandas as pd

from .model import SBFLModel
from .voxelizer import Voxel


def empirical_percentiles(values: np.ndarray) -> np.ndarray:
    """Tie-aware percentile ranks in [0, 1], with a neutral singleton at 0.5."""

    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or not np.isfinite(data).all() or len(data) == 0:
        raise ValueError("percentiles require a non-empty finite one-dimensional array")
    if len(data) == 1:
        return np.asarray([0.5], dtype=np.float64)
    order = np.argsort(data, kind="mergesort")
    result = np.empty(len(data), dtype=np.float64)
    start = 0
    while start < len(data):
        end = start+1
        while end < len(data) and data[order[end]] == data[order[start]]:
            end += 1
        average_zero_based_rank = 0.5*(start+end-1)
        result[order[start:end]] = average_zero_based_rank/(len(data)-1)
        start = end
    return result


@dataclass
class TrajectoryScorer:
    model: SBFLModel

    def __post_init__(self) -> None:
        self._voxel_index = {
            tuple(int(value) for value in voxel): index
            for index, voxel in enumerate(self.model.voxels)
        }
        self._percentiles = np.column_stack([
            empirical_percentiles(self.model.scores[formula_id])
            for formula_id in self.model.formula_ids
        ])

    def score(
        self,
        visits: Mapping[str, Sequence[Voxel]],
        episodes: pd.DataFrame,
    ) -> pd.DataFrame:
        """Return finite trajectory risk scores and coverage diagnostics."""

        if "episode_id" not in episodes.columns:
            raise ValueError("episodes must contain episode_id")
        if episodes["episode_id"].duplicated().any():
            raise ValueError("episodes contains duplicate episode_id values")
        if set(episodes["episode_id"].astype(str)) != set(visits):
            raise ValueError("episodes and visits contain different episode IDs")
        rows = []
        for episode in episodes.itertuples(index=False):
            episode_id = str(episode.episode_id)
            covered = visits[episode_id]
            known_indices = [self._voxel_index[tuple(voxel)] for voxel in covered
                             if tuple(voxel) in self._voxel_index]
            known = len(known_indices)
            total = len(covered)
            if total == 0:
                risk = np.full(len(self.model.formula_ids), 0.5, dtype=np.float64)
                status = "empty_trajectory"
            elif known:
                risk = np.max(self._percentiles[known_indices, :], axis=0)
                status = "scored"
            else:
                risk = np.full(len(self.model.formula_ids), 0.5, dtype=np.float64)
                status = "no_training_support"
            row = {
                "episode_id": episode_id,
                "episode_voxels": total,
                "known_voxels": known,
                "unknown_voxels": total-known,
                "known_fraction": known/total if total else 0.0,
                "scoring_status": status,
            }
            for formula_id, value in zip(self.model.formula_ids, risk):
                row["score__"+formula_id] = float(value)
            rows.append(row)
        result = pd.DataFrame(rows)
        metadata_columns = [column for column in episodes.columns
                            if column != "episode_id" and not column.startswith("_")]
        if metadata_columns:
            result = result.merge(
                episodes[["episode_id", *metadata_columns]],
                on="episode_id", how="left", validate="one_to_one")
        score_columns = ["score__"+formula_id for formula_id in self.model.formula_ids]
        if not np.isfinite(result[score_columns].to_numpy(dtype=float)).all():
            raise ValueError("trajectory scoring produced a non-finite value")
        return result
