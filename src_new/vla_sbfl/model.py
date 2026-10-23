"""Fitted SBFL model and transparent on-disk artifact format."""

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .formulas import (
    FORMULA_REGISTRY,
    FormulaEvaluation,
    Spectrum,
    evaluate_formula_set,
    formula_catalog_rows,
)
from .voxelizer import Voxel, VoxelGrid, normalize_voxels

ARTIFACT_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def episode_id_hash(episode_ids: Sequence[str]) -> str:
    """Stable identity for detecting accidental train/evaluation reuse."""

    payload = "\n".join(sorted(str(value) for value in episode_ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass
class SBFLModel:
    grid: VoxelGrid
    formula_ids: Tuple[str, ...]
    voxels: np.ndarray
    counts: np.ndarray
    scores: Mapping[str, np.ndarray]
    formula_diagnostics: Mapping[str, Mapping[str, int]]
    training_episode_ids: Tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.voxels = np.asarray(self.voxels, dtype=np.int64)
        self.counts = np.asarray(self.counts, dtype=np.int64)
        if self.voxels.ndim != 2 or self.voxels.shape[1] != 3:
            raise ValueError("voxels must have shape (n, 3)")
        if self.counts.shape != (len(self.voxels), 4):
            raise ValueError("counts must have shape (n, 4)")
        if (self.counts < 0).any():
            raise ValueError("contingency counts cannot be negative")
        if len(self.counts):
            failed_totals = self.counts[:, 0]+self.counts[:, 2]
            passed_totals = self.counts[:, 1]+self.counts[:, 3]
            if np.unique(failed_totals).size != 1 or np.unique(passed_totals).size != 1:
                raise ValueError("all voxel rows must share the same class totals")
            if (self.counts[:, 0]+self.counts[:, 1] <= 0).any():
                raise ValueError("artifact cannot contain training-uncovered voxel rows")
        if len(set(self.formula_ids)) != len(self.formula_ids):
            raise ValueError("formula_ids cannot contain duplicates")
        if set(self.formula_ids) != set(self.scores):
            raise ValueError("score maps do not match formula_ids")
        self.training_episode_ids = tuple(str(value) for value in self.training_episode_ids)
        if len(set(self.training_episode_ids)) != len(self.training_episode_ids):
            raise ValueError("training_episode_ids cannot contain duplicates")
        normalized_scores = {}
        for formula_id in self.formula_ids:
            values = np.asarray(self.scores[formula_id], dtype=np.float64)
            if values.shape != (len(self.voxels),) or not np.isfinite(values).all():
                raise ValueError("invalid score map for " + formula_id)
            normalized_scores[formula_id] = values
        self.scores = normalized_scores
        if len(self.voxels):
            expected = evaluate_formula_set(
                self.formula_ids,
                Spectrum(
                    ef=self.counts[:, 0], ep=self.counts[:, 1],
                    nf=self.counts[:, 2], npass=self.counts[:, 3]))
            for formula_id, result in expected.items():
                if not np.allclose(self.scores[formula_id], result.scores,
                                   rtol=1e-12, atol=1e-12):
                    raise ValueError("stored scores do not match formula " + formula_id)

    @classmethod
    def fit(
        cls,
        visits: Mapping[str, Sequence[Voxel]],
        labels: Mapping[str, int],
        grid: VoxelGrid,
        formula_ids: Sequence[str],
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "SBFLModel":
        """Fit suspiciousness maps using binary training coverage only."""

        if set(visits) != set(labels):
            missing_visits = sorted(set(labels)-set(visits))
            missing_labels = sorted(set(visits)-set(labels))
            raise ValueError("visits and labels differ; missing visits={}, missing labels={}".format(
                missing_visits[:3], missing_labels[:3]))
        normalized_labels = {episode_id: int(value) for episode_id, value in labels.items()}
        if not normalized_labels or set(normalized_labels.values()) - {0, 1}:
            raise ValueError("training labels must be non-empty and binary")
        failed_total = sum(normalized_labels.values())
        passed_total = len(normalized_labels)-failed_total

        failed_visits: Counter = Counter()
        passed_visits: Counter = Counter()
        normalized_visits = {}
        for episode_id, covered in visits.items():
            voxels = normalize_voxels(covered)
            if not voxels:
                raise ValueError("episode has empty voxel coverage: " + episode_id)
            normalized_visits[episode_id] = voxels
            target = failed_visits if normalized_labels[episode_id] == 1 else passed_visits
            target.update(voxels)

        ordered_voxels = sorted(set(failed_visits) | set(passed_visits))
        ef = np.asarray([failed_visits.get(voxel, 0) for voxel in ordered_voxels], dtype=np.int64)
        ep = np.asarray([passed_visits.get(voxel, 0) for voxel in ordered_voxels], dtype=np.int64)
        nf = failed_total-ef
        npass = passed_total-ep
        spectrum = Spectrum(ef=ef, ep=ep, nf=nf, npass=npass)
        evaluated = evaluate_formula_set(formula_ids, spectrum)
        diagnostics = {
            formula_id: {
                "undefined_neutral": result.undefined_count,
                "positive_infinity": result.positive_infinity_count,
                "negative_infinity": result.negative_infinity_count,
            }
            for formula_id, result in evaluated.items()
        }
        combined_metadata = dict(metadata or {})
        combined_metadata.update({
            "training_episodes": len(labels),
            "training_failures": failed_total,
            "training_passes": passed_total,
            "training_has_both_classes": failed_total > 0 and passed_total > 0,
            "training_voxels": len(ordered_voxels),
            "training_episode_ids_sha256": episode_id_hash(tuple(labels)),
        })
        return cls(
            grid=grid,
            formula_ids=tuple(formula_ids),
            voxels=np.asarray(ordered_voxels, dtype=np.int64),
            counts=np.column_stack([ef, ep, nf, npass]),
            scores={formula_id: result.scores for formula_id, result in evaluated.items()},
            formula_diagnostics=diagnostics,
            training_episode_ids=tuple(sorted(labels)),
            metadata=combined_metadata,
        )

    def save(self, artifact_directory: Path) -> None:
        """Write a portable CSV/JSON artifact, with metadata committed last."""

        directory = Path(artifact_directory).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        scores_path = directory / "voxel_scores.csv.gz"
        catalog_path = directory / "formula_catalog.csv"
        episodes_path = directory / "training_episode_ids.csv.gz"
        edge_cases_path = directory / "formula_edge_cases.csv.gz"
        metadata_path = directory / "metadata.json"
        frame = pd.DataFrame({
            "voxel_x": self.voxels[:, 0],
            "voxel_y": self.voxels[:, 1],
            "voxel_z": self.voxels[:, 2],
            "ef": self.counts[:, 0],
            "ep": self.counts[:, 1],
            "nf": self.counts[:, 2],
            "np": self.counts[:, 3],
        })
        for formula_id in self.formula_ids:
            frame[formula_id] = self.scores[formula_id]
        if not np.isfinite(frame[list(self.formula_ids)].to_numpy(dtype=float)).all():
            raise ValueError("artifact cannot contain NaN or infinite suspiciousness")

        scores_tmp = directory / "voxel_scores.csv.tmp.gz"
        catalog_tmp = directory / "formula_catalog.csv.tmp"
        episodes_tmp = directory / "training_episode_ids.csv.tmp.gz"
        edge_cases_tmp = directory / "formula_edge_cases.csv.tmp.gz"
        metadata_tmp = directory / "metadata.json.tmp"
        frame.to_csv(scores_tmp, index=False, compression="gzip", encoding="utf-8")
        pd.DataFrame(formula_catalog_rows(self.formula_ids)).to_csv(
            catalog_tmp, index=False, encoding="utf-8")
        pd.DataFrame({"episode_id": self.training_episode_ids}).to_csv(
            episodes_tmp, index=False, compression="gzip", encoding="utf-8")
        spectrum = Spectrum(
            ef=self.counts[:, 0], ep=self.counts[:, 1],
            nf=self.counts[:, 2], npass=self.counts[:, 3])
        evaluated = evaluate_formula_set(self.formula_ids, spectrum)
        edge_rows = []
        for formula_id, result in evaluated.items():
            for index in np.flatnonzero(result.statuses != "finite"):
                edge_rows.append({
                    "formula_id": formula_id,
                    "voxel_x": int(self.voxels[index, 0]),
                    "voxel_y": int(self.voxels[index, 1]),
                    "voxel_z": int(self.voxels[index, 2]),
                    "status": str(result.statuses[index]),
                    "encoded_score": float(result.scores[index]),
                })
        pd.DataFrame(edge_rows, columns=(
            "formula_id", "voxel_x", "voxel_y", "voxel_z", "status", "encoded_score"
        )).to_csv(edge_cases_tmp, index=False, compression="gzip", encoding="utf-8")
        scores_tmp.replace(scores_path)
        catalog_tmp.replace(catalog_path)
        episodes_tmp.replace(episodes_path)
        edge_cases_tmp.replace(edge_cases_path)
        artifact_metadata = {
            "artifact_version": ARTIFACT_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "grid": self.grid.to_dict(),
            "formula_ids": list(self.formula_ids),
            "formula_diagnostics": self.formula_diagnostics,
            "training": self.metadata,
            "files": {
                scores_path.name: _sha256(scores_path),
                catalog_path.name: _sha256(catalog_path),
                episodes_path.name: _sha256(episodes_path),
                edge_cases_path.name: _sha256(edge_cases_path),
            },
            "score_policy": {
                "voxel_scores": "raw formula values with finite rank-preserving infinity sentinels",
                "trajectory_aggregation": "maximum empirical-percentile suspiciousness among known voxels",
                "partial_unknown_voxels": "ignored and reported",
                "all_unknown_trajectory": "neutral percentile 0.5",
            },
        }
        metadata_tmp.write_text(json.dumps(artifact_metadata, indent=2, sort_keys=True), encoding="utf-8")
        metadata_tmp.replace(metadata_path)

    @classmethod
    def load(cls, artifact_directory: Path) -> "SBFLModel":
        directory = Path(artifact_directory).expanduser().resolve()
        metadata_path = directory / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(metadata_path)
        artifact = json.loads(metadata_path.read_text(encoding="utf-8"))
        if artifact.get("artifact_version") != ARTIFACT_VERSION:
            raise ValueError("unsupported artifact version")
        for filename, expected_hash in artifact.get("files", {}).items():
            path = directory / filename
            if not path.is_file() or _sha256(path) != expected_hash:
                raise ValueError("artifact file is missing or modified: " + filename)
        formula_ids = tuple(artifact["formula_ids"])
        unknown = set(formula_ids)-set(FORMULA_REGISTRY)
        if unknown:
            raise ValueError("artifact references unknown formulas: " + ", ".join(sorted(unknown)))
        frame = pd.read_csv(directory / "voxel_scores.csv.gz", compression="gzip")
        training_episodes = pd.read_csv(
            directory / "training_episode_ids.csv.gz", compression="gzip", dtype={"episode_id": str})
        if set(training_episodes.columns) != {"episode_id"} or training_episodes["episode_id"].duplicated().any():
            raise ValueError("invalid training episode identity file")
        required = {"voxel_x", "voxel_y", "voxel_z", "ef", "ep", "nf", "np", *formula_ids}
        missing = required-set(frame.columns)
        if missing:
            raise ValueError("artifact score columns missing: " + ", ".join(sorted(missing)))
        if frame[["voxel_x", "voxel_y", "voxel_z"]].duplicated().any():
            raise ValueError("artifact contains duplicate voxels")
        return cls(
            grid=VoxelGrid.from_dict(artifact["grid"]),
            formula_ids=formula_ids,
            voxels=frame[["voxel_x", "voxel_y", "voxel_z"]].to_numpy(dtype=np.int64),
            counts=frame[["ef", "ep", "nf", "np"]].to_numpy(dtype=np.int64),
            scores={formula_id: frame[formula_id].to_numpy(dtype=np.float64)
                    for formula_id in formula_ids},
            formula_diagnostics=artifact.get("formula_diagnostics", {}),
            training_episode_ids=tuple(training_episodes["episode_id"]),
            metadata=artifact.get("training", {}),
        )
