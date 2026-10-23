"""Deterministic conversion of XYZ trajectories to binary voxel coverage."""

from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Iterable, Tuple

import numpy as np

Voxel = Tuple[int, int, int]


@dataclass(frozen=True)
class VoxelGrid:
    """A fixed 3-D voxel grid shared by training and evaluation."""

    voxel_size: float
    origin: Tuple[float, float, float]
    coordinate_frame: str = "robot_base"

    def __post_init__(self) -> None:
        if not np.isfinite(self.voxel_size) or self.voxel_size <= 0:
            raise ValueError("voxel_size must be a finite positive number")
        if len(self.origin) != 3 or not np.isfinite(np.asarray(self.origin)).all():
            raise ValueError("origin must contain three finite coordinates")
        if not self.coordinate_frame.strip():
            raise ValueError("coordinate_frame cannot be blank")

    def transform_points(self, xyz: np.ndarray) -> np.ndarray:
        """Return integer voxel indices for an ``(n, 3)`` XYZ array."""

        points = np.asarray(xyz, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("xyz must have shape (n, 3)")
        if not np.isfinite(points).all():
            raise ValueError("xyz contains NaN or infinite coordinates")
        origin = np.asarray(self.origin, dtype=np.float64)
        return np.floor((points - origin) / self.voxel_size).astype(np.int64)

    def transform_trajectory(self, xyz: np.ndarray) -> FrozenSet[Voxel]:
        """Return binary coverage: every visited voxel appears exactly once."""

        indices = self.transform_points(xyz)
        return frozenset(tuple(int(value) for value in row) for row in indices)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "voxel_size": float(self.voxel_size),
            "origin": [float(value) for value in self.origin],
            "coordinate_frame": self.coordinate_frame,
            "rule": "floor((xyz - origin) / voxel_size)",
            "coverage": "binary_per_episode",
        }

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "VoxelGrid":
        return cls(
            voxel_size=float(value["voxel_size"]),
            origin=tuple(float(item) for item in value["origin"]),
            coordinate_frame=str(value["coordinate_frame"]),
        )


def normalize_voxels(voxels: Iterable[Iterable[int]]) -> FrozenSet[Voxel]:
    """Validate and normalize an iterable of voxel identifiers."""

    normalized = []
    for voxel in voxels:
        values = tuple(int(value) for value in voxel)
        if len(values) != 3:
            raise ValueError("each voxel must have exactly three indices")
        normalized.append(values)
    return frozenset(normalized)

