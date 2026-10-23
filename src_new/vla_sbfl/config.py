"""Configuration objects shared by the command-line entrypoints."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class DatasetSelection:
    """Optional filters applied before trajectories are loaded."""

    manifest: Optional[Path] = None
    split: Optional[str] = None
    seed: Optional[int] = None
    algorithm: Optional[str] = None

    def __post_init__(self) -> None:
        if self.manifest is None and (self.split is not None or self.seed is not None):
            raise ValueError("split and seed require a split manifest")
        if self.split is not None and not self.split.strip():
            raise ValueError("split cannot be blank")
        if self.algorithm is not None and not self.algorithm.strip():
            raise ValueError("algorithm cannot be blank")

