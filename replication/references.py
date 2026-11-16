"""Locate released references while retaining compatibility with historical checkouts."""
from pathlib import Path

REFERENCE_NAMES = {
    "RQ1/results/phase_a_primary":"rq1",
    "RQ2_baselines/results/all_models":"rq2",
    "RQ2/results/phase_a_granularity":"rq3",
    "RQ3_v2/results/learning_curves":"rq4",
    "RQ3/results/phase_a_coverage":"coverage",
}

def reference_path(root, historical_directory):
    root=Path(root)
    key=REFERENCE_NAMES.get(str(historical_directory))
    candidate=root/"data/reference"/key if key else None
    if candidate is not None and candidate.is_dir():
        return candidate
    return root/"experiments"/historical_directory
