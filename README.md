# VLA-SBFL-Coverage

**Spectrum-Based Fault Localization (SBFL) for Vision-Language-Action (VLA) models.**

This repository contains **only** the prepared experiment inputs, SBFL-related Python scripts, and analysis results.

## Research questions

- **RQ1:** Which of the 14 SBFL techniques provides the strongest held-out failure discrimination from spatial trajectory coverage?
- **RQ2:** How does training spatial coverage affect the performance and ranking of the 14 SBFL techniques?

**Interpretation:** ROC-AUC and average precision (AP) evaluate how well a trajectory-level score distinguishes failed from successful executions. They do **not**, on their own, establish which physical voxel caused a failure.

## Experimental inputs

| Dimension | Values |
| --- | --- |
| VLA models | `Pi0`, `EO1`, `Groot`, `SpatialVLA` |
| Test-generation algorithms | `DH`, `DH_HybridDT`, `DH_HybridRF`, `DH_WeightedRF` |
| Scene budgets | `200`, `400` |
| Repeated runs | `run_1` through `run_10` per model/algorithm/budget |
| Episode labels | `failure = 1` (failed), `failure = 0` (successful) |
| Trajectory representation | Time-ordered TCP positions `(x, y, z)` |

**Extracted input summary (22 October 2026):** 320 runs, 96,000 episodes, and 7,680,000 TCP position records; the all-model extraction reported 0 failed runs. These are extraction counts; consult the validation reports for group-level outcomes.

## Repository layout

```text
VLA_SBFL_Coverage/
├── input/
│   ├── Pi0/
│   ├── EO1/
│   ├── Groot/
│   └── SpatialVLA/
├── src/
│   ├── extract_all_models_sbfl.py
│   └── ... (SBFL preprocessing and RQ1/RQ2 scripts)
├── output/
│   ├── input_validation/
│   │   └── all_models_summary.csv
│   ├── RQ1/
│   └── RQ2/
├── README.md
└── requirements.txt
```

Each model directory follows this structure (example):

```text
input/Pi0/DH_200scenes/run_1/
├── episodes.csv
├── trajectories.csv
└── validation_summary.json
```

### Input file schemas

`episodes.csv` contains:

| Column | Meaning |
| --- | --- |
| `model` | VLA model name |
| `algorithm` | Test-generation method and scene budget |
| `budget` | Nominal scene budget (`200` or `400`) |
| `run_id` | Original run (`run_1`, etc.) |
| `eval_id` | Evaluation ID **within the run** |
| `episode_id` | Globally unique identifier combining model, algorithm, run, and evaluation |
| `failure` | Binary outcome: 1 = failure; 0 = success |
| `object_name` | Manipulation target from original evaluation record |
| `num_steps` | Number of TCP poses extracted for that episode |

`trajectories.csv` contains `episode_id`, `step`, `x`, `y`, `z`. The `episode_id` value links each position to exactly one episode in `episodes.csv`. `step` preserves trajectory order. The coordinates are extracted from the first three values of each seven-element `tcp_poses.json` pose; quaternion/orientation data is not used in the current position-based SBFL setup.

`validation_summary.json` records extraction counts for each run. `output/input_validation/all_models_summary.csv` aggregates the run summaries.

## Setup

Use **Python 3.9** or another version compatible with the versions in `requirements.txt`.

```bash
python -m venv .venv
# Windows PowerShell:
# .\.venv\Scripts\Activate.ps1
# Git Bash on Windows:
# source .venv/Scripts/activate
python -m pip install -r requirements.txt
```

The input extraction scripts use Python's **standard library** (`csv`, `json`, `math`, `pathlib`, etc.). The packages in `requirements.txt` are for current or planned numerical analysis, ROC-AUC/AP calculation, statistics, and figures.

## Working with the prepared dataset

The clean `input/` directory is designed to support the SBFL pipeline **without loading the original models or rerunning a simulator**.

1. Read each run's `episodes.csv` and `trajectories.csv`.
2. Link rows with the globally unique `episode_id` (do not use `eval_id` across runs).
3. Convert TCP XYZ positions to spatial voxels using one fixed coordinate frame, voxel size, and grid origin. The established initial π0 protocol uses a **0.02 m** voxel size and origin **`[-1, -1, 0]`**; retain these only after checking that the same coordinate convention is valid for the other models.
4. Generate reproducible train/test splits and fit voxel suspiciousness **only on training labels**.
5. Convert voxel suspiciousness to test-episode risk scores and evaluate on untouched held-out outcomes.
6. Save all intermediate CSVs, configurations, figure data, and plots under `output/RQ1/` and `output/RQ2/`.

### RQ1 protocol

- Compare all **14 SBFL techniques** on identical data partitions.
- Use an approximately **70% training / 30% testing** split with an appropriate grouping variable to avoid scene leakage; an `eval_id` alone does not prove two episodes correspond to different scenes.
- Use fixed split seeds: `11, 23, 37, 41, 53, 67, 79, 83, 97, 101`.
- Estimate per-voxel suspiciousness from training-set failed/passed visitation counts (`e_f`, `e_p`, `n_f`, `n_p`).
- Score held-out episodes (e.g., maximum suspiciousness among visited training-known voxels), with a documented policy for unseen voxels and ties.
- Report ROC-AUC, average precision (AP), failure prevalence, per-split results, algorithm-level variation, and paired comparisons.

### RQ2 protocol

- Reuse the same RQ1 outer splits and **fixed held-out test episodes**.
- Construct comparable-size, outcome-balanced training subsets with lower, medium, and higher **measured** voxel coverage.
- Refit all 14 SBFL maps for each subset; do not reuse suspiciousness learned from a different coverage condition.
- Track achieved voxel coverage, voxel support, train/test overlap (diagnostic only), ROC-AUC/AP, and changes in method rankings.
- Coverage–performance differences from selected training subsets are associations under this design, not automatically proof of causality.

**Status:** The extraction and run-summary steps have been completed. The RQ1/RQ2 analysis commands will be documented here as their scripts are added and validated; no unimplemented evaluation command is implied.

## Re-extracting inputs (optional)

`src/extract_all_models_sbfl.py` is a **data-preparation script** used to construct the compact inputs from original DeepHyperion outputs. In the working project from which this repository was prepared, it expects a `Data/` directory adjacent to `src/`, containing `dh_evaluations.csv` and per-scene `tcp_poses.json` files.

The original `Data/` folder is **not included** in this SBFL-only GitHub repository. Re-extraction therefore requires separately obtaining the raw DeepHyperion results and adapting the source path if stored elsewhere. Re-extraction is **not required** for analyses that consume the included `input/` CSV files.

## Reproducibility and scope

- Keep models, algorithms, budgets, run IDs, outcome labels, and trajectory order intact.
- Keep training and testing data separated; never use test outcomes to fit suspiciousness scores or select coverage subsets.
- Preserve original raw data in separate local storage rather than deleting it after producing the compact inputs.
- Track configurations, seeds, exclusions, and validation outcomes with each experiment.
- Do not commit `.venv/`, model checkpoints, simulator source, raw experiment videos, or full unfiltered `Data/` directories.
- Before pushing datasets, verify the repository size and individual file sizes against GitHub limits. If necessary, decide on data hosting/LFS separately rather than silently omitting input records.

## License / citation

Add the appropriate project license, attribution, and research citation once approved by the project maintainers. This README makes no claim about third-party dataset redistribution permissions.
