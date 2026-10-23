# VLA-SBFL-Coverage

**Spectrum-Based Fault Localization (SBFL) for Vision-Language-Action (VLA) models using robotic trajectory coverage.**

This repository contains **compact experimental inputs, SBFL-related Python scripts, and generated research results**. It is designed to support SBFL analysis **without downloading VLA checkpoints or rerunning robotic simulators**. Model weights, LeRobot/Transformers frameworks, videos, simulators, and the full raw DeepHyperion experiment directories are intentionally excluded.

## Research questions

- **RQ1 — SBFL technique comparison:** Which of 14 SBFL techniques best distinguishes failed from successful VLA trajectories using spatial coverage?
- **RQ2 — Spatial coverage:** How does the spatial coverage of training trajectories affect the performance and relative ranking of SBFL techniques?

**Scope of the evidence:** ROC-AUC and average precision (AP) measure **held-out trajectory-level failure discrimination**. They do not, by themselves, demonstrate that a physical voxel caused a failure or prove causal fault localization.

## Current progress (23 October 2026)

| Experiment | Input extraction | RQ1 status | Results |
| --- | --- | --- | --- |
| **Pi0 — 200 scenes** | Complete | **Completed (Steps 01–10)** | [`output/RQ1/pi0_200/`](output/RQ1/pi0_200/) |
| **Pi0 — 400 scenes** | Complete | **Completed (Steps 03–10; preceding input preparation/metadata completed)** | [`output/RQ1/pi0_400/`](output/RQ1/pi0_400/) |
| EO1 — 200 / 400 scenes | Complete | Planned / not yet completed | — |
| Groot — 200 / 400 scenes | Complete | Planned / not yet completed | — |
| SpatialVLA — 200 / 400 scenes | Complete | Planned / not yet completed | — |
| RQ2 — spatial coverage comparison | Inputs available | Planned | — |

**Input extraction summary:** 4 VLA models × 4 test-generation algorithms × 2 scene budgets × 10 runs = **320 runs**, containing **96,000 episodes** and **7,680,000 time-ordered TCP positions**. The extraction run reported **320/320 successful runs and 0 failed runs**. These are dataset-extraction results, not 320 completed RQ1 experiments.

## RQ1 findings: Pi0

The table reports **descriptive equal-algorithm mean performance** over four test-generation algorithms and 10 fixed 70/30 split seeds per algorithm. Higher values indicate better discrimination for these metrics.

| Scene budget | Highest mean ROC-AUC technique | Mean ROC-AUC | Mean AP | ROC-AUC runner-up | Runner-up ROC-AUC |
| --- | --- | ---: | ---: | --- | ---: |
| **200** | **Zoltar** | **0.713047** | **0.798002** | Ochiai2 | 0.667142 |
| **400** | **Zoltar** | **0.705812** | **0.818452** | Barinel | 0.675390 |

- **Pi0 / 200:** Zoltar's paired, equal-algorithm mean ROC-AUC advantage over Ochiai2 was approximately **+0.0459**; its algorithm-level difference was positive for **4 of 4** algorithms.
- **Pi0 / 400:** Zoltar's corresponding advantage over Barinel was approximately **+0.0304**, positive for **3 of 4** algorithms and negative for one.
- The 200- and 400-scene sets contain different evaluations. Their results are **descriptively comparable, not paired episode-by-episode**.
- The 10 split seeds reuse episodes, so they must **not** be treated as 10 independent experimental samples for naive statistical significance testing. These results establish descriptive rankings, **not** formal superiority claims.

**Result locations:**

- [`Pi0 200 — final report`](output/RQ1/pi0_200/step10_final_report/)
- [`Pi0 400 — final report`](output/RQ1/pi0_400/step10_final_report/)
- [`Pi0 400 — ROC and PR plots`](output/RQ1/pi0_400/step08_roc_ap/figures/)

### Illustrative ROC figures

The Step 8 ROC figures show **all 14 techniques on a single held-out split** (seed 11), separately for each algorithm. They are **not** the 10-seed average ROC curves.

- [DeepHyperion, Pi0 400, seed 11 — ROC](output/RQ1/pi0_400/step08_roc_ap/figures/05_roc_seed11_DH_400scenes.png)
- [HybridDT, Pi0 400, seed 11 — ROC](output/RQ1/pi0_400/step08_roc_ap/figures/05_roc_seed11_DH_HybridDT_400scenes.png)
- [HybridRF, Pi0 400, seed 11 — ROC](output/RQ1/pi0_400/step08_roc_ap/figures/05_roc_seed11_DH_HybridRF_400scenes.png)
- [WeightedRF, Pi0 400, seed 11 — ROC](output/RQ1/pi0_400/step08_roc_ap/figures/05_roc_seed11_DH_WeightedRF_400scenes.png)

In the DH 400-scene seed-11 example, Tarantula and Barinel had ROC-AUC around **0.755** and Zoltar around **0.712**. This does not contradict Zoltar's overall lead across algorithms and seeds.

## Experimental inputs

| Dimension | Values |
| --- | --- |
| Models | `Pi0`, `EO1`, `Groot`, `SpatialVLA` |
| Test-generation methods | `DH`, `DH_HybridDT`, `DH_HybridRF`, `DH_WeightedRF` |
| Scene budgets | `200`, `400` |
| Source repeats | `run_1` through `run_10` for each combination |
| Outcomes | `failure = 1`: failed; `failure = 0`: successful |
| Trajectories | Time-ordered TCP `(x, y, z)` positions |
| Pi0 voxelization | **0.02 m** voxel size; origin **`(-1, -1, 0)`** |

For a given algorithm, the 10 source runs are pooled for RQ1; the **10 split seeds** produce different reproducible train/test partitions of that pooled dataset. The source run number and split seed serve different purposes.

### Repository structure

```text
VLA_SBFL_Coverage/
├── input/
│   ├── Pi0/
│   │   ├── DH_200scenes/                # run_1 ... run_10
│   │   ├── DH_400scenes/
│   │   ├── DH_HybridDT_200scenes/
│   │   ├── DH_HybridDT_400scenes/
│   │   ├── DH_HybridRF_200scenes/
│   │   ├── DH_HybridRF_400scenes/
│   │   ├── DH_WeightedRF_200scenes/
│   │   ├── DH_WeightedRF_400scenes/
│   │   ├── scene_metadata_candidates_200.csv
│   │   └── scene_metadata_candidates_400.csv
│   ├── EO1/
│   ├── Groot/
│   └── SpatialVLA/
├── src/
│   ├── extract_all_models_sbfl.py
│   ├── 01_rq1_audit_pi0_200.py         # Original Pi0 200 10-step workflow
│   ├── ...
│   ├── 10_rq1_finalize_pi0_200.py
│   ├── RQ1/                             # Optional consolidated Pi0 200 scripts
│   └── RQ1_Pi0_400/                     # Pi0 400 Steps 03–10
├── output/
│   ├── input_validation/
│   └── RQ1/
│       ├── pi0_200/
│       └── pi0_400/
├── README.md
└── requirements.txt
```

The abbreviated layout above describes the principal research files; additional scripts or intermediate folders may be present. Results under `output/` should be retained because they document the frozen experiment, CSV measurements, plots, and validation outcomes.

Example input record layout:

```text
input/Pi0/DH_400scenes/run_1/
├── episodes.csv
├── trajectories.csv
└── validation_summary.json
```

### Input file schemas

`episodes.csv` contains `model`, `algorithm`, `budget`, `run_id`, `eval_id`, `episode_id`, `failure`, `object_name`, and `num_steps`.

`trajectories.csv` contains `episode_id`, `step`, `x`, `y`, and `z`. The globally unique `episode_id` joins trajectories to episode outcomes; **`eval_id` alone is only local to a source run**. Coordinates were extracted from the first three entries of each seven-element TCP pose; the current SBFL voxel representation uses XYZ positions only.

## RQ1 calculation workflow

The same main workflow was applied to **Pi0 200** and **Pi0 400**:

```text
DeepHyperion trajectories + success/failure outcomes
        |
        v
Data audit, scene metadata recovery and validation
        |
        v
Grouped 70% training / 30% testing split (10 fixed seeds)
        |
        +---------------------------+
        |                           |
        v                           v
  TRAINING (70%)              TESTING (30%)
  - Visit 3D voxels            - Visit 3D voxels
  - Compute e_f,e_p,n_f,n_p    - Keep outcomes held out
  - Apply 14 SBFL formulas     - Use frozen training maps
  - Save voxel scores          - Score each test episode
        |                           |
        +-------------+-------------+
                      |
                      v
            Compute ROC-AUC and AP
                      |
                      v
           Compare, rank and report
```

**The 10 experimental stages:**

| Stage | Purpose | Example output |
| --- | --- | --- |
| **01** | Audit episode and trajectory inputs | `step01_data_audit/` |
| **02** | Recover/review scene configuration metadata | `step02_scene_metadata/` |
| **03** | Make reproducible grouped 70/30 splits; verify no shared groups | `step03_grouped_splits/` |
| **04** | Convert XYZ trajectories into binary visited-voxel sets | `step04_voxel_coverage/` |
| **05** | Calculate training-only `e_f`, `e_p`, `n_f`, `n_p` counts | `step05_training_contingency/` |
| **06** | Apply the 14 SBFL formulas to create voxel suspiciousness maps | `step06_suspiciousness/` |
| **07** | Score held-out trajectories using frozen voxel suspiciousness | `step07_heldout_scoring/` |
| **08** | Compute ROC-AUC/AP and generate ROC/PR curves | `step08_roc_ap/` |
| **09** | Paired descriptive comparisons and ranking stability | `step09_paired_comparison/` |
| **10** | Final report and reproducibility checks | `step10_final_report/` |

For Pi0 400, the already-prepared compact dataset and scene metadata were used in place of repeating the full raw-data audit and metadata recovery; **the maintained `src/RQ1_Pi0_400/` scripts begin at Step 03**.

### Suspiciousness calculation and held-out scoring

For each voxel, using **only training episodes**:

- `e_f`: failed training episodes that visited the voxel
- `e_p`: successful training episodes that visited the voxel
- `n_f`: failed training episodes that did not visit the voxel
- `n_p`: successful training episodes that did not visit the voxel

Step 06 evaluates **14 SBFL formulas** (Tarantula, Barinel, Kulczynski2, Wong2, Ochiai2, Ochiai, DStar2, Jaccard, Wong3, GP13, OP2, Wong1, Zoltar, Ample) from these counts. These are formula-based scores: **no VLA model is trained**.

Step 07 assigns a failure-risk score to a held-out trajectory using **the maximum suspiciousness among voxels it visited that were known during training**. The implementation records its policy for unseen voxels and ties. Step 08 uses held-out success/failure labels **only to evaluate the frozen scores**, not to fit them.

### Split seeds and scene leakage

The fixed 10 random seeds are **`11, 23, 37, 41, 53, 67, 79, 83, 97, 101`**. The specific numeric values are arbitrary, predetermined choices. All 14 SBFL techniques see the **same** training/test partition for a given algorithm and seed.

- **Pi0 200:** approximately 1,400 train / 600 test episodes per algorithm/seed.
- **Pi0 400:** approximately 2,800 train / 1,200 test episodes per algorithm/seed.
- Groups were derived from the candidate scene configuration (`object_name`, `orientation_id`, XY scene position). Identical candidate configurations were kept within one partition.
- **Pi0 400:** a reviewed near-duplicate pair in HybridRF was grouped together before splitting. The group-overlap checks reported **zero shared scene groups** in all 40 splits.
- These candidate keys do **not** independently prove that every hidden simulator setting was identical; this remains a protocol limitation.

## Reproduce Pi0 experiments

### Environment

Use a compatible Python environment with `numpy`, `pandas`, `matplotlib`, `scipy`, and `scikit-learn`. The completed Windows analyses used Python 3.10. The supplied `requirements.txt` contains installation constraints; capture the exact active package versions when reproducing results in a new environment.

From the repository root in **PowerShell**:

```powershell
python -m pip install -r requirements.txt
```

### Pi0 400: rerun individual downstream stages

**Use the existing frozen Step 03 splits** unless deliberately starting a new experiment. Regenerating splits may change downstream scores, reports, and checksums.

```powershell
python .\src\RQ1_Pi0_400\04_rq1_voxelize_pi0_400.py --repo-root .
python .\src\RQ1_Pi0_400\05_rq1_train_voxel_counts_pi0_400.py --repo-root .
python .\src\RQ1_Pi0_400\06_rq1_calculate_suspiciousness_pi0_400.py --repo-root .
python .\src\RQ1_Pi0_400\07_rq1_score_heldout_pi0_400.py --repo-root .
python .\src\RQ1_Pi0_400\08_rq1_evaluate_roc_ap_pi0_400.py --repo-root .
python .\src\RQ1_Pi0_400\09_rq1_paired_method_comparison_pi0_400.py --repo-root .
python .\src\RQ1_Pi0_400\10_rq1_finalize_pi0_400.py --repo-root .
```

The grouped split manifest is at `output/RQ1/pi0_400/step03_grouped_splits/split_manifest.csv`. The reviewed scene grouping should **not** be reconfirmed or recreated without inspecting the exact/near-duplicate review files and documenting the decision. For the Pi0 200 workflow, the corresponding original scripts are `src/01_rq1_...pi0_200.py` through `src/10_rq1_finalize_pi0_200.py`.

### Where to find the plots and data

- **Voxel suspiciousness maps:** `output/RQ1/pi0_400/step06_suspiciousness/voxel_suspiciousness_scores.csv.gz`
- **Held-out predictions:** `output/RQ1/pi0_400/step07_heldout_scoring/test_episode_scores.csv.gz`
- **ROC-AUC and AP per method/split:** `output/RQ1/pi0_400/step08_roc_ap/per_split_metrics.csv`
- **ROC figures:** `output/RQ1/pi0_400/step08_roc_ap/figures/05_roc_seed11_*.png`
- **PR figures:** `output/RQ1/pi0_400/step08_roc_ap/figures/06_pr_seed11_*.png`
- **Final method ranking:** `output/RQ1/pi0_400/step10_final_report/final_method_leaderboard.csv`
- **Final written interpretation:** `output/RQ1/pi0_400/step10_final_report/step10_final_rq1_research_report.md`

CSV data, figure source points, PNG/PDF plots, and validation reports are stored alongside the outputs of their respective processing stage. The optional `consolidated/` folders are separate from the **original 10-step validation trail**.

## RQ2: planned experimental protocol

RQ2 has **not** been reported as completed. The proposed coverage-controlled analysis will:

1. Reuse frozen outer RQ1 splits and the same held-out test episodes.
2. Build training subsets with equal/comparable numbers of episodes, outcome balance and measured low/medium/high voxel coverage.
3. Recalculate all 14 suspiciousness maps from each coverage condition's training subset.
4. Score the same test trajectories and compare ROC-AUC/AP, ranking changes and achieved coverage.
5. Report coverage differences as associations under the chosen subset design, not automatically as causal effects.

## Re-extracting inputs (optional)

`src/extract_all_models_sbfl.py` creates compact input tables from original `dh_evaluations.csv` and per-scene `tcp_poses.json` files. The source `Data/` directory and simulation assets are **not** distributed with this repository. Re-extraction requires separately available raw DeepHyperion outputs, but **is not necessary** to rerun the existing RQ1 analysis on the included compact `input/` files.

## Reproducibility and research limitations

- Preserve the provided split manifests, episode IDs, scene-group review records, seeds, code versions, output CSVs, and figures.
- Never use held-out test labels to fit voxel suspiciousness scores or select favorable models/parameters.
- The reported Zoltar rankings are **descriptive** for Pi0; their generality across EO1, Groot and SpatialVLA remains to be tested.
- Split seeds create repeated partitions of overlapping episodes, not independent replications. Formal inferential analysis requires an appropriate dependency-aware design.
- Scene identification comes from available configuration fields and has not been proved to cover all simulator state.
- The optional consolidated Pi0 200 scripts previously showed a small maximum difference of approximately **0.000245** against the original 10-step metrics; keep the original 10-step outputs as the reference until that difference is resolved.
- Preserve raw source data outside the SBFL-only repository. Do not upload checkpoints, full simulator folders, videos, model weights or virtual environments.
- Before GitHub uploads, check individual file and push-size limits; use an explicitly documented large-data hosting strategy if necessary.

## License and citation

Add project-maintainer-approved license, attribution and research citation information before publication. This README does not grant third-party data redistribution rights.
