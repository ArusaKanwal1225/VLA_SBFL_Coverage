<div align="center">

# VoxelSBFL: Voxel-Level Spectrum-Based Failure Prediction for Vision-Language-Action Models

> [Arusa Kanwal](https://scholar.google.com/citations?hl=en&user=OtQVrtAAAAAJ)<sup>1</sup>, [Pablo Valle](https://scholar.google.com/citations?user=-3y0BlAAAAAJ&hl=en)<sup>1</sup>, [Paola Roggero](https://scholar.google.es/citations?hl=es&user=_ZXnvWEAAAAJ)<sup>1</sup>, [Miren Illarramendi](https://scholar.google.com/citations?hl=en&user=GN9Lgs8AAAAJ)<sup>1</sup>, [Shaukat Ali](https://scholar.google.com/citations?user=S_UVLhUAAAAJ&hl=en)<sup>2</sup>, [Aitor Arrieta](https://scholar.google.com/citations?user=ft06jF4AAAAJ&hl=en)<sup>1</sup><br>
> Mondragon Unibertsitatea<sup>1</sup>, Simula Research Laboratory<sup>2</sup>

[📦 Replication Data](#provided-datasets) · [🚀 Installation](#installation) · [📊 Reproduce the Results](#usage) · [📚 Citation](#citation)

</div>

In this paper, we introduce **VoxelSBFL**, a black-box approach that adapts Spectrum-Based Fault Localization (SBFL) to Vision-Language-Action (VLA) robots. Recorded tool-center-point trajectories provide voxel coverage, which is combined with successful and failed execution outcomes to build spatial suspiciousness maps. These maps score unseen trajectories according to the regions they visit.

Our evaluation compares **34 SBFL techniques** across **four VLA models**, using **96,000 Pick-Up executions**. We compare the techniques against action- and trajectory-based baselines, investigate voxel-size and training-set-size sensitivity, and assess transfer to Move-Near using **1,109 selected target-task executions**.

This repository provides the **offline replication package**: analysis scripts, frozen configurations, integrity checks, and automated CSV/LaTeX table and figure generation. The analysis runs on **CPU** using recorded executions; loading VLA models or running a simulator is not required.

## Index

1. [Repo Structure](#repo-structure)
2. [Provided Datasets](#provided-datasets)
3. [Hardware and Software Requirements](#hardware-and-software-requirements)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Troubleshooting](#troubleshooting)
7. [Citation](#citation)
8. [Acknowledgements](#acknowledgements)

## Repo Structure

The repository is organized as follows:

| Path | Description |
| --- | --- |
| [replicate.py](replicate.py) | Main entry point for data verification, table generation, statistical analysis, and evaluation. |
| [experiments](experiments) | Scientific protocols and scripts for the five research questions. |
| [src_new](src_new) | Reusable SBFL implementation: voxelization, spectra, formulas, scoring, metrics, and scientific tests. |
| [replication](replication) | Dataset extraction/checksums, paper-table exports, voxel-map plots, bootstrap audit, and release metadata. |
| [docs](docs) | Detailed replication instructions, paper-artifact mapping, provenance, and Zenodo publication guidance. |
| [requirements.txt](requirements.txt) | Recorded dependency versions for the CPU analysis environment. |
| [CITATION.cff](CITATION.cff) | Citation metadata for the replication software. |

After extracting the dataset and executing an analysis, the local layout includes:

~~~text
VLA_SBFL_Coverage/
├── replicate.py
├── experiments/
│   ├── RQ1/                 # Paper RQ1: SBFL technique comparison
│   ├── RQ2_baselines/       # Paper RQ2: baseline comparison
│   ├── RQ2/                 # Paper RQ3: voxel-size sensitivity
│   ├── RQ3_v2/              # Paper RQ4: training-size / coverage
│   └── RQ5/                 # Paper RQ5: cross-task transfer
├── src_new/
├── replication/
├── docs/
├── data/                    # Extracted Zenodo data; ignored by Git
│   ├── pickup/
│   ├── rq5/
│   ├── reference/
│   └── files_manifest.csv
└── reproduced/              # Generated outputs; ignored by Git
~~~

> **Directory naming:** The source tree retains historical folder names. Command-line options use the **paper's numbering**: <code>--rq rq3</code> selects voxel-size sensitivity, and <code>--rq rq4</code> selects the learning-curve study. The older <code>experiments/RQ3</code> directory supplies compatibility helpers/tests, rather than an additional paper experiment.

## Provided Datasets

The benchmark covers four VLA models: [EO-1](https://github.com/SHAILAB-IPEC/EO1), [Isaac GR00T](https://github.com/NVIDIA/Isaac-GR00T), [pi0](https://github.com/Physical-Intelligence/openpi), and [SpatialVLA](https://github.com/SpatialVLA/SpatialVLA).

Data are distributed separately from the code as **vla-sbfl-data-20261116.zip**:

| Component | Contents |
| --- | --- |
| **Pick-Up executions** | 96,000 episodes and 7,680,000 ordered TCP positions: four models, four trajectory generators, budgets of 200/400, and ten runs per configuration. |
| **Baseline evidence** | Original per-execution metric JSON files and outcome records. |
| **Move-Near executions** | 504 candidates per model, totaling 2,016 executions, with trajectories, outcomes, scenes, and sample manifests. |
| **Frozen reconstructed maps** | The 320 Pick-Up map artifacts used for the RQ5 transfer evaluation. |
| **Reference artifacts** | Historical audits, aggregate metrics, exact statistical reviews, publication artifacts, and voxel-count diagnostics. |
| **Integrity manifest** | Relative paths, byte sizes, and SHA-256 fingerprints for every payload file. |

### Data availability

**The archive is prepared for Zenodo.** Its download URL and DOI will be added after publication. Until then, obtain the ZIP from the authors. [replication/archive.json](replication/archive.json) records the exact filename, size, and checksum.

The ZIP occupies approximately **3.43 GB**, containing approximately **10.80 GB** of uncompressed files. Small-file filesystem overhead requires additional space. Execution videos, model checkpoints, duplicate raw TCP/action streams, and simulator dependencies are excluded.

<details>
<summary><b>Archive integrity</b></summary>

SHA-256:

~~~text
329453c7fb0dcd19934d0b3e3e44e0f5a073c88e12d0a98b0e57613a29343da0
~~~

The extraction tool verifies this checksum and each payload's SHA-256. It rejects unsafe paths and refuses to overwrite an existing nonempty data directory.

</details>

### Final RQ5 sample

| Model | Candidate executions | Selected pass | Selected fail | Selected N |
| --- | ---: | ---: | ---: | ---: |
| EO1 | 504 | 100 | 123 | 223 |
| GR00T | 504 | 100 | 146 | 246 |
| pi0 | 504 | 100 | 198 | 298 |
| SpatialVLA | 504 | 100 | 242 | 342 |
| **Total** | **2,016** | **400** | **709** | **1,109** |

Selection uses a fixed seed and spatial constraints to approximate each model's Pick-Up outcome proportions. Integer counts do not preserve those percentages with mathematical exactness.

RQ5 applies the **released reconstructed Pick-Up maps** without Move-Near retraining or calibration. These are not the exact original historical RQ1 map files. Original artifact metadata and fingerprints are preserved; see [map provenance](experiments/RQ5/PROVENANCE.md).

## Hardware and Software Requirements

| Requirement | Details |
| --- | --- |
| **Python** | Python 3.11; validation used Python 3.11.16. |
| **Dependencies** | Pinned in [requirements.txt](requirements.txt). |
| **CPU** | CPU-only analysis; GPU/CUDA setup is not required. |
| **Memory** | Start with 8 GiB RAM; full runs may require more. |
| **Storage** | Allow at least 30 GiB free for extraction overhead and outputs; full fitting may need additional space. |
| **Validated platform** | Linux x86_64. Windows activation commands are included below. |

Table generation and aggregate-only statistics normally take minutes. Full evaluation fits hundreds of maps and 16,000 learning-curve cells, with runtime depending on CPU and storage performance.

## Installation

### 1. Clone the repository

Use a shallow clone to avoid dataset blobs retained in older Git commits:

~~~bash
git clone --depth 1 https://github.com/ArusaKanwal1225/VLA_SBFL_Coverage.git
cd VLA_SBFL_Coverage
~~~

### 2. Install the analysis environment

<details open>
<summary><b>Python virtual environment — Linux / macOS</b></summary>

~~~bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
~~~

</details>

<details>
<summary><b>Python virtual environment — Windows PowerShell</b></summary>

~~~powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
~~~

</details>

<details>
<summary><b>Conda environment</b></summary>

~~~bash
conda create -n voxelsbfl python=3.11 -y
conda activate voxelsbfl
python -m pip install -r requirements.txt
~~~

This is a dedicated analysis environment. Model-specific environments such as PI0 or GR00T are not needed.

</details>

### 3. Install the dataset

After obtaining the archive:

~~~bash
python replication/data.py unpack /path/to/vla-sbfl-data-20261116.zip
~~~

The tool extracts and verifies the files under <code>data/</code>.

<details>
<summary><b>Automatic download after Zenodo publication</b></summary>

Once the published URL is recorded in <code>replication/archive.json</code>:

~~~bash
python replication/data.py download
python replication/data.py unpack vla-sbfl-data-20261116.zip
~~~

Automatic download is unavailable while the Zenodo record is pending.

</details>

## Usage

Run commands from the **repository root**, with the analysis environment activated.

### Quick start: reproduce the paper tables

~~~bash
python replicate.py tables
~~~

The main outputs are written to **reproduced/tables/paper/**:

- **table_02.tex through table_15.tex:** evaluation tables numbered according to the final paper.
- **figure_05.pdf and figure_05.png:** the four-model voxel suspiciousness visualization.
- **Supporting CSV/JSON files:** selected counts, aggregated voxels, fingerprints, and table mappings.

Source reports, complete exact-review tables, and supplementary learning-curve figures are retained in their RQ output directories. LaTeX requires <code>booktabs</code>; complete review tables also use <code>longtable</code>.

Table 1 and Figures 1–4 are worked examples, diagrams, or simulator observations. They are not offline evaluation outputs. See [the paper-artifact map](docs/PAPER_ARTIFACTS.md) for the correspondence.

<details>
<summary><b>Verify data, outcomes, and frozen maps</b></summary>

~~~bash
python replicate.py verify
~~~

This checks payload hashes and validates RQ5 final simulator outcomes, scene/prompt fingerprints, maps, selected class counts, and spatial-cell quotas.

</details>

<details>
<summary><b>Recompute exact statistical results</b></summary>

~~~bash
python replicate.py stats
~~~

This regenerates exact tests from the released aggregate observations, compares them against the references, and exports paper tables under <code>reproduced/stats/</code>.

| RQ | Holm correction family |
| --- | --- |
| RQ1 | All 561 unordered pairs among 34 techniques. |
| RQ2 | Eight baseline comparisons against Harmonic Mean. |
| RQ3 | 68 technique-size effects and 66 interactions, corrected separately. |
| RQ4 | Four adjacent-size, 33 sample-efficiency, and 34 conditional-coverage tests, corrected separately. |

This mode does not refit maps or regenerate bootstrap intervals. Paper tables combine the approved exact p-values with original pointwise confidence intervals; historical reference files remain unchanged.

</details>

<details>
<summary><b>Run a short pipeline check</b></summary>

~~~bash
python replicate.py smoke
~~~

This checks one RQ1 fold, one baseline fold, fine/coarse voxel-size folds, and one nested learning-curve repetition across all five training sizes. For RQ5 it regenerates tables; full rescoring is performed by <code>run</code>.

Outputs are written to <code>reproduced/smoke/</code>. This checks execution, not full-design significance.

</details>

<details>
<summary><b>Recompute the full evaluation</b></summary>

~~~bash
python replicate.py run
~~~

The workflow evaluates:

- **RQ1:** 320 leave-one-run-out folds and 34 SBFL techniques.
- **RQ2:** baseline metrics on the matched Pick-Up executions.
- **RQ3:** 640 additional folds at 0.01/0.05 m, reusing the 0.02 m RQ1 results.
- **RQ4:** 16,000 nested-suite cells and 544,000 technique-level measurements.
- **RQ5:** the selected Move-Near samples, using the exact released reconstructed maps.

Outputs are written to <code>reproduced/run/</code>, including exact reviews, tables, figures, and <code>replication_run.json</code>.

</details>

<details>
<summary><b>Select RQs, choose outputs, and resume</b></summary>

~~~bash
# RQ1 and baseline comparison:
python replicate.py run --rq rq1 rq2 --output reproduced/pickup-review

# Complete RQ5 transfer analysis:
python replicate.py run --rq rq5 --output reproduced/transfer-review

# Exact statistics for RQ1–RQ4:
python replicate.py stats --rq rq1 rq2 rq3 rq4 --output reproduced/statistics-review

# Tables without figures:
python replicate.py tables --no-figures --output reproduced/tables-review

# Preview execution commands:
python replicate.py run --dry-run

# Resume the default full run:
python replicate.py run --resume
~~~

| Option | Description |
| --- | --- |
| <code>--rq</code> | One or more paper RQs: rq1, rq2, rq3, rq4, rq5. |
| <code>--data</code> | Extracted dataset location; defaults to data/. |
| <code>--output</code> | Output directory; choose a new directory for independent invocations. |
| <code>--resume</code> | Resume a full run with matching settings. |
| <code>--no-figures</code> | Omit plot generation. |
| <code>--no-verify</code> | Skip rehashing data already verified during extraction. |
| <code>--dry-run</code> | Print planned commands without executing analyses. |

A standalone full RQ3 run adds RQ1 to obtain its fitted 0.02 m folds. RQ1–RQ4 support resuming completed cells. Completed RQ5 reproduction is skipped; an interrupted RQ5 attempt must use a fresh output directory.

</details>

<details>
<summary><b>Run tests and additional audits</b></summary>

~~~bash
python -m unittest discover -s src_new/tests -v
python -m unittest discover -s replication/tests -v

# Complete-block bootstrap sensitivity:
python replication/bootstrap.py --data data --output reproduced/bootstrap-review

# Recover the 960 training-voxel counts without fitting SBFL:
python replication/voxel_diagnostics.py --data data --output reproduced/training_voxels.csv
~~~

Data-dependent integration tests are explicitly skipped before installing the archive. Validation with the data installed passed **72 tests**, table/statistical reproduction, full raw-baseline evaluation, and full RQ5 rescoring. See [the validation record](replication/validation.json) for the exact scope.

</details>

## Troubleshooting

| Issue | What to do |
| --- | --- |
| **Dataset not installed** | Run the dataset unpack command before analysis. |
| **Zenodo URL is unset** | Obtain the prepared ZIP from the authors until the published URL is added to archive.json. |
| **Checksum mismatch** | Check the expected SHA-256 in [archive metadata](replication/archive.json) and obtain an intact copy. Preserve the frozen files. |
| **Output directory is not empty** | Choose another output directory, or resume a full run with matching settings. |
| **Interrupted RQ5 scoring** | Select a fresh output directory; partial files are preserved. |
| **Python/dependency mismatch** | Use Python 3.11 and the pinned requirements in a dedicated environment. |
| **Insufficient disk space** | Account for uncompressed data, small-file overhead, and newly fitted artifacts. |
| **Historical paths in metadata** | These are provenance records; the launcher supplies the installed data paths explicitly. |

Original protocol documents and approximate statistical files remain available for provenance. The launcher follows the final paper's numbering and exact workflow. [Replication details](docs/REPLICATION.md) document tolerances and platform-dependent formatting.

ROC-AUC/AP assess trajectory-level failure discrimination; they do not establish that an individual voxel caused a failure.

## Citation

If you use this package, please cite the VoxelSBFL paper and its Zenodo dataset once their publication identifiers are available. Software citation metadata are provided in [CITATION.cff](CITATION.cff).

The following entry cites the replication software:

~~~bibtex
@misc{kanwal2026voxelsbfl_replication,
  title   = {VoxelSBFL: Paper Replication Package},
  author  = {Arusa Kanwal and Pablo Valle and Paola Roggero and
             Miren Illarramendi and Shaukat Ali and Aitor Arrieta},
  year    = {2026},
  url     = {https://github.com/ArusaKanwal1225/VLA_SBFL_Coverage},
  version = {2026.11.16}
}
~~~

## Acknowledgements

We acknowledge the developers of [EO-1](https://github.com/SHAILAB-IPEC/EO1), [Isaac GR00T](https://github.com/NVIDIA/Isaac-GR00T), [pi0](https://github.com/Physical-Intelligence/openpi), and [SpatialVLA](https://github.com/SpatialVLA/SpatialVLA), whose models were used to produce the recorded executions.

The baseline comparison builds on [Evaluating Uncertainty and Quality of Visual Language Action-enabled Robots](https://github.com/pablovalle/VLA_UQ). The offline analysis uses NumPy, pandas, SciPy, scikit-learn, and Matplotlib.

Additional documentation: [paper artifacts](docs/PAPER_ARTIFACTS.md) · [map provenance](experiments/RQ5/PROVENANCE.md) · [Zenodo release instructions](docs/ZENODO.md) · [cleanup record](docs/CLEANUP.md).
