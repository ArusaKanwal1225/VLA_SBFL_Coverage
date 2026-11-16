# VoxelSBFL — paper replication package

CPU-only replication of **VoxelSBFL: Voxel-Level Spectrum-Based Failure
Prediction for Vision-Language-Action Models**, aligned with the final manuscript
TOSEM_Arusa (5).pdf and GitHub scientific source commit **44b2113b**.

**No GPU, simulator, model checkpoints or videos are required.** Code is kept in
GitHub; analysis data are distributed as a separate ZIP for Zenodo.
[Archive metadata](replication/archive.json) records the exact size and SHA-256.
The Zenodo URL/DOI remain unset until the authors publish the upload.

## Quick start

Clone the published code revision without historical dataset blobs:

~~~bash
git clone --depth 1 https://github.com/ArusaKanwal1225/VLA_SBFL_Coverage.git
cd VLA_SBFL_Coverage
~~~

Use **Python 3.11**, from the repository root:

~~~bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python replication/data.py unpack /path/to/vla-sbfl-data-20261116.zip
python replicate.py tables
~~~

On Windows, create/activate the environment with:

~~~powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
~~~

Use python for subsequent commands. Unpacking checks the archive checksum and
each payload's SHA-256, rejects unsafe paths and refuses an existing nonempty
data directory.

The last command regenerates **paper Tables 2–15** in
**reproduced/tables/paper/**, together with **Figure 5** and supporting CSVs.
It uses the released exact statistical reviews and original pointwise confidence
intervals. Supplementary tables/learning-curve figures are also generated.
Table 1 and Figures 1–4 are worked examples, illustrations or simulator images,
rather than analysis outputs.

## Five RQs: paper numbering and original code paths

| Paper RQ | Evaluation | Code directory |
| --- | --- | --- |
| RQ1 | 34 SBFL techniques | experiments/RQ1 |
| RQ2 | Baselines plus the top-five RQ1 techniques | experiments/RQ2_baselines |
| RQ3 | Voxel sizes 0.01 / 0.02 / 0.05 | experiments/RQ2 |
| RQ4 | Training-size / coverage learning curves | experiments/RQ3_v2 |
| RQ5 | Pick-Up → Move-Near transfer | experiments/RQ5 |

Original directory names are retained for compatibility. The launcher uses the
paper's numbering. The older controlled-coverage code under experiments/RQ3
supplies reusable helpers/tests; it is not a sixth paper experiment.

## What to execute

~~~bash
# Verify all files and RQ5 final outcomes, scene hashes, maps and selection:
python replicate.py verify

# Recompute exact tests from the frozen aggregates and regenerate paper tables:
python replicate.py stats

# A short pipeline check across all five RQs:
python replicate.py smoke

# Refit/re-evaluate from recorded executions and regenerate all paper results:
python replicate.py run
~~~

Each mode writes to its own directory under reproduced/. Choose a new output
directory for independent repetitions:

~~~bash
python replicate.py tables --output reproduced/reviewer-tables
python replicate.py stats --rq rq1 rq2 rq3 rq4 --output reproduced/reviewer-stats
python replicate.py run --rq rq5 --output reproduced/reviewer-rq5
python replicate.py run --resume
python replicate.py run --dry-run
~~~

Tables and stats normally take minutes. A full run refits 320 RQ1 folds,
640 additional voxel-size folds, the baseline catalogue and 16,000 learning-curve
cells, then rescores RQ5. Allow additional time/disk space; timings depend on
hardware. The short test does not establish full-design significance.

The statistical mode does not refit maps or regenerate bootstrap intervals.
It recomputes the exact tests from block-level observations and checks them
against the released references. Existing historical approximate outputs remain
unchanged in the immutable data directory.

## Data

| Component | Contents |
| --- | --- |
| data/pickup | 96,000 Pick-Up episodes, 7,680,000 ordered TCP positions, original baseline JSONs and outcome evidence |
| data/rq5 | 504 Move-Near candidates per model, exact frozen reconstructed maps, scenes and final sample manifests |
| data/reference/rq1–rq4 | Historical audits, aggregates, exact reviews, publication artifacts and diagnostics |
| data/files_manifest.csv | Every payload's relative path, original bytes and SHA-256 |

Four models are evaluated: EO1, GR00T, Pi0 and SpatialVLA.
RQ5 fixes 100 passes per model and selects 123/146/198/242 failures respectively:
**N = 223/246/298/342**, totaling **1,109 selected executions**.
Sampling approximates the RQ1 proportions with integer counts and balanced
spatial-cell quotas; it does not preserve percentages with mathematical exactness.

**RQ5 uses reconstructed Pick-Up maps, not the exact original historical RQ1
map files.** These exact released reconstructed artifacts are used without
target-task retraining/calibration. See [map provenance](experiments/RQ5/PROVENANCE.md).

The data occupy much more disk space after extraction than their ZIP size.
See archive.json for exact bytes/counts and allow for filesystem overhead and
newly fitted artifacts. No raw videos, checkpoints or duplicate action/TCP streams
are included.

## Scientific and reporting safeguards

* Same held-out runs and scoring rules for all techniques.
* Failure is the positive class; RQ5 labels are checked against final simulator logs.
* Exact Wilcoxon tests and the complete predeclared Holm families are preserved.
* Paper tables use exact-review p-values; historical pointwise bootstrap CIs are retained.
* Frozen maps, input provenance and historical discrepancies are checked explicitly.
* Data and reference files are never overwritten by analysis commands.

Outputs evaluate trajectory-level failure discrimination, not proof of a voxel's
causal responsibility for a failure. Original protocol documents are immutable
provenance artifacts; the final exact statistical workflow is described here and
in [replication details](docs/REPLICATION.md).

## Tests and release instructions

~~~bash
python -m unittest discover -s src_new/tests -v
python -m unittest discover -s replication/tests -v
~~~

Data-dependent integration tests are explicitly skipped before installing the
Zenodo archive. After installation, all tests can run.

See [validation](replication/validation.json), [cleanup record](docs/CLEANUP.md),
[paper artifact mapping](docs/PAPER_ARTIFACTS.md) and
[Zenodo publication instructions](docs/ZENODO.md).
No remote push or rewrite of published history was performed.
