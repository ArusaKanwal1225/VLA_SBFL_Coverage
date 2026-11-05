# RQ5: Pick-Up to Move Near transfer

This directory contains the complete RQ5 analysis, scene/execution scripts, and
the analysis data needed to reproduce the reported results. Start with the
verification commands below. Model checkpoints, a simulator and a GPU are not
required to verify the stored executions or reproduce their analysis.

## Map provenance

**RQ5 used reconstructed maps, not the original historical RQ1 map files.**
The 320 published maps were reconstructed from the then-current SBFL Pick-Up
inputs using the frozen RQ1 protocol: 34 formulas, four configurations per
budget, budgets 200/400 and ten leave-one-run-out folds per model.
All target trajectories were scored using those frozen reconstructed maps,
without training or calibration on Move Near.

The reconstruction metadata records differences from the historical RQ1
episode audit. These are 61 configuration-episode records across the four
models; overlapping budgets and the ten folds must not be interpreted as
independent extra executions. See [PROVENANCE.md](PROVENANCE.md) and
[data/map_provenance.json](data/map_provenance.json).

The exact reconstructed map artifacts used for the reported RQ5 results are
included. Their bytes and metadata hashes have been preserved. Reproducing
RQ5 with these files does not require the current Pick-Up inputs and must not
start by rebuilding maps from another copy of the Pick-Up data.

## Final sample and task definition

The full candidate dataset contains 504 completed Move Near episodes per
model (2,016 executions). The simulator task is
google_robot_move_near_customizable, with execution seed 2024.
An episode passes exactly when the final log entry has success=true.
No separate pick-and-place positional tolerance or stability-period criterion
was implemented by this analysis.

The scene generation seed was 20261028. The XY input design has 72 cells
(9 x 8) of width 0.05 m, with seven scenes per cell in the full candidate set.
These input-design cells are distinct from the frozen RQ1 scoring grid:
0.02 m 3-D voxels, origin (-1,-1,0), in robot_base coordinates.

The final analysis uses the retrospective fixed-pass selection, not the earlier
100-pass/100-fail sample or the exploratory 288/325/360-per-model alternatives:

| Model | Candidate pass/fail | Selected pass | Selected fail | Selected N | Selected scenes per cell |
| --- | --- | ---: | ---: | ---: | --- |
| EO1 | 177 / 327 | 100 | 123 | 223 | 3 or 4 |
| GR00T | 225 / 279 | 100 | 146 | 246 | 3 or 4 |
| Pi0 | 109 / 395 | 100 | 198 | 298 | 4 or 5 |
| SpatialVLA | 151 / 353 | 100 | 242 | 342 | 4 or 5 |

The sample uses exactly 100 passes and at least 100 failures per model, with
model-specific N chosen to approximate the pooled RQ1 pass ratio as closely as
integer counts and available outcomes permit. It is outcome-stratified and
retrospective. It does not preserve the RQ1 ratio with mathematical exactness.
Cell quotas differ by at most one within each model; selected scene IDs need
not be identical across models. All SBFL formulas and baselines within a
model use the same selected episodes.

## Install analysis dependencies

Use Python 3.11; the recorded run used Python 3.11.16. From the repository root:

~~~bash
python3.11 -m venv .venv-rq5
source .venv-rq5/bin/activate
python -m pip install -r experiments/RQ5/requirements-analysis.txt
~~~

On Windows, activate the virtual environment through its Scripts directory.
The requirements record the numerical package versions used for verification.

## Verify the published files and labels

~~~bash
python experiments/RQ5/rq5_reproduce.py verify
~~~

This verifies the SHA-256 and size of every published data/map/reference file,
checks all 504 episode labels against final simulator logs, checks scene/prompt
hashes, and validates the final sample's class counts and cell quotas.
Videos, images, raw action streams and execution timing files are not included.

## Reproduce the selection, both analyses and tables

~~~bash
python experiments/RQ5/rq5_reproduce.py run \
  --output /path/to/empty/rq5-reproduced
~~~

The output must be empty. The command regenerates the seeded selection,
evaluates the 34 techniques using the published reconstructed maps, computes
the frozen RQ2 baselines, generates both table types and compares fold,
configuration, model and baseline metrics against the published reference
with numeric tolerance 1e-12. It writes verification_report.json on success.

The main LaTeX outputs are:

* sbfl/roc_auc_table.tex: 34 SBFL techniques, ordered by Overall, with ranks.
* baselines/baseline_top5_sbfl.tex: available baselines plus the descriptive
  top-five SBFL ranking from RQ5; ties at the fifth rank are retained.
* baselines/baseline_performance.tex: available baselines alone.

LaTeX tables require booktabs. Overall weights each of the four models equally;
SBFL ROC-AUC is additionally averaged over the RQ1 folds and configurations.
Repeated scoring of the same target sample does not create independent target
replications. Average precision reflects the selected outcome proportions.
Token metrics and execution variability are empty in these logs; the baseline
availability rules exclude them, with no imputation or episode removal.

## Generate tables only

To regenerate the two table types from the included reference metrics,
without rescoring trajectories:

~~~bash
python experiments/RQ5/rq5_reproduce.py tables \
  --output /path/to/empty/rq5-tables
~~~

This is distinct from running the full reproduction and integrity verification.

## Data layout

See [data/README.md](data/README.md). The published analysis bundle includes:

* move_near_504/: episode CSVs, TCP trajectory CSVs, run configurations,
  final-outcome logs and the 13 baseline catalogue JSON files for every episode.
* reconstructed_pickup_maps/: the 320 frozen reconstructed map artifacts.
* scenes_504/: scene specifications, prompts, cell/pose metadata and generation report.
* reference/: the final selection and previously computed metrics and tables.
* files_manifest.csv: relative paths, byte sizes and SHA-256 fingerprints.
* map_provenance.json: reconstruction provenance and historical-audit differences.

No new raw Pick-Up inputs, model checkpoints, pilot execution data, videos or
alternative exploratory RQ5 samples are added by this verification bundle.
Reference analysis configuration files retain original server paths as
historical metadata; reproduction uses the relative bundle paths above.

## Low-level analysis commands

The existing entry points remain available. To use the exact published maps:

~~~bash
python experiments/RQ5/rq5_transfer.py evaluate \
  --maps experiments/RQ5/data/reconstructed_pickup_maps \
  --target experiments/RQ5/data/move_near_504 \
  --sample-manifest experiments/RQ5/data/reference/selection/sample_manifest.csv \
  --output /path/to/empty/transfer

python experiments/RQ5/rq5_baselines.py \
  --target experiments/RQ5/data/move_near_504 \
  --sample experiments/RQ5/data/reference/selection/sample_manifest.csv \
  --sbfl-results /path/to/empty/transfer \
  --output /path/to/empty/baselines
~~~

The evaluate command obtains the RQ1 configuration design from the frozen fold
audit, not from current Pick-Up input directories. prepare remains an optional
reconstruction command for researchers with the corresponding Pick-Up inputs.
It records their current fingerprints and audit discrepancies; a new
reconstruction is not assumed to be identical to the published artifacts.

## Scene generation and model execution

Generate_Pick_Place_100.py and run_move_near_100*.py/.sh cover the historical
independent pilot. rq5_pilot_table.py reports its original sample-size reasoning.
Those pilot executions were excluded from the 504-scene candidate dataset.

Generate_Pick_Place_Scenes.py generates the 504-scene candidate set.
The original scene files used for the published executions are included in
data/scenes_504, so regeneration is not needed for analysis verification.

To execute models again, use a separate Deephyperion_VLA checkout and the
model environments/checkpoints described by environment_mount/*.sh:

~~~bash
export DEEPHYPERION_ROOT=/path/to/Deephyperion_VLA
bash experiments/RQ5/run_move_near_504_models.sh --video pi0 gr00t eo1 spatialvla-4b
# Resume a failed run with matching seed, video and output settings:
bash experiments/RQ5/run_move_near_504_models.sh --resume --video gr00t
~~~

The runner installs/reuses one model environment at a time. Successful runs
remove that environment and its checkpoint directories; failures preserve them.
These simulator dependencies and cleanup steps are separate from the analysis
verification commands above. Execution results default to the Deephyperion
results/move_near_504 directory, not the historical pilot directory.
