# RQ5: pick-and-place pilot

This directory contains the scripts used to generate and run the 100-scene
pilot and calculate the target sample size for RQ5. It does **not** implement
the final SBFL transfer experiment or train or select suspiciousness maps.

## What has been done

* `Generate_Pick_Place_100.py` generated 100 fixed move-near scene
  specifications and their prompts with seed `20261026`. Source and target
  object/orientation schedules are balanced; positions are sampled within the
  workspace subject to a minimum separation of 0.20 m.
* The same task and prompt files were run with PI0, GR00T, EO1 and SpatialVLA
  in `google_robot_move_near_customizable`, using simulation seed 2024. The
  runner records TCP poses, per-step logs, metrics and optional videos.
* The `success` value in the final log step determines the episode outcome:
  true is a pass and false is a fail. Older logs may contain the strings
  `"true"` and `"false"`; the analysis supports both strings and JSON booleans.
* All four models have 100 completed pilot logs at the original results path
  when this README was written. Their pass/fail counts were PI0 21/79, GR00T
  41/59, EO1 28/72, and SpatialVLA 33/67. Always regenerate the table from the
  actual logs rather than relying on these counts.
* The minimum pilot minority proportion was 0.21. Applying the stated formula
  gives `N = K * ceil(500 / K)` for every model. K has not been established by
  the scene generator; it must be specified for a numeric final N.

The pilot uses move-near success from the simulator. It has not established a
separate placement tolerance or stability period, nor a balance across discrete
destination regions. Do not claim those properties without implementing and
verifying them. The pilot scenes must be excluded from the final dataset.

## Files

| File | Purpose |
| --- | --- |
| `Generate_Pick_Place_100.py` | Generate the scene JSON, prompts and design figure. |
| `run_move_near_100.py` | Run one model over the 100 scenes and export episode CSVs. |
| `run_fuzzer_allMetrics_integrated.py` | `SimRunner`, used by the pilot runner. |
| `model_interface_allMetrics.py` | Model and simulator interface used by `SimRunner`. |
| `DH_Hybrid_DT.py` | DeepHyperion reference functions used to calculate trajectory coverage and CSV features. Its search is not run. |
| `run_move_near_100_models.sh` | Install/reuse, run and clean up model environments sequentially. |
| `environment_mount/*.sh` | Model-specific environment installers used by the orchestration script. |
| `rq5_pilot_table.py` | Read the pilot logs and generate the LaTeX sample-size table. |

These are RQ5-local copies of the execution scripts and installers used in
`Deephyperion_VLA`. Future fixes to the originals are not automatically shared.

## Dependencies and paths

The simulator and models still require a separate Deephyperion checkout, its
submodules, model-specific Conda environments, CUDA, checkpoints, and Xvfb.
The default checkout is `/home/ubuntu/Deephyperion_VLA`. To use another path:

```bash
export DEEPHYPERION_ROOT=/path/to/Deephyperion_VLA
```

The generation and table scripts require Python 3; the generator also requires
Matplotlib. Execution uses the dependencies installed by the model setup
scripts, including PyTorch, SimplerEnv, ManiSkill and the relevant model code.
Downloads require network access and sufficient disk space. The scripts assume
a Linux environment with Conda and `xvfb-run`; the setup scripts may require
permissions to install system packages. Check the installer commands before use.

Default inputs are `<VLA_SBFL_Coverage>/output/Pick_and_place_100/` and default
execution outputs are `<DEEPHYPERION_ROOT>/results/move_near_100/`. Use
`--output` to store results elsewhere. The table script defaults to the latter
path; pass `--input` and `--output` explicitly when relocating results. Results,
checkpoints and Conda environments are not part of this directory.

## Reproduce the pilot

Run commands from the root of `VLA_SBFL_Coverage`. The generation command
rewrites the JSON and figure in its output directory:

```bash
python3 experiments/RQ5/Generate_Pick_Place_100.py \
  --seed 20261026 --output-dir output/Pick_and_place_100
```

Run selected models sequentially; omit the model names to run all four:

```bash
bash experiments/RQ5/run_move_near_100_models.sh --video pi0 gr00t eo1 spatialvla-4b
```

Each model is installed or reused, run for 100 episodes, and then its named
Conda environment and checkpoint directories are **deleted on success**,
including when they existed before the run. Other checkpoint files such as
`.py` files are preserved. If execution fails or is interrupted, the script
keeps the environment and checkpoints; resume using the same seed, output and
video options:

```bash
bash experiments/RQ5/run_move_near_100_models.sh --resume --video pi0
```

If SpatialVLA was interrupted during installation and its checkpoint is
missing, the orchestration script resumes its installer. Flash-attention is
compiled without build isolation using Ninja. Set `MAX_JOBS` to control build
parallelism, for example:

```bash
MAX_JOBS=16 bash experiments/RQ5/run_move_near_100_models.sh --resume --video spatialvla-4b
```

For a preconfigured environment, the runner can also be invoked directly:

```bash
MUJOCO_GL=glx xvfb-run -s '-screen 0 1280x720x24' -a \
  conda run --no-capture-output -n PI0 python3.10 \
  experiments/RQ5/run_move_near_100.py --model pi0 --resume --video
```

Available model values are `pi0`, `gr00t`, `eo1`, and `spatialvla-4b`. The
runner loads one model per invocation and shows a progress bar with an ETA.
With `--resume` it requires the same input hashes, model, seed and video setting
as the existing `run_config.json`; it skips completed logs and rebuilds CSVs.
Each model output includes `allMetrics/<episode>/`, `episodes.csv`,
`dh_evaluations.csv`, and `run_config.json`. CSV fields specific to a
DeepHyperion search are empty because these are fixed scenes, not a search.

## Generate the LaTeX table

Specify the actual number of destination regions K when known. The output path
can be outside both repositories:

```bash
python3 experiments/RQ5/rq5_pilot_table.py \
  --input /path/to/move_near_100 --regions 5 --output /path/to/rq5_pilot_table.tex
```

Without `--regions`, the selected N is left as a formula in K. The table has
one row per model, including pass/fail counts, minority proportion, the raw
minimum `ceil(100/q_m)`, and the final common N after the 500-scene floor and
rounding to K. It validates the presence of exactly IDs 0--99 and matching task,
prompt and seed signatures. A zero minority proportion yields no finite N.

## Remaining RQ5 work

1. Define reachable destination regions and K, and the final task's success
   tolerance and stability criterion. Verify balance, reachability and task
   comparability for all four models.
2. Decide N using the pilot table, then generate a separate final dataset with
   an equal region allocation and no overlap with the pilot. Expected counts of
   100 for each outcome are not a guarantee of observed counts.
3. Execute the final dataset, record trajectories and outcomes, and apply the
   34 RQ1-trained SBFL maps with the frozen voxelization and scoring protocol.
   Do not use final target trajectories to select or tune techniques.
4. Analyse the transfer results separately. The LaTeX table generated here is
   only a sample-size table; the draft suggests omitting it if N is common.



## Generate the 504-scene candidate final set

`Generate_Pick_Place_Scenes.py` generates 504 scene specifications using a
fixed seed (default `20261028`) and rejects exact pilot scene overlaps. It uses
72 cells in the input XY range, 7 scenes per cell, each object once as source
and once as target in each cell, and each distinct ordered object pair 12 times.
Each object/orientation combination occurs 14 or 15 times per role. These are
input-design guarantees, not evidence of physical reachability or collision-free
meshes. Position separation is at least 0.20 m by default.

```bash
python3 experiments/RQ5/Generate_Pick_Place_Scenes.py
```

Defaults to `output/Pick_and_place_504/` and writes the 504-scene JSON, prompt
list, per-scene cell/pose metadata and a generation report. To create a new
set, specify a different empty `--output-dir` and record the `--seed`. Existing
output files are not overwritten. The pilot JSON is required for overlap checks;
use `--pilot` if it has moved. The generator does not run the simulator.

## Run the 504-scene set

The separate `run_move_near_504.py` and `run_move_near_504_models.sh` use the
504-scene inputs by default. Their default output is
`<DEEPHYPERION_ROOT>/results/move_near_504/`, separate from the pilot's
`move_near_100/`. The runner refuses the default pilot output directory. Do not
point `--output` at any other location containing pilot results.

```bash
bash experiments/RQ5/run_move_near_504_models.sh --video pi0 gr00t eo1 spatialvla-4b
# After a failure, with the same video and seed options:
bash experiments/RQ5/run_move_near_504_models.sh --resume --video pi0
```

As with the pilot, successful completion deletes the named model environment
and checkpoint directories; a failed run preserves them. Each model's output
contains `episodes.csv`, `dh_evaluations.csv`, `trajectories.csv`, the individual
`allMetrics` folders and `run_config.json`. The trajectory CSV contains
`episode_id,step,x,y,z` and joins to `episodes.csv` by episode ID.

The runner uses the current simulator move-near final `success` outcome. Define
and freeze the intended final success criterion, and validate the scenes in
the simulator, before treating these data as the confirmatory RQ5 dataset.



## RQ5 transfer evaluation

`rq5_transfer.py` applies the 34 RQ1 SBFL formulas to move-near trajectories.
It uses the frozen RQ1 grid (0.02 m, origin `(-1,-1,0)`, `robot_base`), binary
voxel visits, training-voxel percentile normalization, maximum known-voxel
aggregation and neutral 0.5 for all-unknown trajectories. A seeded sample of
100 pass and 100 fail is selected **once per model**, without replacement, and
is shared by every formula and RQ1 fold. Full 504-episode runs and log/CSV label
agreement are required before sampling; insufficient classes stop the analysis.

The transfer analysis preserves the RQ1 hierarchy: per VLA it evaluates all
four training conditions, both budgets, and 10 leave-one-run-out pickup models.
It averages AUCs over folds and then configurations; the reused target sample
means these folds are not independent target replications. Balanced sampling
also means that average precision is measured at 50% failure prevalence and is
not the natural-prevalence AP. This evaluates failure discrimination, not
causal fault localization. Verify that target TCPs really use the same physical
coordinate frame as RQ1 before interpreting the results.

The original RQ1 model artifacts are absent from the current checkout. If they
are available elsewhere, pass their parent directory containing `folds/` as
`--maps`. Otherwise `prepare` reconstructs the same pickup-only training folds
and 34 scores; it never trains on target data. It verifies the frozen RQ1 code signature, records discrepancies with the
historical episode audit, and treats the current SBFL pick-up input as
authoritative. A SHA-256 fingerprint of the current input files is stored in
each map; existing artifacts are reused only if their provenance and current
input fingerprint match.

```bash
conda run -n rq2-baselines python experiments/RQ5/rq5_transfer.py prepare \
  --maps /path/to/rq1_pickup_maps

conda run -n rq2-baselines python experiments/RQ5/rq5_transfer.py evaluate \
  --maps /path/to/rq1_pickup_maps \
  --target /home/ubuntu/Deephyperion_VLA/results/move_near_504 \
  --output /path/to/new/rq5_transfer_results --seed 20261028
```

Use `--models pi0` on both commands to process a subset; default is all four.
`evaluate` refuses a nonempty output directory. It writes the sampled IDs,
configuration and input counts, individual predictions and fold metrics,
configuration/model summaries, and CSV/LaTeX ROC-AUC tables. Map metadata
hashes are included in fold metrics. Use a new output directory for a new
analysis; results are not written into the input datasets.

**Provenance:** Some current pick-up trajectories differ from the historical
RQ1 audit. By user decision, preparation uses the current SBFL inputs and
records discrepancy counts; the resulting maps should not be described as
bit-for-bit identical to the unavailable historical maps. All requested models
need 504 completed target episodes and at least 100 of each outcome.

### RQ5 ROC-AUC table formatting

The evaluation writes `roc_auc_table.csv` and `roc_auc_table.tex` in the same
format as RQ1: rows are ordered by the equal-model-weight `Overall` ROC-AUC;
each column shows its ROC-AUC and tie-aware competition rank. Rank 1 is bold;
ranks 2--5 are underlined. For partial runs, Overall averages only the included
models and is explicitly identified as partial in the caption. The LaTeX table
uses the `booktabs` package.

Regenerate just the table from an existing evaluation without rescoring:

```bash
conda run -n rq2-baselines python experiments/RQ5/generate_transfer_table.py \
  --input /path/to/rq5_transfer_results/model_metrics.csv
```


### RQ2 baselines on the RQ5 sample

`rq5_baselines.py` computes the 13 frozen RQ2 baseline definitions, without
Harmonic Mean, using the exact `sample_manifest.csv` of an RQ5 transfer run.
It checks episode IDs and final log labels, averages the numeric JSON values
per episode and applies the fixed RQ2 score directions. A metric enters the
performance table only if it is available for every sampled episode of every
selected model. The full availability and per-episode scores are saved as CSVs.

```bash
conda run -n rq2-baselines python experiments/RQ5/rq5_baselines.py \
  --sample /home/ubuntu/Deephyperion_VLA/results/rq5_transfer_final/sample_manifest.csv \
  --target /home/ubuntu/Deephyperion_VLA/results/move_near_504 \
  --output /path/to/new/rq5_baselines_results
```

The RQ2-style performance table uses a model-macro ROC-AUC and a rank, with
best values bold and second-best underlined. The task-aware Optimal Trajectory
is marked with a dagger because it uses object/goal state. A partial sample
produces a partial overall. Token metrics and Execution Variability are empty
in the current move-near logs; they cannot be calculated from those stored
values and are excluded rather than imputed. This table reports trajectory-level
failure discrimination, not localization. The `booktabs` LaTeX package is needed.

### RQ1 outcome distribution and RQ5 N sensitivity

`rq5_sample_size_comparison.py` compares the 200/400-episode RQ1 held-out run
outcomes against the independent RQ5 pilot, then writes CSV and LaTeX tables,
scenario calculations and suggested paper text. It never reads the final RQ5
outcomes for sample-size selection.

```bash
python3 experiments/RQ5/rq5_sample_size_comparison.py --output /path/to/new/analysis
```

The defaults reproduce K=72, the current target of 100 expected minority cases
and minimum N=500, plus an illustrative alternative target of 90 and floor of
400. Configure the latter with `--smaller-minority-target` and
`--smaller-floor`; it is a sensitivity scenario, not a prespecified result or
a power calculation. Outcomes and input file hashes are documented in the
output. Avoid interpreting a post-hoc subset as an independent evaluation.

### Retrospective RQ1-ratio sample

`rq5_select_rq1_ratio.py` selects a separate sample of 288 episodes per VLA
matching each model's pooled RQ1 200+400 pass/fail ratio, rounded to integer
counts, and selecting four scenes per destination cell. The conditional
sampling is seeded and uniform across subsets satisfying the fixed class and
cell constraints. This is a retrospective outcome-stratified sample; it does
not satisfy the previously stated 100-per-class target for every model, and its
per-model samples are not necessarily paired by scene. The RQ1 200 and 400
budgets can overlap; their pooled ratio gives twice the weight to the 400 budget.

```bash
python3 experiments/RQ5/rq5_select_rq1_ratio.py --output /path/to/new/selection
conda run -n rq2-baselines python experiments/RQ5/rq5_transfer.py evaluate \
  --maps /home/ubuntu/Deephyperion_VLA/results/rq1_pickup_maps \
  --target /home/ubuntu/Deephyperion_VLA/results/move_near_504 \
  --sample-manifest /path/to/new/selection/sample_manifest.csv \
  --output /path/to/new/transfer_results
conda run -n rq2-baselines python experiments/RQ5/rq5_baselines.py \
  --sample /path/to/new/selection/sample_manifest.csv \
  --output /path/to/new/baseline_results
```

When `--sample-manifest` is supplied, the transfer evaluator validates and uses
the provided IDs and labels instead of selecting 100 per class. Baseline
availability rules are unchanged. With unequal class proportions, average
precision reflects the selected distribution, not the original 504 executions.

For a retrospective RQ1-ratio sample with at least 100 pass and 100 fail per
model, add `--min-per-class 100`. With four scenes per cell, this produces 288
per model; the class minimum forces Pi0 and SpatialVLA away from the exact RQ1
ratios. Alternatively, add `--total 325 --min-per-class 100` to obtain 325 per
model and a closer ratio, with four or five scenes per cell rather than exact
uniform allocation. Cell quotas may differ between models. These samples were
chosen after observing outcomes and should be reported as retrospective.

A further retrospective sample can be generated with `--per-cell 5
--min-per-class 100`: it contains 360 episodes per model, exactly five per
cell, and uses the nearest RQ1 class ratio subject to both the class minimum
and the available class counts. In the current data this uses all 109 Pi0
passes, giving 30.28% passes versus 33.59% in pooled RQ1. The generated
`sample_manifest.csv` can be passed to both RQ5 transfer and baseline commands
above. These per-model samples need not have the same scene IDs.

To fix exactly 100 passes per model and select the number of failures yielding
the closest attainable pooled RQ1 pass ratio, use `--fixed-pass 100
--min-per-class 100`. N may differ per VLA; each region receives either floor(N/72)
or ceil(N/72) scenes. The selection is retrospective and only approximates
the exact RQ1 proportions because episodes are integers.

```bash
python3 experiments/RQ5/rq5_select_rq1_ratio.py --fixed-pass 100 --min-per-class 100 \
  --output /path/to/new/fixed100_selection
```

The transfer and baseline evaluators accept this same manifest and support
different sample sizes by VLA. Within a VLA all methods use the same episodes.

### Combine baselines and the top-five SBFL techniques

`generate_baseline_sbfl_table.py` adds all SBFL techniques ranked 1--5 by
Overall to the complete baseline table, marking SBFL methods with a double dagger and
recomputing a joint ranking. Ties at the fifth rank are retained. It verifies
that both analyses use the same sample-manifest hash before combining them.
Both Overall values are equally weighted across models; baseline values are
trajectory-level, whereas SBFL values are additionally averaged across the RQ1
training folds and configurations. The table uses `booktabs`.

```bash
conda run -n rq2-baselines python experiments/RQ5/generate_baseline_sbfl_table.py \
  --baseline-dir /path/to/baseline_results \
  --sbfl-dir /path/to/transfer_results \
  --output-dir /path/to/output_tables
```

Alternatively, pass `--sbfl-results /path/to/transfer_results` to
`rq5_baselines.py` to generate the combined table after the baseline analysis.
