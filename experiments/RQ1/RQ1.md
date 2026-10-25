# RQ1 protocol: comparison of spectrum-based techniques

**Status:** approved and frozen for Phase A execution on 2026-10-23. Phase B
remains pending because the current dataset has no causal voxel labels.

## Research question

> Which is the best spectrum-based technique for localizing voxels that can be
> problematic in the context of Vision-Language-Action models?

## Validity boundary

The current data contain trajectory coordinates and an episode-level pass/fail
label, but no label identifying the voxel that caused a failure. Consequently:

1. The existing data can measure how well voxel suspiciousness separates unseen
   failed and passed trajectories.
2. A direct claim about localization requires a second benchmark with a known or
   annotated problematic region.

The protocol therefore has two phases. Phase A is mandatory and uses the current
data. Phase B is mandatory before making a causal voxel-localization claim.

## Dataset and factors

The current benchmark contains 96,000 trajectories arranged as:

- VLA: `EO1`, `Groot`, `Pi0`, and `SpatialVLA`.
- Trajectory-generation condition: `DH`, `DH_HybridDT`, `DH_HybridRF`, and
  `DH_WeightedRF`.
- Budget: 200 or 400 scenes per run.
- Repetitions: 10 runs for every VLA/condition/budget combination.

This produces 32 experimental configurations and 320 held-out-run folds. The
200- and 400-scene datasets must first be checked for shared scenes. If they
overlap, budget is treated as a sensitivity analysis rather than as an
independent replication.

### Frozen pre-launch audit decision

The audit completed before formula evaluation found seven exact trajectory
matches between the 200- and 400-scene datasets. The schema has no stable scene
identifier, so additional semantic overlap cannot be excluded. In accordance
with the rule above, the primary inferential unit is therefore one of the 16
`VLA × trajectory-generation condition` blocks. The two budget results are
averaged within each block for the primary ranking, while budget-specific results
are reported separately as a sensitivity analysis. All 32 configurations and
320 folds are still evaluated.

## Techniques under comparison

Compare 34 techniques: the 31 formulas from Table 4 plus Tarantula, Ochiai, and
Ochiai2. The list, formulas, numerical edge-case policy, and implementation
version are frozen before any confirmatory result is examined. Every technique
receives exactly the same folds and preprocessing.

## Phase A: held-out trajectory discrimination

### Splitting strategy

Use deterministic leave-one-run-out cross-validation independently inside every
VLA/condition/budget configuration:

- Train on 9 complete runs.
- Test on the remaining run.
- Repeat until each of the 10 runs has been the test run once.
- Never split individual steps or trajectories from one run across train and test.
- Preserve the observed pass/fail prevalence; do not rebalance the test set.

Run identifiers are the evaluation blocks. Episode IDs and their hashes are
stored in every artifact so overlap can be rejected automatically.

### Frozen representation and scoring

- Coordinate frame: `robot_base`.
- Primary voxel size: 0.02 coordinate units, with origin `(-1, -1, 0)`.
- Coverage: binary per trajectory; revisiting a voxel does not increase coverage.
- Formula input: `ef`, `ep`, `nf`, and `np` computed only from the nine training runs.
- Undefined `0/0`: the formula's explicit neutral value, recorded in diagnostics.
- Mathematical infinities: finite rank-preserving sentinels, recorded in diagnostics.
- Voxel-score normalization: empirical percentile fitted on training voxels only.
- Primary trajectory aggregation: maximum percentile among known visited voxels.
- Unknown voxels: ignored when known voxels exist; an all-unknown trajectory gets
  the neutral score 0.5 and is reported as unsupported.

The aggregation rule is fixed for all 34 techniques. It is not tuned separately
for each technique.

### Outcomes

Primary outcome:

- ROC-AUC for failed-versus-passed trajectories in each held-out run.

Secondary outcomes:

- Average precision, reported together with failure prevalence.
- Mean known-voxel fraction and number of all-unknown trajectories.
- Number of distinct trajectory scores and tied-score rate.
- Formula undefined/infinite counts.
- Runtime and artifact size.

No classification threshold is selected.

## Phase B: direct voxel-localization benchmark

Create a held-out subset where the problematic spatial region is known without
using the SBFL result. Preferred sources, in order, are:

1. Controlled simulator interventions attached to a recorded voxel region.
2. Collision, contact, or task-failure telemetry providing a spatial position.
3. Independent annotation by at least two annotators, with agreement reported.

For each failed trajectory, convert the reference position or region using the
same voxel grid. Evaluation uses only the frozen suspiciousness map and reports:

- Voxel-level average precision as the primary localization metric.
- Voxel-level ROC-AUC as a threshold-free secondary metric.
- Hit rate in the top 1%, top 5%, and top 10% of ranked visited voxels.
- Reciprocal rank of the first reference voxel.
- Distance from the reference region to the highest-ranked voxel.

If Phase B cannot be constructed, the conclusion must say "failure-associated
voxel ranking" rather than "fault localization".

## Statistical analysis and selection rule

1. Compute every metric per technique and held-out run.
2. Average the 10 folds within each VLA/condition/budget configuration.
3. Macro-average the 32 configuration results so every configuration has equal
   weight; do not pool all trajectories into one micro-average.
4. Report results overall and stratified by VLA, trajectory condition, and budget.
5. Compare techniques as paired observations using configuration-level ranks.
6. Use a Friedman omnibus test, followed only if significant by paired post-hoc
   comparisons against the leading technique with Holm correction.
7. Report paired effect sizes and 95% hierarchical-bootstrap confidence intervals
   for ROC-AUC differences, resampling configurations and runs.

The result is an equivalence group rather than an arbitrary single winner when
techniques differ by less than 0.01 macro ROC-AUC and the corrected comparison is
not significant. The preferred technique among practically equivalent candidates
is the one with better voxel-level average precision in Phase B, then lower
runtime as a final tie-breaker.

## Robustness analyses

These analyses are reported separately and cannot replace the primary result:

- Voxel sizes of 0.01 and 0.05 versus the primary 0.02.
- Maximum aggregation versus the 95th percentile and mean of the top 5%.
- 200-scene versus 400-scene datasets.
- Per-VLA and per-trajectory-condition technique rankings.
- Leave-one-object-out evaluation using `object_name`.
- Training-size learning curves, using the existing preliminary experiment only
  as motivation and new confirmatory samples for the actual analysis.

Cross-VLA map transfer is a different question. It should be a separate
experiment because a spatial map learned for one policy/model is not necessarily
expected to transfer to another, even when the technique itself is robust.

## Pre-launch checklist

- Audit missing coordinates, duplicate steps, empty trajectories, class balance,
  and coordinate ranges for all 32 configurations.
- Verify whether the 200- and 400-scene datasets overlap.
- Verify that every train and test fold contains both outcome classes.
- Freeze the 34-formula catalogue, voxelization, aggregation, and edge-case policy.
- Generate and hash all fold manifests before computing any technique metrics.
- Specify the Phase B ground-truth source and its sample-size/power analysis.
- Freeze the analysis script and expected output schema.
- Record software versions and random seeds.

Only after every checklist item is satisfied should the confirmatory RQ1 run be
launched.
