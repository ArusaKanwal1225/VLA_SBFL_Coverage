# RQ2 protocol: comparison with trajectory-level baselines

**Status:** implementation ready; EO1 is currently available. The confirmatory
cross-VLA analysis remains pending until the raw `allMetrics` data for all four
VLAs have been collected.

## Research question

> How does VoxelSBFL compare with state-of-the-art trajectory-level uncertainty
> and quality metrics for discriminating failed from successful VLA executions?

## Scope and interpretation

This RQ compares failure-discrimination performance, not localization power. The
external metrics assign one score to an execution but do not rank spatial voxels.
Consequently, an external baseline may obtain a higher ROC-AUC without replacing
VoxelSBFL's diagnostic output. The results must describe such a method as a better
trajectory-level failure discriminator, not as a better faulty-voxel localizer.

`Optimal Trajectory` depends on object/goal state and is identified as task-aware
or oracle-informed. Token metrics require model internals, and Execution
Variability requires repeated inference. These observability differences are
reported and cannot be hidden by a single overall ranking.

## Dataset and experimental cells

Use exactly the RQ1 pick-up executions:

- VLAs: EO1, GR00T, Pi0, and SpatialVLA.
- Generators: `DH`, `DH_HybridDT`, `DH_HybridRF`, and `DH_WeightedRF`.
- Budgets: 200 and 400 trajectories per run.
- Repetitions: ten runs per VLA/generator/budget configuration.

Each raw episode is joined to the compact RQ1 input by VLA, configuration, run,
and `eval_id`. Its final `failure` and `object_name` fields must agree exactly.
The final outcome in `dh_evaluations.csv`, rather than any transient success in
the step log, is the ground-truth label.

## Methods

The frozen VoxelSBFL reference is Harmonic Mean at voxel size 0.02, using the
completed RQ1 held-out predictions. Rogot2 is omitted from the main comparison
because RQ1 established that it produces identical predictions.

The external catalogue contains the 13 metrics from the baseline paper:

1. Token Probability, PCS, DeepGini, and Entropy.
2. Action Position, Velocity, and Acceleration Instability.
3. Execution Variability.
4. TCP Position, Velocity, and Acceleration Instability.
5. Trajectory Instability and Optimal Trajectory.

The stored per-step and per-dimension values are reduced to one trajectory score
using their arithmetic mean. The score direction is frozen before evaluation:
Token Probability and PCS are negated because lower confidence represents greater
failure risk; every uncertainty, instability, deviation, and distance quantity
uses its stored increasing direction. Directions must never be selected by
examining ROC-AUC.

## Missing-metric policy

A metric is evaluated in a held-out run only when it is available for every
episode in that run. Episodes are never dropped to make a metric complete. The
primary cross-VLA comparison contains only metrics available in every selected
experimental block. Model-specific metrics may be reported separately with their
availability, but their partial macro-average cannot be compared with a complete
cross-VLA average.

## Outcomes and aggregation

Failed trajectories are the positive class. The primary outcome is ROC-AUC and
the secondary outcome is average precision. No threshold is selected.

Aggregation follows RQ1 exactly:

1. Compute ROC-AUC and average precision separately for each test run.
2. Average the ten runs within each VLA/generator/budget configuration.
3. Average the two budgets within each VLA/generator primary block.
4. Macro-average the primary blocks overall and separately by VLA.

Although an external baseline is not fitted, each raw run is treated as the same
test cell as its corresponding leave-one-run-out RQ1 fold. This creates paired
observations without granting either method a different test distribution.

## Statistical analysis

The omnibus comparison uses the complete methods and the 16 matched
VLA--generator primary blocks. If appropriate, each external method is compared
with Harmonic Mean using paired Wilcoxon signed-rank tests and one Holm correction.
The analysis reports candidate-minus-reference ROC-AUC gaps, paired rank-biserial
effect sizes, and 95% hierarchical-bootstrap confidence intervals. The bootstrap
resamples primary blocks and folds while keeping both budgets equally weighted.
Differences below 0.01 ROC-AUC are practically negligible only when the corrected
comparison does not reject equality.

EO1-only results are preliminary because they contain only four primary blocks
and cannot establish cross-model generality.
