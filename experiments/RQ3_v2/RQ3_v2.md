# RQ3 v2 protocol: test-suite size and voxel-coverage sensitivity

**Status:** implemented; full execution and publication-artifact generation complete.

## Research question

> How does the number of labelled trajectories used to construct the spectra,
> and the resulting voxel coverage, affect the held-out failure-discrimination
> performance and relative ranking of SBFL techniques for VLA models?

The number of spectrum-construction trajectories is the primary intervention.
Voxel coverage is a measured mechanism rather than a proxy for test-suite size.
The untouched evaluation run is never enlarged or used to construct a model.

The experiment has three parts:

- **RQ3.1 -- Learning curves:** how do coverage and ROC-AUC change as the
  spectrum-construction suite grows?
- **RQ3.2 -- Sample efficiency:** which SBFL techniques perform best across the
  complete learning curve rather than only at the largest suite size?
- **RQ3.3 -- Conditional coverage:** among suites containing the same number of
  trajectories, is greater achieved coverage associated with greater ROC-AUC?

This remains a trajectory-level discrimination experiment. Pass/fail labels do
not identify the physical voxel that caused a failure, so causal faulty-voxel
localization cannot be claimed without additional spatial ground truth.

## Hypotheses

- **H3.1:** More spectrum-construction trajectories increase calibration and
  held-out voxel coverage, with diminishing returns.
- **H3.2:** Held-out ROC-AUC generally improves with the number of trajectories,
  although individual techniques need not be monotonic.
- **H3.3:** The techniques differ in sample efficiency and learning-curve shape.
- **H3.4:** At a fixed trajectory count, suites with greater calibration
  coverage have greater held-out ROC-AUC.

## Benchmark factors

The frozen RQ1/RQ2 benchmark is retained:

- VLA: `EO1`, `Groot`, `Pi0`, and `SpatialVLA`;
- trajectory-generation condition: `DH`, `DH_HybridDT`, `DH_HybridRF`, and
  `DH_WeightedRF`;
- generation budget: 200 and 400 scenes;
- runs: ten for every VLA/condition/budget configuration;
- techniques: the 31 Table 4 formulas frozen in RQ1;
- primary voxel size: `0.02`, origin `(-1, -1, 0)`, frame `robot_base`.

The primary inferential unit is the `VLA x condition` block. The two generation
budgets receive equal weight inside each block.

## Leakage-safe split

For each outer fold:

1. one complete run is the untouched evaluation set;
2. the next run in cyclic order is an outcome-blind calibration set;
3. the remaining eight runs form the spectrum-construction pool.

Calibration coordinates measure coverage without accessing evaluation data.
Calibration labels are used only for post-hoc diagnostics. Evaluation
coordinates and labels are never used to draw or select a training suite.

## Repeated nested trajectory suites

The primary suite sizes are `10`, `20`, `50`, `100`, and `200`. For every
configuration, outer fold, and repetition, trajectories are randomly ordered
separately within the passing and failing classes. A suite of size `n` contains
the first `n/2` trajectories from each ordering. Consequently:

- every suite is exactly 50% passing and 50% failing;
- the five suites are nested within a repetition;
- a change between adjacent sizes is a paired addition of trajectories;
- independent deterministic repetitions quantify subset-sampling variability.

The default design uses ten repetitions. Repetitions are not treated as
independent benchmark subjects: fold-level results are first averaged within
configuration, and budgets are then averaged equally inside primary blocks.

## Model fitting and trajectory scoring

Each nested suite is voxelized and used to fit all 31 formulas simultaneously.
The frozen RQ1 policy is retained:

- spectra contain binary trajectory-level voxel visits;
- suspiciousness uses training-only `ef`, `ep`, `nf`, and `np` counts;
- voxel scores are transformed to empirical training percentiles;
- trajectory risk is the maximum percentile among known voxels;
- unknown voxels are ignored and recorded;
- an all-unknown trajectory receives the neutral score `0.5`;
- undefined and infinite formula results follow the finite RQ1 policy.

## Outcomes

The primary effectiveness outcome is held-out trajectory ROC-AUC with failed
trajectories as the positive class. Average precision is secondary.

Coverage diagnostics include:

- unique calibration-voxel coverage;
- mean per-trajectory known-voxel fraction on calibration data;
- unique held-out evaluation-voxel coverage;
- mean evaluation known-voxel fraction, overall and by outcome;
- occupied training voxels and all-unknown evaluation trajectories.

Evaluation coverage is diagnostic only. Calibration coverage is used for the
conditional coverage analysis so that evaluation data cannot influence the
model or experimental grouping.

## Learning-curve and sample-efficiency measures

For every technique, the report includes:

- ROC-AUC at every suite size, globally and by VLA;
- adjacent-size ROC-AUC changes;
- coverage at every suite size;
- normalized area under the ROC-AUC learning curve (NAULC), integrating over
  `log2(n)` so that doubling intervals receive comparable weight;
- ROC-AUC at 200 trajectories and the 10-to-200 gain;
- the smallest suite reaching 95% of the observed improvement above chance.

NAULC is the primary sample-efficiency summary. The 95% size is descriptive
because empirical learning curves need not be monotonic.

## Statistical analysis

For the complete primary design:

1. Harmonic Mean's four adjacent-size effects are tested over the 16 matched
   primary blocks using paired Wilcoxon signed-rank tests and one Holm
   correction.
2. Every technique's block-level NAULC is compared with Harmonic Mean using
   paired Wilcoxon tests and one Holm correction across the 30 comparisons.
   An absolute difference of `0.01` is the practical-effect threshold.
3. Technique learning-curve gains are compared descriptively and retained per
   VLA; inferential conclusions use matched primary blocks.
4. For RQ3.3, calibration coverage and ROC-AUC are centered within every
   configuration, fold, and suite-size stratum. A block-level fixed-effect
   slope is then estimated from repetition-to-repetition variation. This
   removes the between-size association before testing conditional coverage.

The across-size association between coverage and ROC-AUC must not be described
as a coverage-only causal effect: adding trajectories also improves the
estimates for voxels that were already known.

## Publication tables and visualizations

- one combined main-paper figure showing coverage and ROC-AUC against suite
  size, faceted by VLA;
- a VLA-stratified coverage/performance table, a VLA-stratified sample-
  efficiency table, and one matched adjacent-size statistical table;
- complete supplementary ROC-AUC and NAULC tables for all 31 techniques and
  each VLA;
- a complete technique-rank heatmap across VLA models and suite sizes;
- an unadjusted ROC-AUC-versus-coverage plot and the full conditional-coverage
  table in the supplement. The unadjusted plot is descriptive and must not be
  interpreted as a coverage-only causal effect.

## Validity checks

- Verify exact class balance and set inclusion for every nested suite.
- Verify disjoint pool, calibration, and evaluation episode IDs.
- Verify identical held-out trajectories across suite sizes and techniques.
- Verify deterministic subset hashes under the frozen seed.
- Verify finite scores and metrics for all 31 techniques.
- Retain non-monotonic learning curves rather than repairing them.
- Do not infer causal voxel localization from trajectory-level ROC-AUC.
