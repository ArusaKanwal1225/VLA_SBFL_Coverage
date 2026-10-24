# RQ2 protocol: voxel granularity and SBFL technique interaction

**Status:** prepared and smoke-validated on 2026-10-23; the full confirmatory
experiment has not yet been executed.

## Research question

> How does voxel granularity affect the effectiveness, robustness, and
> computational cost of spectrum-based fault localization for VLA models, and
> does the best SBFL technique depend on voxel size?

The confirmatory Phase A question is whether voxel size changes held-out
failed-versus-passed trajectory discrimination. Direct spatial localization
remains a Phase B question because the current data do not identify the voxel or
physical region that caused an episode to fail.

## Relationship to RQ1

RQ1 froze `0.02` coordinate units as its primary voxel size and listed `0.01`
and `0.05` as sensitivity analyses. RQ2 elevates that sensitivity analysis to a
separate matched experiment without changing the RQ1 result.

The completed RQ1 `0.02` artifacts are reused, rather than recomputed, after the
RQ2 runner verifies their grid definition, completeness, formula catalogue, and
fold identities. The `0.01` and `0.05` cells use the unchanged RQ1 execution
engine. RQ2 owns only the cross-size orchestration and statistical analysis.

Harmonic Mean is the frozen reference technique because RQ1 placed it in a
numerically identical first-place tie with Rogot2 for all four VLAs. Rogot2 is
still evaluated and reported; choosing one reference avoids duplicate tests.

## Dataset and matched design

Use exactly the RQ1 Phase A benchmark:

- VLA: `EO1`, `Groot`, `Pi0`, and `SpatialVLA`.
- Trajectory condition: `DH`, `DH_HybridDT`, `DH_HybridRF`, and
  `DH_WeightedRF`.
- Budget: 200 and 400 scenes.
- Repetitions: 10 runs per VLA/condition/budget.
- Validation: leave one complete run out, train on the other nine runs.

The factors are:

- Voxel size: `0.01`, `0.02`, and `0.05` coordinate units.
- SBFL technique: all 31 Table 4 techniques frozen in RQ1.

Every size uses the same episode IDs, labels, train/test folds, coordinate frame,
origin, coverage rule, formula catalogue, edge-case policy, normalization, and
trajectory aggregation. This produces 960 size/fold cells and 29,760
technique-size-fold measurements. Only 640 folds are newly fitted because the
320 `0.02` folds already exist.

As in RQ1, seven exact trajectories overlap across the 200- and 400-scene
datasets and semantic overlap cannot be ruled out. The primary inferential unit
is therefore one of 16 `VLA x trajectory condition` blocks. Results are averaged
over runs within budget and then equally over the two budgets within each block.

## Frozen representation

- Coordinate frame: `robot_base`.
- Voxel sizes: `0.01`, `0.02`, `0.05`.
- Origin: `(-1, -1, 0)` for every size.
- Voxel index: `floor((xyz - origin) / voxel_size)`.
- Coverage: binary per trajectory.
- Suspiciousness input: training-only `ef`, `ep`, `nf`, and `np`.
- Voxel normalization: empirical percentile over training voxels.
- Trajectory score: maximum percentile over known visited voxels.
- Unknown voxels: ignored if known voxels exist; otherwise neutral score `0.5`.
- Undefined and infinite formula values: the frozen RQ1 finite policy, with
  diagnostics retained.

The fixed origin isolates the requested size comparison. Grid-boundary shifts
are a separate robustness analysis and cannot replace the primary result.

## Outcomes

Primary effectiveness outcome:

- Macro ROC-AUC over the 16 primary blocks.

Secondary effectiveness outcomes:

- Average precision together with failure prevalence.
- Technique rank at every voxel size.
- Pairwise rank correlation between voxel sizes.
- Known-voxel fraction and all-unknown trajectories.
- Distinct trajectory scores and tied-score fraction.
- Undefined and infinite formula counts.

Efficiency and sparsity outcomes:

- Number of occupied training voxels.
- Model artifact size.
- Fit and scoring time.
- Distribution of trajectory support per voxel when Phase B is added.

No classification threshold is selected.

## Confirmatory statistical analysis

### RQ2.1: voxel-size effect for the frozen RQ1 leader

1. Use Harmonic Mean ROC-AUC in the 16 matched primary blocks.
2. Apply a paired Friedman omnibus test across the three sizes.
3. If significant, compare the size with the highest macro ROC-AUC against the
   other sizes using paired Wilcoxon signed-rank tests with Holm correction.
4. Report paired effect sizes and matched hierarchical-bootstrap 95% confidence
   intervals. The bootstrap resamples primary blocks, then run folds within each
   budget, using identical resamples for every size and technique.
5. Treat sizes as practically equivalent when their macro ROC-AUC differs by
   less than `0.01` and the corrected comparison is not significant.

### RQ2.2: technique-by-size interaction

For each technique and each non-primary size:

1. Estimate the paired ROC-AUC change relative to `0.02`.
2. Estimate a difference-in-differences relative to Harmonic Mean:

   `(technique_size - technique_0.02) - (HM_size - HM_0.02)`.

3. Test the 60 non-reference interaction contrasts using paired Wilcoxon tests
   with one Holm correction family.
4. Require both corrected `p < 0.05` and an absolute interaction of at least
   `0.01` ROC-AUC to call a practically meaningful interaction.
5. Report complete technique rankings at every size and their pairwise Spearman
   rank correlations.

Technique-specific size changes are also reported for all 62 contrasts (31
techniques by two non-primary sizes) with a separate Holm correction. They are
secondary because a size main effect is not itself evidence of an interaction.

The analysis must not select a formula separately on each dataset and then use
the same observations to claim confirmatory superiority. A changed winner is a
descriptive result unless its paired comparison and the interaction criterion
both support it.

## Selecting a voxel size

Phase A can select the best size only for trajectory-level failure
discrimination. The selection rule is:

1. Retain every size in the practical equivalence group around the Phase A
   leader.
2. If one size is outside that group and significantly better, identify it as
   the Phase A discrimination leader.
3. Do not call it the best localization size until Phase B is available.
4. In Phase B, prefer lower physical localization error and better
   volume-normalized region retrieval among Phase A-equivalent sizes.
5. Use lower runtime and artifact size only as the final tie-breaker.

This prevents a coarse grid from winning solely because smoothing improves
trajectory discrimination while destroying useful spatial precision.

## Phase B requirements

The same independently obtained problematic positions or regions specified by
RQ1 are required. Report localization in physical coordinate units so different
grid resolutions remain comparable:

- Distance from the reference region to the highest-ranked voxel.
- Region retrieval at equal physical-volume budgets.
- Voxel-level average precision and ROC-AUC, with class prevalence.
- Hit rate and reciprocal rank, accompanied by the physical search volume.

If Phase B is absent, conclusions must use "failure-associated voxel ranking"
and not causal fault localization.

## Pre-launch checks

- Verify that the completed `0.02` source contains all 320 folds and all 31
  techniques.
- Verify identical configuration, held-out run, episode count, failure count,
  and formula identities across sizes.
- Verify all three grids use the frozen origin and coordinate frame.
- Run unit tests for matched bootstrap, interaction contrasts, ranking
  correlation, and cell validation.
- Run one smoke fold at `0.01` and `0.05` before the full 640-fold execution.
- Freeze this protocol and the RQ2 code hashes before examining cross-size
  results.

### Smoke-validation record

The implementation completed one `EO1/DH/200` held-out fold at both `0.01` and
`0.05`. Both cells produced all 31 finite formula metrics and reloadable model
artifacts. The observed known-voxel fractions (`0.895` and `0.997`, respectively)
are execution diagnostics only and are not inferential RQ2 results.
