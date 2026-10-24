# RQ3 protocol: effect of voxel coverage on SBFL techniques

**Status:** implemented and smoke-validated. The confirmatory experiment has
not been executed.

## Research question

> How does training voxel coverage affect the held-out failure-discrimination
> performance and relative ranking of spectrum-based techniques for VLA models,
> after controlling for the number of training trajectories?

RQ3 studies coverage rather than using trajectory count as a proxy for
coverage. At every fixed training size, it compares outcome-balanced subsets
with relatively low, median, and high spatial coverage.

This remains a Phase A trajectory-level study. The available pass/fail labels
do not identify the physical voxel that caused a failure, so the experiment
evaluates failure-associated voxel ranking rather than causal localization.

## Hypotheses

- **H3.1:** Increasing training coverage improves held-out ROC-AUC after
  controlling for training-set size.
- **H3.2:** The improvement exhibits diminishing returns as coverage approaches
  saturation.
- **H3.3:** The relative performance of the 31 SBFL techniques changes with
  coverage; some techniques are more robust in sparse regimes.

## Dataset and blocking

Use the same benchmark factors as RQ1 and RQ2:

- VLA: `EO1`, `Groot`, `Pi0`, and `SpatialVLA`.
- Condition: `DH`, `DH_HybridDT`, `DH_HybridRF`, and `DH_WeightedRF`.
- Budget: 200 and 400 scenes.
- Runs: ten per VLA/condition/budget configuration.
- Techniques: all 31 Table 4 formulas frozen in RQ1.

The primary inferential block is `VLA x condition`. The two scene budgets are
averaged equally inside each block, matching RQ1 and RQ2.

## Leakage-safe three-way split

For every outer held-out run:

1. The outer run is the untouched test set.
2. The next run in cyclic order is the calibration set.
3. The remaining eight runs form the candidate training pool.

Calibration trajectory coordinates, but not their outcomes, determine the
coverage tier of training subsets. Calibration outcomes are used only for
post-selection class-conditional coverage diagnostics. The test run is never
used to choose a subset.

Rotating the calibration run ensures that each run serves as test once and as
calibration once across the ten outer folds.

## Controlled subset selection

The primary training sizes are `10`, `20`, `50`, `100`, and `200`
trajectories. All sizes are even so every subset contains exactly 50% passing
and 50% failing trajectories. This fixes the SBFL class totals within a
training-size stratum and prevents failure prevalence from masquerading as a
coverage effect.

For every configuration, test fold, and training size:

1. Draw 200 distinct class-balanced candidate subsets using a deterministic
   seed.
2. Form the union of voxels visited by each candidate.
3. Measure the fraction of unique calibration voxels covered by that union.
4. Select the distinct candidates nearest the empirical 10th, 50th, and 90th
   percentiles. These are the `low`, `medium`, and `high` tiers.
5. Fit all 31 techniques on each selected subset and evaluate them on the same
   outer test run.

The default design uses one candidate-bank replicate. The ten outer folds and
16 primary blocks provide matched replication. Additional independent
candidate-bank replicates can be requested as a sampling-robustness analysis.

Coverage tiers are relative within each configuration/fold/training-size cell.
The actual achieved coverage is always retained and reported; tier labels are
not assumed to represent identical absolute coverage across VLAs.

## Voxel representation

The primary experiment uses voxel size `0.02`, the RQ1 reference grid:

- Coordinate frame: `robot_base`.
- Origin: `(-1, -1, 0)`.
- Voxel index: `floor((xyz - origin) / voxel_size)`.
- Coverage: binary per trajectory.

Voxel sizes `0.01` and `0.05` are secondary robustness analyses. They must be
analyzed as separate strata because RQ2 showed that voxel size strongly changes
both sparsity and technique rankings. Voxel size must not be used as the RQ3
coverage manipulation.

The scoring policy remains unchanged:

- suspiciousness is fitted from training-only `ef`, `ep`, `nf`, and `np`;
- voxel scores are converted to empirical training percentiles;
- trajectory risk is the maximum known-voxel percentile;
- partially unknown voxels are ignored and reported;
- a trajectory with no known voxel receives neutral score `0.5`;
- undefined and infinite formula values use the frozen finite RQ1 policy.

## Coverage outcomes

The selection metric is:

`calibration_unique_coverage = |V_train intersect V_calibration| / |V_calibration|`.

The analysis also reports:

- unique held-out test-voxel coverage;
- mean per-trajectory known-voxel fraction on calibration and test data;
- pass- and fail-specific mean known fractions on the test set;
- number of all-unknown test trajectories;
- occupied training voxels and score-tie fraction.

The mean per-trajectory known fraction preserves the repository's binary
per-trajectory spectrum semantics. A step-frequency-weighted measure would
require retaining raw visit multiplicities and is outside this frozen design.

## Effectiveness outcomes

Primary outcome:

- Held-out trajectory ROC-AUC; no classification threshold is selected.

Secondary outcomes:

- Average precision together with failure prevalence.
- Technique rank at low, medium, and high coverage.
- High-minus-low ROC-AUC and coverage changes within matched cells.
- Coverage-performance results at every training size.
- Fit/scoring time and optional model artifact size.

Unknown test trajectories are retained with the frozen neutral policy. Dropping
them would condition on successful coverage and could make sparse subsets appear
artificially strong.

## Confirmatory analysis

For the primary `0.02` grid:

1. Average repetitions within each configuration and then average the two
   budgets equally inside each `VLA x condition` block.
2. For every technique, compute the matched high-minus-low ROC-AUC effect at
   every training size and average training sizes equally inside each block.
3. Report the mean high-minus-low effect across the 16 blocks.
4. Test technique-by-coverage interactions against Harmonic Mean using paired
   Wilcoxon signed-rank tests on the 16 block effects.
5. Apply one Holm correction across the 30 non-reference techniques.
6. Call an interaction meaningful only when corrected `p < 0.05` and the
   absolute interaction is at least `0.01` ROC-AUC.

The intended-treatment contrast uses calibration-selected high and low tiers.
Actual test coverage is reported to verify that the manipulation generalizes.
Coverage-response plots should additionally use achieved continuous coverage
and distinguish training size by color or facet.

## Interpretation rules

- A positive high-minus-low contrast supports a coverage benefit at fixed
  trajectory count.
- A simple improvement with training size does not by itself support a coverage
  effect.
- A changed descriptive winner does not establish an interaction unless the
  paired corrected test and practical threshold agree.
- Conclusions about `0.01` or `0.05` are sensitivity results unless those grids
  are run with the complete matched design.
- No result may be described as causal voxel localization without independent
  problematic-region ground truth.

## Pre-launch checks

- Verify exact 50/50 pass/fail balance in every selected subset.
- Verify disjoint training, calibration, and test episode IDs.
- Verify that low, medium, and high candidates are distinct.
- Verify that selection uses calibration coordinates only.
- Verify finite scores and metrics for all 31 formulas.
- Verify that calibration and test coverage generally increase from low to
  high; retain inversions rather than silently replacing them.
- Run one `EO1/DH/200` fold as a smoke test before full execution.
- Freeze the protocol and code hashes before inspecting confirmatory results.
