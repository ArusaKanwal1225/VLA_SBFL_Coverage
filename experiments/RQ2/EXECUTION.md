# RQ2 execution record

**Status:** completed successfully with the expanded 34-technique catalogue on
2026-10-24.

The frozen RQ2 Phase A experiment executed 320 folds at voxel size `0.01` and
320 folds at `0.05`, then validated and reused the 320 completed RQ1 folds at
`0.02`. The matched analysis therefore contains 960 size/fold cells and 32,640
formula-size-fold measurements.

## Headline results

- Harmonic Mean size comparison: `0.02` = `0.7039`, `0.01` = `0.6946`, and
  `0.05` = `0.6930` macro ROC-AUC.
- The Harmonic Mean size effect was not significant: Friedman statistic
  `1.625`, `p = 0.443747`.
- `0.01` and `0.02` satisfy the frozen practical-equivalence rule for Harmonic
  Mean; `0.05` falls just outside the `0.01` ROC-AUC margin.
- Technique leader at `0.01`: Harmonic Mean/Rogot2, `0.6946` ROC-AUC.
- Technique leader at `0.02`: Harmonic Mean/Rogot2, `0.7039` ROC-AUC.
- Technique leader at `0.05`: Tarwid, `0.7295` ROC-AUC. Tarantula obtained the
  same rounded ROC-AUC and joined Tarwid in the within-size practical-equivalence
  group.
- Eight technique-by-size interaction contrasts pass both Holm-corrected
  significance and the frozen absolute `0.01` practical threshold.

These are trajectory-level failure-discrimination results. Selecting the best
spatial localization resolution remains pending Phase B problematic-region
ground truth.

## Verification

- All 640 newly computed folds completed.
- The 320 reused folds passed provenance, grid, formula-catalogue, and identity
  checks.
- All 32,640 critical metric rows are finite and bounded.
- No held-out trajectory was entirely unsupported at any size.
- Sampled saved models at both new sizes reloaded successfully.
- All automated tests passed after execution.

The complete results are under `experiments/RQ2/results/phase_a_granularity`.
The generated concise report is `aggregate/RESULTS.md` and the machine-readable
conclusion is `aggregate/summary.json`.
