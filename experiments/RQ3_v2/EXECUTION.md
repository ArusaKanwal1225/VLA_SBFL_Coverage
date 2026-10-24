# RQ3 v2 execution record

## Smoke validation

The five-size smoke design completed successfully for `EO1/DH/200`, outer
evaluation `run_1`, one nested-suite repetition, and voxel size `0.02`.

| Spectrum trajectories | Calibration unique coverage | Held-out unique coverage |
|---:|---:|---:|
| 10 | 8.5% | 8.0% |
| 20 | 16.3% | 16.2% |
| 50 | 33.2% | 35.0% |
| 100 | 51.0% | 51.4% |
| 200 | 70.9% | 71.8% |

All five suites were exactly outcome-balanced and strictly nested. The run
produced 155 finite technique-size metric rows (31 techniques at five sizes),
and the aggregate learning-curve and sample-efficiency reports completed.
Confirmatory tests correctly remained disabled because a one-fold,
one-repetition smoke run is not the complete 16-block design.

The complete repository test suite passed after the implementation.

## Full experiment

The full primary experiment completed successfully on 2026-10-24:

- 16,000 of 16,000 fitted subset cells;
- 496,000 finite technique-level metric rows;
- 32 VLA/condition/budget configurations, ten folds, ten repetitions, five
  nested sizes, and all 31 techniques;
- elapsed fitting and aggregation time: 687.5 seconds;
- all three planned statistical analyses completed;
- PNG and PDF versions of all three planned figures generated.

Across all benchmark blocks, mean calibration coverage increased from 7.1% at
10 trajectories to 62.2% at 200 trajectories. Harmonic Mean's mean ROC-AUC
increased significantly at every adjacent size transition: `+0.0187`,
`+0.0296`, `+0.0268`, and `+0.0285`. Every transition was positive in all 16
primary blocks and had Holm-adjusted `p = 0.00193`.

Dennis had the largest normalized learning-curve area (`0.5994`), followed by
Rogot2 (`0.5972`) and Harmonic Mean (`0.5971`). Dennis's advantage over
Harmonic Mean was small (`+0.0023`), was below the `0.01` practical threshold,
and was not statistically significant after Holm correction (`p = 0.2786`).
Harmonic Mean led descriptively at 200 trajectories with ROC-AUC `0.6534`.

After controlling for trajectory count through within-size centering, none of
the 31 technique-specific coverage slopes was significant after Holm
correction. This supports interpreting the strong across-size improvement as a
joint test-suite-size/coverage effect rather than a demonstrated independent
coverage effect.

The completed experiment can be reproduced or resumed from the repository root
with:

```bash
PYTHONPATH="$PWD:$PWD/src_new" python experiments/RQ3_v2/run_experiment.py
```
