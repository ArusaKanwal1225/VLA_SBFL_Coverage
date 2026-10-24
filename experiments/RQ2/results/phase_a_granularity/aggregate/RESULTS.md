# RQ2 Phase A results

This analysis compares voxel sizes for held-out failure discrimination. It does not determine spatial localization quality without Phase B ground truth.

## Voxel-size comparison for Harmonic Mean

| Rank | Voxel size | Macro ROC-AUC | 95% CI | Delta vs 0.02 | Equivalent |
|---:|---:|---:|---:|---:|:---:|
| 1 | 0.02 | 0.7039 | [0.6796, 0.7280] | +0.0000 | yes |
| 2 | 0.01 | 0.6946 | [0.6727, 0.7178] | -0.0093 | yes |
| 3 | 0.05 | 0.6930 | [0.6652, 0.7188] | -0.0110 | no |

Friedman: statistic=1.6250, df=2, p=0.443747.

## Technique leader by voxel size

| Voxel size | Technique | Macro ROC-AUC |
|---:|---|---:|
| 0.01 | Harmonic Mean | 0.6946 |
| 0.02 | Harmonic Mean | 0.7039 |
| 0.05 | Tarwid | 0.7295 |

Practically meaningful technique-by-size interactions: **5**.

Final wording must remain **failure-associated voxel ranking** until Phase B is completed.
