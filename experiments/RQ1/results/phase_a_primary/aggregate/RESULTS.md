# RQ1 Phase A results

This run evaluates held-out failure discrimination from voxel suspiciousness. It does not establish causal voxel localization because Phase B ground truth is absent.

| Rank | Technique | Macro ROC-AUC | 95% CI | Macro AP | Mean rank | Equivalent |
|---:|---|---:|---:|---:|---:|:---:|
| 1 | Harmonic Mean | 0.7039 | [0.6796, 0.7280] | 0.7743 | 2.00 | yes |
| 2 | Rogot2 | 0.7039 | [0.6796, 0.7280] | 0.7743 | 2.00 | yes |
| 3 | Dennis | 0.6880 | [0.6668, 0.7104] | 0.7508 | 3.88 | no |
| 4 | Phi (Geometric Mean) | 0.6818 | [0.6595, 0.7054] | 0.7427 | 5.00 | no |
| 5 | Tarantula | 0.6549 | [0.6357, 0.6754] | 0.7145 | 7.19 | no |
| 6 | Tarwid | 0.6549 | [0.6357, 0.6754] | 0.7144 | 7.19 | no |
| 7 | Arithmetic Mean | 0.6530 | [0.6323, 0.6761] | 0.7153 | 7.25 | no |
| 8 | Michael | 0.6451 | [0.6237, 0.6686] | 0.7107 | 8.75 | no |
| 9 | Cohen | 0.6429 | [0.6213, 0.6668] | 0.7095 | 9.19 | no |
| 10 | Zoltar | 0.6428 | [0.6142, 0.6723] | 0.7345 | 10.19 | no |

Friedman: statistic=440.6350, df=33, p=8.93189e-73.

Final wording must remain **failure-associated voxel ranking** until Phase B is completed.
