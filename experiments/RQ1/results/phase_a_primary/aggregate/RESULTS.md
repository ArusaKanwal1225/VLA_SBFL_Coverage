# RQ1 Phase A results

This run evaluates held-out failure discrimination from voxel suspiciousness. It does not establish causal voxel localization because Phase B ground truth is absent.

| Rank | Technique | Macro ROC-AUC | 95% CI | Macro AP | Mean rank | Equivalent |
|---:|---|---:|---:|---:|---:|:---:|
| 1 | Harmonic Mean | 0.7039 | [0.6796, 0.7280] | 0.7743 | 1.94 | yes |
| 2 | Rogot2 | 0.7039 | [0.6796, 0.7280] | 0.7743 | 1.94 | yes |
| 3 | Dennis | 0.6880 | [0.6668, 0.7104] | 0.7508 | 3.69 | no |
| 4 | Phi (Geometric Mean) | 0.6818 | [0.6595, 0.7054] | 0.7427 | 4.81 | no |
| 5 | Tarwid | 0.6549 | [0.6357, 0.6754] | 0.7144 | 6.56 | no |
| 6 | Arithmetic Mean | 0.6530 | [0.6323, 0.6761] | 0.7153 | 6.75 | no |
| 7 | Michael | 0.6451 | [0.6237, 0.6686] | 0.7107 | 8.12 | no |
| 8 | Cohen | 0.6429 | [0.6213, 0.6668] | 0.7095 | 8.50 | no |
| 9 | Zoltar | 0.6428 | [0.6142, 0.6723] | 0.7345 | 9.31 | no |
| 10 | Mountford | 0.5858 | [0.5414, 0.6333] | 0.6931 | 11.31 | no |

Friedman: statistic=393.9708, df=30, p=4.61134e-65.

Final wording must remain **failure-associated voxel ranking** until Phase B is completed.
