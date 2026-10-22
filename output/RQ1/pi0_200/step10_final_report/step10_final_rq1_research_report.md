# RQ1 Final Result: Pi0, 200 scenes

## Research question

Which of the 14 spatial SBFL methods best discriminates failed versus
successful held-out Pi0 trajectories under the fixed 200-scene protocol?

## Experimental design

- Model: Pi0. Test-generation algorithms: DH, HybridDT, HybridRF, WeightedRF.
- Ten 200-episode runs per algorithm; 2,000 episodes per algorithm; 8,000 total.
- Ten repeated 70/30 candidate-scene-grouped splits per algorithm.
- Four algorithm means weighted equally for the overall descriptive comparison.
- Robot TCP XYZ visited voxels, 0.02-m grid, origin (-1, -1, 0).
- Compute 14 voxel suspiciousness formulas on TRAINING episodes only.
- Score each TEST episode by maximum known-voxel suspiciousness.
- Evaluate ROC-AUC (primary) and Average Precision (secondary).
- Test failure prevalence is the no-skill AP baseline.
- All 14 methods in the same algorithm/split share the exact same held-out episodes.

## Final descriptive method ranking

| Rank | SBFL technique | Mean ROC-AUC | Mean AP | Mean AP - prevalence |
|---:|---|---:|---:|---:|
| 1 | Zoltar | 0.7130 | 0.7980 | 0.1589 |
| 2 | Ochiai2 | 0.6671 | 0.7585 | 0.1194 |
| 3 | Tarantula | 0.6218 | 0.7091 | 0.0700 |
| 4 | Barinel | 0.6218 | 0.7091 | 0.0700 |
| 5 | Ample | 0.5314 | 0.6654 | 0.0263 |
| 6 | Wong2 | 0.4902 | 0.6544 | 0.0153 |
| 7 | Kulczynski2 | 0.4233 | 0.6053 | -0.0338 |
| 8 | Ochiai | 0.4227 | 0.6052 | -0.0339 |
| 9 | Jaccard | 0.4227 | 0.6052 | -0.0339 |
| 10 | DStar2 | 0.4227 | 0.6051 | -0.0340 |
| 11 | GP13 | 0.4227 | 0.6051 | -0.0340 |
| 11 | OP2 | 0.4227 | 0.6051 | -0.0340 |
| 11 | Wong3 | 0.4227 | 0.6051 | -0.0340 |
| 14 | Wong1 | 0.4227 | 0.6051 | -0.0340 |

## Answer to RQ1 for the current configuration

**Zoltar** has the highest descriptive equal-algorithm mean held-out
ROC-AUC (0.7130), followed by
**Ochiai2** (0.6671).
Their observed paired mean delta AUC is **+0.0459** in favor
of Zoltar; 4/4 algorithms have a positive mean delta.
These are descriptive estimates, not a formal hypothesis-test result.

## Algorithm-specific best methods (ties shown)

- DH_200scenes: Zoltar (mean ROC-AUC 0.7522).
- DH_HybridDT_200scenes: Zoltar (mean ROC-AUC 0.7076).
- DH_HybridRF_200scenes: Zoltar (mean ROC-AUC 0.6665).
- DH_WeightedRF_200scenes: Zoltar (mean ROC-AUC 0.7258).

## Protocol checks

- All 40 algorithm/seed held-out splits have zero overlap under the
  candidate scene-group key.
- The Step08 and Step09 frozen input/result hashes are verified.
- Step07 found 23,994 held-out
  prediction rows (repeated across split seeds).
- Test labels did not influence suspiciousness estimation or episode risk;
  they were used only for the evaluation metrics.

## Limits and threats to validity

- ROC-AUC/AP measure held-out failure discrimination, **not** whether
  an identified voxel was a causal fault location.
- The ten split seeds overlap in training and testing episodes.
  Do not treat 40 algorithm/split blocks as independent replicates;
  there are no naive p-values or 95% confidence intervals.
- The scene-group key (object_name, orientation_id, X, Y) is a proxy
  for scene identity. Other simulator settings may need verification.
- The leading technique was selected using this dataset; the observed
  gap could be optimistic. Validate on 400-scene budgets and other models.
- Failure prevalence varies by algorithm; interpret AP relative to its
  held-out prevalence baseline.
- The max-visited-voxel rule can emphasize low-support suspicious voxels;
  keep sparse-voxel diagnostics and assess sensitivity if reporting strong claims.

## Next experimental extension

Repeat the frozen RQ1 protocol for Pi0 400 scenes and other VLA models.
Then address RQ2 by varying training spatial coverage at fixed test
episodes and approximately matched training sample sizes.

## Output index

- final_method_leaderboard.csv, final_algorithm_method_table.csv
- final_leader_vs_others.csv, final_leader_runner_by_algorithm.csv
- final_heldout_class_balance.csv, scene_group_leakage_checks.csv
- final_protocol_checks.csv, input_file_checksums.csv
- figures/ (5 PNG and 5 PDF); step10_validation_report.json
