# RQ1 Final Result: Pi0, 400 scenes

## Research question

Which of the 14 spatial SBFL methods best discriminates failed versus
successful held-out Pi0 trajectories under the fixed 400-scene protocol?

## Experimental design

- Model: Pi0. Test-generation algorithms: DH, HybridDT, HybridRF, WeightedRF.
- Ten 400-episode runs per algorithm; 4,000 episodes per algorithm; 16,000 total.
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
| 1 | Zoltar | 0.7058 | 0.8185 | 0.1421 |
| 2 | Barinel | 0.6754 | 0.7742 | 0.0978 |
| 3 | Tarantula | 0.6754 | 0.7742 | 0.0978 |
| 4 | Ochiai2 | 0.6645 | 0.7907 | 0.1143 |
| 5 | Ample | 0.5254 | 0.7008 | 0.0244 |
| 6 | Wong2 | 0.4539 | 0.6716 | -0.0048 |
| 7 | Kulczynski2 | 0.4090 | 0.6387 | -0.0377 |
| 8 | Ochiai | 0.4085 | 0.6386 | -0.0378 |
| 9 | Jaccard | 0.4085 | 0.6386 | -0.0378 |
| 10 | DStar2 | 0.4084 | 0.6385 | -0.0378 |
| 11 | GP13 | 0.4083 | 0.6385 | -0.0378 |
| 11 | OP2 | 0.4083 | 0.6385 | -0.0378 |
| 11 | Wong3 | 0.4083 | 0.6385 | -0.0378 |
| 14 | Wong1 | 0.4083 | 0.6385 | -0.0378 |

## Answer to RQ1 for the current configuration

**Zoltar** has the highest descriptive equal-algorithm mean held-out
ROC-AUC (0.7058), followed by
**Barinel** (0.6754).
Their observed paired mean delta AUC is **+0.0304** in favor
of Zoltar; 3/4 algorithms have a positive mean delta.
These are descriptive estimates, not a formal hypothesis-test result.

## Algorithm-specific best methods (ties shown)

- DH_400scenes: Barinel, Tarantula (mean ROC-AUC 0.7369).
- DH_HybridDT_400scenes: Zoltar (mean ROC-AUC 0.7017).
- DH_HybridRF_400scenes: Zoltar (mean ROC-AUC 0.6645).
- DH_WeightedRF_400scenes: Zoltar (mean ROC-AUC 0.7539).

## Protocol checks

- All 40 algorithm/seed held-out splits have zero overlap under the
  exact scene key, reviewed near-duplicate clusters and merged split key.
- The Step08 and Step09 frozen input/result hashes are verified.
- Step07 found 47,975 held-out
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
  gap could be optimistic. Validate across other scene budgets and VLA models.
- Failure prevalence varies by algorithm; interpret AP relative to its
  held-out prevalence baseline.
- The max-visited-voxel rule can emphasize low-support suspicious voxels;
  keep sparse-voxel diagnostics and assess sensitivity if reporting strong claims.

## Next experimental extension

Compare Pi0 400-scene results descriptively with earlier 200-scene results.
This is an UNPAIRED budget comparison and not a controlled causal estimate.
Then address RQ2 by varying training spatial coverage at fixed test
episodes and approximately matched training sample sizes.

## Output index

- final_method_leaderboard.csv, final_algorithm_method_table.csv
- final_leader_vs_others.csv, final_leader_runner_by_algorithm.csv
- final_heldout_class_balance.csv, scene_group_leakage_checks.csv
- final_protocol_checks.csv, input_file_checksums.csv
- figures/ (5 required PNG/PDF pairs, plus optional Pi0 200 vs 400 pair);
- step10_validation_report.json and optional pi0_200_vs_400_descriptive.csv
