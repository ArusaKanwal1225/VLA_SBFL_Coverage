# RQ1 final descriptive report: Pi0, 200 scenes

## Study design
Four DeepHyperion-based algorithms, 10 fixed 70/30 scene-grouped splits per
algorithm, and 14 SBFL suspiciousness formulas applied to spatial TCP voxels.
Voxel size: 0.02 m; origin: (-1,-1,0). Episode risk is the maximum known-
training-voxel suspiciousness. ROC-AUC and Average Precision use held-out labels.

## Main result
**Descriptive first place:** Zoltar, mean ROC-AUC 0.713047,
mean AP 0.798002.
**Second:** Ochiai2, mean ROC-AUC 0.667142,
mean AP 0.758454.
Mean paired ROC-AUC advantage: **+0.045905**;
positive algorithm-level mean differences: **4/4**.

## Important limitations
- The 10 split seeds reuse episodes, so these 40 split observations are not
  40 independent experiments. Differences and ranks are descriptive only;
  this report does not claim statistically significant superiority.
- The scene-group identity is based on inferred configuration fields and
  may omit simulator state not present in the extracted input.
- ROC-AUC/AP are metrics of held-out *failure discrimination*, not proof of
  the physical voxel that actually caused a failure.
- Conclusions here apply to Pi0 200-scene data, not all VLA models.

## Files
See final_method_leaderboard.csv, final_by_algorithm_method.csv,
paired_by_split.csv.gz, paired_by_algorithm.csv, paired_overall.csv,
leader_vs_other_methods.csv, and the saved figure PNG/PDF pairs.
