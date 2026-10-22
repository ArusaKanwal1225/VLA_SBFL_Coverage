# RQ1 Step 09 - Paired descriptive comparison

## Experimental scope

- Model: Pi0; scene budget: 200; four algorithms; ten repeated 70/30 splits each.
- Methods: 14; observations: 560 algorithm x split x method ROC-AUC/AP results.
- All methods within each algorithm/seed are evaluated on identical held-out episodes.
- The repeated splits overlap: results are descriptive, not 40 independent replications.

## Descriptive findings

- Highest equal-algorithm mean ROC-AUC: **Zoltar** (0.7130).
- Second-highest: **Ochiai2** (0.6671).
- Paired mean ROC-AUC difference, Zoltar - Ochiai2: **0.0459**.
- Positive/negative algorithm mean differences (leader vs runner-up): 4/0 (0 ties).
- The figures and tables also report the prevalence-adjusted AP lift.

## Interpretation limitations

- No naive p-values, significance stars or confidence intervals are reported here.
- Using overlapping 70/30 splits as independent observations would overstate evidence.
- These results evaluate held-out **failure discrimination**; they do not identify ground-truth faulty voxels.
- Scene-group identity was inferred from source fields and has not been independently verified in full.

## Next step

Step 10: protocol checks, numerical summary, and careful RQ1 write-up. Consider a new independent dataset or other-model/budget replication before broad claims.
