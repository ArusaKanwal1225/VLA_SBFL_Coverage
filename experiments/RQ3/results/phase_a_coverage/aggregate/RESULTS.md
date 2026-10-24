# RQ3 Phase A results

Status: **complete** (4800 of 4800 planned cells).

Training subsets are outcome-balanced and selected by calibration coverage; the final test run is not used for subset selection.

| Voxel size | Coverage tier | Leading technique | Macro ROC-AUC |
|---:|---|---|---:|
| 0.02 | high | Dennis | 0.6008 |
| 0.02 | low | Dennis | 0.5995 |
| 0.02 | medium | Dennis | 0.5987 |

These are trajectory-level discrimination results. They do not identify causal fault voxels without independent spatial ground truth.
