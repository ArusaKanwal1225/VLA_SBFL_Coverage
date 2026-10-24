# RQ3 v2 results

Status: **complete** (16000 of 16000 planned subset cells).

| Spectrum trajectories | Leading technique | Macro ROC-AUC | Calibration coverage | Held-out coverage |
|---:|---|---:|---:|---:|
| 10 | Dennis | 0.5543 | 7.1% | 7.1% |
| 20 | Dennis | 0.5722 | 13.2% | 13.3% |
| 50 | Dennis | 0.6016 | 27.8% | 27.8% |
| 100 | Dennis | 0.6253 | 44.1% | 44.1% |
| 200 | Harmonic Mean | 0.6534 | 62.2% | 62.2% |

The descriptive sample-efficiency leader was **Dennis** with normalized log-scale learning-curve area `0.5994`.

Across-size improvements combine the effects of more spectrum observations and broader voxel coverage. The within-size analysis estimates the residual coverage association separately.
