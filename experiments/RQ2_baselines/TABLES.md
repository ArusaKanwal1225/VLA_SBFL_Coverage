# RQ2 generated tables

`generate_tables.py` creates three paper-facing tables:

1. `rq2_baseline_performance`: ROC-AUC by VLA and overall for methods complete
   across the selected design.
2. `rq2_baseline_statistics`: paired comparisons with Harmonic Mean.
3. `rq2_metric_availability`: availability of every external metric by VLA.

`Optimal Trajectory` is marked with a dagger because it uses task state. The
performance table is a failure-discrimination comparison and must not be
described as a voxel-localization ranking.

The performance and availability LaTeX tables always reserve columns for EO1,
GR00T, Pi0, and SpatialVLA. Before the four-VLA execution, uncomputed cells plus
the overall result and rank remain blank, so the generated files can already be
placed in Overleaf without presenting an EO1-only value as the global result.
