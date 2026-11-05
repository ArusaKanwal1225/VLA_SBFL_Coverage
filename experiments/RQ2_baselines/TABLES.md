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

The reporting script also produces `rq2_baseline_performance_top5_rq1.tex`
and `.csv`, combining the complete baselines with the techniques ranked in
the top five by the RQ1 overall ROC-AUC. Ties at the fifth rank are included.
A star marks SBFL techniques and a dagger marks task-aware baselines. Harmonic
Mean is included only once. The existing statistical-comparison table still
compares baselines against Harmonic Mean; adding rows to the performance table
does not add inferential tests.

The complete four-VLA RQ2 analysis is available under
`results/baseline_comparison_all_models/`, using the baseline metrics in
`input/`. Its `tables/` directory contains the extended table with populated
VLA, Overall, and Rank columns. The older `results/baseline_comparison/` run
covers EO1 only and is preserved separately. For partial runs, incomplete
overall values and joint rankings are left blank.

For a different RQ1 aggregate directory, pass `--rq1-aggregate-dir PATH` to
`generate_tables.py`. You can also generate only the extended table with
`generate_rq2_top5_rq1.py --rq1-aggregate PATH --rq2-aggregate PATH
--output-dir PATH`.
