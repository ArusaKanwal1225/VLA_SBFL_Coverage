# Replication details

## Environment and execution

Use Python 3.11 and requirements.txt. The tested clean environment used
Python 3.11.16. The launcher passes all paths explicitly, uses the active
interpreter, sets a noninteractive Matplotlib backend and limits BLAS/OpenMP
threads to one. No GPU/VLA software is imported by the analysis tools.

The commands verify, tables, stats, smoke and run use the paper numbering:
RQ1 → RQ1, RQ2 → RQ2_baselines, RQ3 → RQ2, RQ4 → RQ3_v2, RQ5 → RQ5.

Tables regenerates the released report without fitting. Stats additionally
recomputes all exact tests from aggregate observations. Smoke uses one RQ1 fold,
one baseline fold, one fine/coarse voxel-size fold and one five-size nested
learning-curve repetition; RQ5 tables are generated rather than fully rescored.
Run performs the full computations from recorded executions.

Choose an empty output directory. Only run supports --resume. RQ1–RQ4 resume
completed scientific cells after checking signatures. A completed RQ5 run is
skipped; interrupted RQ5 scoring needs a fresh output directory because its
original evaluator refuses nonempty outputs. Partial outputs are preserved.

Use --no-verify only for data already verified during extraction.
Use --no-figures to omit plots. A standalone RQ3 run adds RQ1 as a prerequisite:
its 0.02 m folds are reused by the voxel-size comparison.

## Dataset and storage

The ZIP contains 320 Pick-Up runs: four models × four generators × two budgets
× ten runs, totaling 96,000 episodes and 7,680,000 TCP samples.
The RQ5 candidate pool contains 504 executions per model, or 2,016 total.
Exact archive byte sizes and counts are in replication/archive.json.

Unpacking verifies every file while writing into a temporary staging directory,
then installs data atomically. It needs the full uncompressed size plus margin.
More than a million small baseline files consume additional filesystem blocks.
Start with 8 GiB RAM and at least 30 GiB free space; full fitting and retained
models may need more. ZIP size alone is not the extracted disk requirement.

Episodes.csv records model, condition, budget, run, eval_id, episode_id, failure,
object_name and num_steps. Trajectories.csv records episode_id,step,x,y,z.
Join by the globally unique episode_id and preserve step order.

Original baseline files retain their
results/<scene>/<model>/allMetrics/0 hierarchy. The frozen catalogue contains
13 candidates, eight of which have complete scores in the released data.
Unavailable files/arrays are not imputed and episodes are not removed.
The raw baseline score is the mean of finite numeric JSON leaves with the
catalogue's frozen direction.

Duplicate raw TCP/action streams, media and timing-only files are excluded.
Figure PDFs/PNGs in the historical reference artifacts are analysis outputs,
not execution videos.

## Frozen models and scientific parameters

RQ1 uses binary visits, 34 formulas, 0.02 m 3-D voxels, origin (-1,-1,0) and
robot_base coordinates. Each outer fold trains on nine complete runs and tests
on the tenth. Voxel normalization is training-only empirical percentile;
trajectory scoring uses the maximum supported percentile. All-unknown
trajectories receive 0.5. The primary blocks are 16 model × generator pairs,
averaging the two budgets within each block.

RQ3 evaluates the same folds at 0.01, 0.02 and 0.05 m. Table-only reproduction
uses the released training_voxels.csv: all 960 per-fold counts were recovered
from nine-run voxel unions and reconciled against every historical mean,
median, minimum and maximum in Table 8. This does not fit suspiciousness or use
test outcomes. The recovery command is:

~~~bash
python replication/voxel_diagnostics.py --data data --output reproduced/training_voxels.csv
~~~

RQ4 uses exactly balanced, nested suites of 10/20/50/100/200 trajectories,
ten repetitions, and 320 outer folds: 16,000 cells and 544,000 technique metrics.
The next cyclic run quantifies calibration coverage; it does not select suites.
The outer test run remains untouched.

RQ5's 72 scene-design XY cells (9×8, width 0.05 m, seven candidates each)
differ from the 0.02 m 3-D scoring voxels. Its retrospective selection fixes
100 passes and at least 100 failures per model, approximating each RQ1 ratio
subject to integer counts and spatial constraints. All methods within a model
use the same chosen episodes; selected IDs may differ between models.

## Exact statistical review

The latest source retains historical unsuffixed approximate results and writes
separate method-labelled reviews. Exact tests use two-sided conditional sign
enumeration after removing zeros, average ranks for absolute ties, and at most
20 nonzero pairs. The interpretation still requires valid paired sign changes;
exact enumeration does not establish block independence or symmetry.

Holm families are:

| Paper RQ | Family |
| --- | --- |
| RQ1 | All 561 unordered pairs among the 34 techniques; 33 Harmonic Mean comparisons are a presentation view |
| RQ2 | Eight baseline comparisons against Harmonic Mean |
| RQ3 | 68 technique-size effects and 66 interactions, corrected separately |
| RQ4 | Four adjacent-size, 33 sample-efficiency and 34 conditional-coverage tests, corrected separately |

The original runners retain historical defaults. The portable launcher runs
the explicit exact review after refitting. To render final-paper tables,
replication/paper.py copies aggregates into the output directory and supplies
approved exact p-values to the existing publication formatters. For RQ1 it
joins exact results by formula ID, retaining the original pointwise CIs.
The original inputs are never replaced.

A combined non-significance/small-gap label is descriptive, not a formal
equivalence test. Bootstrap intervals are pointwise and are not Holm-adjusted,
simultaneous, or adjusted for selecting the reference technique.

## Bootstrap sensitivity audit

The complete-block sensitivity audit can be regenerated without changing data:

~~~bash
python replication/bootstrap.py --data data --output reproduced/bootstrap
~~~

It copies RQ1–RQ3 aggregates into its output before invoking the source audit.
This recomputes historical and block-only intervals; it is separate from the
exact-test-only stats command.

## RQ5 provenance and Figure 5

Historical RQ1 audits/run metadata are under data/reference/rq1.
The released Pick-Up snapshot has documented differences from that audit.
RQ5 uses the exact frozen reconstructed maps and their unchanged metadata.

The latest compatibility gate pins scientific module bytes and selected RQ1
training/helper ASTs, separately verifies the historical full-code signature,
and verifies the released frozen-map manifest. Statistical reporting extensions
do not bypass these checks. Path relocation points to the installed archive;
it does not refresh the scientific contract or rebuild maps.

Figure 5 independently normalizes Harmonic Mean within each map, then takes
the median for voxels present in at least 20 of each model's 80 maps.
Absent voxels are missing, not zero. Coordinates are voxel centers; marker
size/opacity increase with median episode support. Reconstruction reproduces
the paper's stable-voxel and top-decile counts for all four models.

## Equality checks

The full command compares key ROC-AUC/AP/coverage metrics with 1e-10 tolerance.
RQ5 additionally compares its exact selection and map metadata hashes, and uses
1e-12 numerical tolerance. Stats compares regenerated exact tests against the
released exact reviews.

Timings, fitted-artifact byte sizes, paths, Git revisions and timestamps can
differ across machines. Figure binaries depend on fonts/renderers.
One historical supplementary n95 entry lies on a binary-rounding boundary:
31.049999999999997 may parse as 31.05, yielding display 31.1 rather than 31.0.
Unrounded values agree to machine precision; ROC-AUC and ranks are unaffected.
