# RQ5: Pick-Up to Move Near transfer

Use the portable entry point from the repository root:

~~~bash
python replicate.py tables --rq rq5 --output reproduced/rq5-tables
python replicate.py run --rq rq5 --output reproduced/rq5-full
~~~

Install the Zenodo archive and Python dependencies using the root README first.
No VLA model, simulator, checkpoint or GPU is needed.

## Frozen maps and inputs

RQ5 uses **reconstructed** Pick-Up maps, not the exact original historical RQ1
map files. The release preserves the exact reconstructed bytes that were scored.
Read [PROVENANCE.md](PROVENANCE.md) for the historical audit discrepancies.
Do not run prepare before reproducing the released RQ5 analysis.

The archive installs the data at data/rq5/ in the repository root:
move_near_504, reconstructed_pickup_maps, scenes_504, reference, and the original
RQ5 files_manifest.csv. Historical RQ1 audits and run metadata are at
data/reference/rq1. The launcher relocates those references explicitly without
disabling their frozen protocol/code signature check.

The full reproduction verifies all RQ5 file hashes, simulator final outcomes,
scene/prompt hashes and cell quotas; regenerates the final seeded selection;
scores the 34 formulas using the published maps; computes baseline scores;
and compares the resulting metrics with the released references.

## Final sample

| VLA | Candidate pass / fail | Selected pass | Selected fail | N |
| --- | --- | ---: | ---: | ---: |
| EO1 | 177 / 327 | 100 | 123 | 223 |
| GR00T | 225 / 279 | 100 | 146 | 246 |
| Pi0 | 109 / 395 | 100 | 198 | 298 |
| SpatialVLA | 151 / 353 | 100 | 242 | 342 |

This is the retrospective fixed-pass selection, seed 20261028, with each model's
N chosen to approximate its pooled RQ1 pass ratio subject to integer counts,
available outcomes and XY-cell quotas. Ratios are not mathematically exact.
Cell allocations differ by at most one. Every method within a model scores
the same selected episodes; selected IDs may differ across models.

The 72 design cells are a 9?8 XY grid with 0.05 m cell width and seven candidates
per cell. They differ from RQ1's frozen 0.02 m 3-D scoring voxels at origin
(-1,-1,0), robot_base coordinates.

The actual task is google_robot_move_near_customizable. Pass means the final
simulator log has success=true. No additional pick-and-place positional
tolerance or stability-period criterion was applied by this analysis.

## Outputs

* sbfl/roc_auc_table.tex: all 34 techniques, Overall ordering and per-column ranks.
* baselines/baseline_top5_sbfl.tex: available baselines and descriptive top-five SBFL.
* baselines/baseline_performance.tex: baselines alone.
* selection/sample_manifest.csv: exact chosen episodes.
* verification_report.json: identity and numeric comparison results.

Overall weights the four VLA models equally. Repeated map scoring is not an
independent target-task replication. Unavailable token/variability metrics are
not imputed. Tables need booktabs.

## Low-level tools

The original analysis entry points remain available; pass paths explicitly:

~~~bash
export VLA_SBFL_RQ1_REFERENCE="$PWD/data/reference/rq1"
python experiments/RQ5/rq5_reproduce.py verify --data data/rq5
python experiments/RQ5/rq5_reproduce.py run --data data/rq5 \
  --output reproduced/rq5-independent
~~~

The output must be empty. To recover from an interrupted RQ5 reproduction,
choose a new output directory. Existing partial files remain available for
inspection. Earlier pilot, simulator and exploratory scripts were archived
outside this evaluation checkout; they are not required for the paper analysis.
