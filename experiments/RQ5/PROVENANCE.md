# Provenance of the RQ5 suspiciousness maps

The original historical RQ1 map artifacts were not reused. RQ5 reconstructed
maps from the then-current SBFL Pick-Up inputs using the frozen RQ1 training,
voxelization and scoring protocol. Reconstruction used Pick-Up trajectories
only. Move Near trajectories were subsequently scored without retraining or
target-task calibration.

The exact reconstructed artifacts used in the reported analysis are published
under data/rq5/reconstructed_pickup_maps. Their original metadata, training
episode-ID lists, input fingerprints, score files and file hashes are retained.

## Historical audit discrepancies

Every map carries training_source=current_sbfl_pickup_input,
current_input_sha256 and historical_audit_comparison in its training metadata.
The comparison uses episode IDs, run/evaluation IDs, outcomes, objects,
trajectory step counts and trajectory hashes. A discrepancy therefore need
not mean that a pass/fail outcome changed.

| VLA | Different configuration-episode records |
| --- | ---: |
| EO1 | 13 |
| GR00T | 10 |
| Pi0 | 21 |
| SpatialVLA | 17 |
| Total | 61 |

Counts are aggregated once per configuration, not once for each of its ten
folds. Configurations and 200/400 budgets can refer to overlapping physical
executions, so these counts must not be interpreted as 61 distinct executions.
The detailed configuration counts, example IDs and input fingerprints are in
data/rq5/map_provenance.json and the unchanged per-fold metadata.

The frozen RQ1 code/protocol signature is:
910e24f9b9d91fdd8863ed3353c12dd082a81de824292f140fc39e5435eeda71

RQ1 was also recalculated with the current inputs. Its scientific CSV/LaTeX
tables matched the historical tables; manifests and the separate comparison
report are not scientific tables and need not be byte-identical.
Matching aggregate tables does not establish identity of historical map files.

## Interpretation for the paper

Suggested wording:

> RQ5 used suspiciousness maps reconstructed from the current Pick-Up dataset
> using the frozen RQ1 training protocol. These reconstructed maps were then
> applied to Move Near trajectories without retraining or target-task
> calibration. Differences from the historical RQ1 episode audit were recorded,
> and the exact reconstructed maps used in the analysis are released.

Do not describe these artifacts as the exact original historical RQ1 maps.

## Reproduction boundary

The published bundle is sufficient to verify the released map files and
reproduce the Move Near scoring, selected sample, baseline analysis and tables.
It does not include an additional copy of the raw Pick-Up training inputs.
Reconstructing maps from another Pick-Up snapshot is a separate operation and
may produce a different fingerprint. Use the published maps for exact RQ5
verification rather than running prepare first.

Each new evaluation writes map_provenance.json alongside analysis_config.json
and the per-fold map metadata hashes, making the origin of the scored maps
explicit in future runs as well.
