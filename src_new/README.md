# VLA-SBFL implementation

This directory contains the model-agnostic replacement for the experiment-specific
scripts in `src/`. Training, scoring, and held-out evaluation are separate operations.

## Formula catalogues

- `table4`: the 31 similarity-coefficient techniques in Table 4 of Wong et al.,
  *A Survey on Software Fault Localization* (2016).
- `study`: the 31 Table 4 techniques plus Tarantula, Ochiai, and Ochiai2 used by
  the RQ1--RQ3 experiments (34 unique techniques).
- `legacy14`: the 14 techniques used by the original repository.
- `all`: 42 unique techniques. Ample, Jaccard, and Zoltar occur in both catalogues;
  Kulczynski and Kulczynski2 are different formulas.

The implementation uses `ef=N_CF`, `nf=N_UF`, `ep=N_CS`, and `np=N_US`.
Every formula returns finite values and diagnostics. Undefined `0/0` values receive
the documented neutral value; true infinities receive finite rank-preserving
sentinels. Unseen test voxels are not confused with undefined formula values.
Single-class training samples are accepted for small-sample experiments and recorded
with `training_has_both_classes=false`; formulas that need the absent class use the
same explicit neutral policy, so these artifacts are finite but underdetermined.

## Install

From the repository root:

```bash
python -m pip install -e src_new
```

## Train

The input can be a single `run_N` directory, an algorithm directory containing runs,
or a model directory. With a split manifest, select one seed and the `train` rows:

```bash
python src_new/train.py \
  --input input/Pi0/DH_200scenes \
  --manifest output/RQ1/pi0_200/step03_grouped_splits/split_manifest.csv \
  --seed 11 \
  --algorithm DH_200scenes \
  --artifact output_new/pi0_dh_200_seed11/model \
  --formula-set table4 \
  --voxel-size 0.02 \
  --origin -1 -1 0 \
  --coordinate-frame robot_base
```

If no manifest is supplied, every episode discovered below `--input` is training data.
The artifact contains `metadata.json`, `formula_catalog.csv`,
`formula_edge_cases.csv.gz`, `training_episode_ids.csv.gz`, and
`voxel_scores.csv.gz`, including hashes and all contingency counts. The sparse edge-case
table identifies the exact voxels whose raw formula result was undefined or infinite.

## Score without labels

```bash
python src_new/score.py \
  --artifact output_new/pi0_dh_200_seed11/model \
  --input path/to/new/runs \
  --output output_new/predictions.csv.gz
```

The output includes one percentile-based continuous risk score per formula and the
known/unknown voxel fraction for every trajectory. Partial unknown coverage is ignored
by the maximum aggregator and reported. A completely unknown trajectory receives the
neutral score `0.5` and status `no_training_support`.

## Held-out evaluation

```bash
python src_new/evaluate.py \
  --artifact output_new/pi0_dh_200_seed11/model \
  --input input/Pi0/DH_200scenes \
  --manifest output/RQ1/pi0_200/step03_grouped_splits/split_manifest.csv \
  --seed 11 \
  --algorithm DH_200scenes \
  --output-dir output_new/pi0_dh_200_seed11/evaluation
```

`evaluate.py` uses pass/fail labels only after scores are frozen. It writes predictions,
ROC-AUC, average precision, prevalence, and coverage diagnostics. It rejects any
evaluation/train episode overlap unless explicitly overridden.

## Tests

```bash
PYTHONPATH=src_new python -m unittest discover -s src_new/tests -v
```
