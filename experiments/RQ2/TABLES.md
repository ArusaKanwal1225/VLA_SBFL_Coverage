# RQ2 statistical analysis and table generation

Recompute the frozen matched statistical analysis from the completed fold
artifacts:

```bash
PYTHONPATH="$PWD:$PWD/src_new" python experiments/RQ2/run_experiment.py \
  --analysis-only
```

Generate the VLA-stratified publication tables:

```bash
python experiments/RQ2/generate_tables.py
```

The default output directory is
`experiments/RQ2/results/phase_a_granularity/tables`. It contains CSV and
LaTeX versions of:

- Harmonic Mean voxel-size sensitivity for each VLA and the macro result;
- four-panel competitive-technique results stratified by VLA;
- coverage and computational diagnostics for every VLA-size combination;
- the global Friedman size test, meaningful technique-size interactions, and
  ranking-stability correlations; and
- four complete 31-technique supplementary tables, one per VLA.

The generator validates all experiment dimensions, reconciles VLA aggregates
against the published macro results, and verifies that per-VLA diagnostics
reconcile to the existing global diagnostics. The output manifest records the
source hashes and reporting choices.
