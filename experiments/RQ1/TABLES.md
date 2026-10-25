# RQ1 table generation

Run the reporting script after the RQ1 aggregate results have been generated:

```bash
python experiments/RQ1/generate_tables.py
```

By default, the script reads `results/phase_a_primary/aggregate` and writes the
publication tables to `results/phase_a_primary/tables`. Use `--aggregate-dir`
and `--output-dir` to select another completed RQ1 run.

The generated artifacts comprise:

- one CSV and one LaTeX table comparing all 34 techniques across the four VLA
  models and overall;
- one CSV and one LaTeX inferential table containing the union of techniques
  ranked in the top five for at least one VLA; and
- one detailed CSV and one detailed LaTeX table per VLA, including ROC-AUC,
  average precision, ranks, and variation across trajectory-generation
  conditions.

The script validates the source schemas and experiment dimensions, reconciles
the VLA summaries against the primary blocks, and recomputes tie-aware ranks.
The generated manifest records source hashes and reporting choices.
