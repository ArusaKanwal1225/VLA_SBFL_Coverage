# RQ3 statistical analysis and table generation

Generate all main-text and supplementary RQ3 tables from the completed
coverage-controlled experiment:

```bash
PYTHONPATH="$PWD:$PWD/src_new" python experiments/RQ3/generate_tables.py
```

The default output directory is
`experiments/RQ3/results/phase_a_coverage/tables`. It contains CSV and LaTeX
versions of:

- the coverage-manipulation check stratified by VLA and training-set size;
- Harmonic Mean performance across coverage tiers for every VLA and size;
- competitive technique sensitivity stratified by VLA;
- the global matched statistical analysis;
- complete 31-technique rankings for each VLA;
- the complete technique-by-coverage interaction analysis; and
- coverage, class-conditional familiarity, unknown-trajectory, voxel-count,
  and tie diagnostics.

The reporting analysis adds the direct H3.1 test omitted from the original
aggregate summary. The overall Harmonic Mean high-minus-low contrast is tested
over 16 matched VLA-condition blocks. Five training-size-specific tests receive
Holm correction. Confidence intervals use 10,000 matched hierarchical
bootstrap draws by default, resampling blocks and then runs within each
equal-weight scene budget.

The generator validates the complete 4,800-cell design, exact pass/fail balance,
all matched coverage pairs, the 31-formula grids, and reconciliation with the
existing rankings and technique-interaction results. The manifest records all
source hashes and reporting choices.
