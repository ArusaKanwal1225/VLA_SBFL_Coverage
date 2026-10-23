# Preliminary experiments

This directory keeps exploratory experiments separate from the confirmatory
experiments used to answer the research questions.

## Contents

- `training_size_quick_test/`: the initial Pi0/DH experiment comparing models
  trained with 1, 5, 50, 100, and 200 trajectories on the same 100 held-out
  trajectories. It contains the executable script and its frozen results.
- `RQ1_PROTOCOL.md`: proposed protocol for comparing the 31 spectrum-based
  techniques. Its status is **draft; do not launch yet**.

Reproduce the training-size experiment from the repository root with:

```bash
PYTHONPATH=src_new python preliminary_experiments/training_size_quick_test/run.py
```

Exploratory results must not be used to tune the confirmatory RQ1 protocol after
the RQ1 test partitions have been inspected.
