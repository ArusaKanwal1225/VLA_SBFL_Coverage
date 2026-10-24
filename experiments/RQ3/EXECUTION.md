# RQ3 execution record

**Status:** full primary experiment completed successfully on 2026-10-23.

The smoke design used `EO1/DH/200`, outer test `run_1`, calibration `run_2`,
training size 10, 20 random candidate subsets, and the three coverage tiers.
All 31 formulas produced finite metrics in all three selected cells.

Calibration unique coverage increased from `0.076` to `0.086` to `0.096` for
the low, medium, and high subsets. Untouched test unique coverage followed the
same ordering (`0.081`, `0.083`, `0.094`), confirming that the manipulation
works end to end in this diagnostic fold. These values are execution checks,
not RQ3 evidence.

## Full primary execution

The complete primary experiment finished in approximately 12 minutes 30
seconds and contains:

- 32 VLA/condition/budget configurations and ten outer test folds.
- Five training sizes and three coverage tiers.
- 4,800 fitted subset cells and 148,800 formula measurements.
- All 31 Table 4 techniques with finite critical metrics.
- Exact 50/50 pass/fail balance in every training subset.

Every high-coverage subset had greater calibration unique coverage than its
matched low-coverage subset. The mean calibration difference was `0.0256`, and
the mean held-out test difference was `0.0190`. Test unique coverage was higher
for the high tier in 96.9% of matched cells.

Dennis was the descriptive ROC-AUC leader at every coverage tier: `0.5995` at
low coverage, `0.5987` at medium coverage, and `0.6008` at high coverage. The
frozen technique-by-coverage analysis found no practically meaningful,
Holm-significant interaction against Harmonic Mean.

The high-minus-low ROC-AUC changes were small: `+0.0013` for Dennis, `-0.0004`
for Harmonic Mean, `-0.0005` for Rogot2, and `-0.0014` for Tarwid when averaged
over training sizes and primary blocks. The dominant descriptive improvement
came from increasing the number of training trajectories, not from moving
between coverage tiers at a fixed training size.
