# Paper RQ4: training-size and coverage learning curves

The directory retains its historical name RQ3_v2. In the final paper this is RQ4.

~~~bash
python replicate.py tables --rq rq4 --output reproduced/rq4-tables
python replicate.py stats --rq rq4 --output reproduced/rq4-statistics
python replicate.py run --rq rq4 --output reproduced/rq4-full
~~~

The design uses nested balanced suites of 10/20/50/100/200 trajectories,
ten repetitions and 320 outer folds: 16,000 cells and 544,000 technique metrics.
The next cyclic run quantifies calibration coverage without selecting suites.
The outer test run remains untouched. Separate exact Holm families contain
4 adjacent-size, 33 sample-efficiency and 34 conditional-coverage tests.

Historical inputs are installed under data/reference/rq4. The original protocol
is retained in RQ3_v2.md for provenance. Use the root instructions for execution.
