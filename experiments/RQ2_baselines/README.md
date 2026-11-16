# RQ2: baselines and the top-five RQ1 techniques

From the repository root, after installing the Zenodo data:

~~~bash
python replicate.py tables --rq rq2 --output reproduced/rq2-tables
python replicate.py run --rq rq2 --output reproduced/rq2-full
~~~

This recomputes the frozen 13-candidate catalogue from the original metric JSONs.
The complete-case availability rule is preserved; unavailable metrics are not
imputed and episodes are not removed. All 320 folds match the RQ1 test runs.

When run independently, the baseline analysis uses the released historical RQ1
fold metrics. When rq1 and rq2 are requested together, it uses the newly
recomputed RQ1 folds. The final table adds the descriptive top-five techniques
from RQ1 with their original names and tie policy.

The original RQ2.md is the frozen protocol. Historical results are under
data/reference/rq2. Original EXECUTION.md/TABLES.md files describe their historical
execution paths; use the root entry point for portable reproduction.

Use python replicate.py stats --rq rq2 to regenerate exact tests from the released
aggregates. Holm covers the eight baseline comparisons against Harmonic Mean.
