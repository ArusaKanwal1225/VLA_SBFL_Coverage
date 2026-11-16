# RQ1: comparing 34 SBFL techniques

The frozen design is in RQ1.md. The portable commands, from the repository root,
are:

~~~bash
python replicate.py tables --rq rq1 --output reproduced/rq1-tables
python replicate.py run --rq rq1 --output reproduced/rq1-full
~~~

The data come from the Zenodo archive. No model or simulator is loaded.
Historical audits/aggregates are in data/reference/rq1.
The analysis uses leave-one-run-out folds and evaluates trajectory-level failure
discrimination. Direct causal voxel localization is outside the available labels.
See the root README and docs/REPLICATION.md for setup and interpretation.
