# Reusable VLA-SBFL library

The paper entry point is ../replicate.py; use the repository README for setup,
data extraction, analyses, tables and figures. The original experiment-specific
preprocessing scripts are retained in the external cleanup backup.

The vla_sbfl package separates trajectory loading, binary voxelization,
contingency counts, suspiciousness formulas, normalization, scoring and
ROC-AUC/average precision evaluation.

## Formula catalogues

* study: the 31 Table-4 coefficients plus Tarantula, Ochiai and Ochiai2: 34.
* table4: the 31 Table-4 coefficients.
* legacy14: historical original-repository formulas, retained for compatibility.
* all: 42 unique formulas.

The paper runners explicitly select their frozen catalogue; generic defaults
in train.py do not redefine those experiments.

Undefined values, finite sentinels for true infinities, percentile ties,
unsupported voxels and single-class training are recorded explicitly.
A completely unsupported trajectory receives a neutral 0.5 risk score.
Partial unknown coverage is reported and the known-voxel maximum is retained.

The generic train.py, score.py and evaluate.py interfaces remain available
for independent experiments. For paper replication, use the frozen experiment
runners through replicate.py rather than inventing a new split.

## Tests

From the repository root after installing requirements.txt:

~~~bash
python -m unittest discover -s src_new/tests -v
~~~
