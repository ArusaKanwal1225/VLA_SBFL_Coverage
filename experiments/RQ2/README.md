# Paper RQ3: voxel-size sensitivity

This historical code directory is named RQ2; the final paper calls this RQ3.
Use the root replication entry point after installing the Zenodo data:

~~~bash
python replicate.py tables --rq rq3 --output reproduced/rq3-tables
python replicate.py stats --rq rq3 --output reproduced/rq3-statistics
python replicate.py run --rq rq3 --output reproduced/rq3-full
~~~

The full run adds RQ1 to obtain the required 0.02 m models, then evaluates the
matched 0.01/0.05 m folds. Table-only generation reads the released 960 voxel
counts, which reconcile with the historical diagnostic table.
Exact Holm families contain 68 size effects and 66 interactions.
The frozen original protocol remains in RQ2.md; final reporting follows the
root README and docs/REPLICATION.md.
