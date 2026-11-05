"""Opt-in bootstrap audit from frozen aggregate CSVs; no experiment execution.

Run with PYTHONPATH=.:src_new python -B -m experiments.bootstrap_review.
Historical hierarchical implementations are reused without changing their RNG.
The sensitivity resamples whole paired blocks and reports pointwise percentile
intervals, not simultaneous intervals or a correction for leader selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.RQ1.analysis import hierarchical_budget_bootstrap
from experiments.RQ2.analysis import matched_size_bootstrap
from experiments.RQ2_baselines.analysis import paired_hierarchical_bootstrap
from vla_sbfl.formulas import STUDY_FORMULA_IDS

ROOT = Path(__file__).resolve().parents[1]


def block_draws(values, iterations, seed):
    """Resample the first axis jointly for every trailing outcome dimension."""
    values = np.asarray(values, dtype=float)
    if values.ndim < 2 or values.shape[0] < 2 or not np.isfinite(values).all():
        raise ValueError("finite paired block outcomes are required")
    if iterations < 100:
        raise ValueError("at least 100 draws are required")
    rng = np.random.default_rng(seed)
    return np.asarray([values[rng.integers(0, len(values), len(values))].mean(axis=0)
                       for _ in range(iterations)])


def tensor(frame, columns, ids):
    frame = frame.copy()
    frame["primary_block_id"] = frame.vla.astype(str) + "__" + frame.condition.astype(str)
    blocks = sorted(frame.primary_block_id.unique())
    index = pd.MultiIndex.from_product([blocks, [200, 400],
        [f"run_{i}" for i in range(1, 11)]],
        names=["primary_block_id", "budget", "test_run"])
    pivot = frame.pivot(index=index.names, columns=columns, values="roc_auc").reindex(
        index=index, columns=ids)
    if len(blocks) != 16 or pivot.isna().any().any():
        raise ValueError("complete 16 x 2 x 10 fold grid is required")
    return pivot.to_numpy().reshape(16, 2, 10, len(ids))


def audit(rq):
    directories = {"RQ1": "RQ1/results/phase_a_primary",
                   "RQ2": "RQ2_baselines/results/all_models",
                   "RQ3": "RQ2/results/phase_a_granularity"}
    aggregate = ROOT / "experiments" / directories[rq] / "aggregate"
    sources, rows = {}, []

    def read(name, json_file=False):
        path = aggregate / name
        sources[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text()) if json_file else pd.read_csv(path)

    def record(identity, historical, reproduced, draws, seed, iterations):
        low, high = np.percentile(draws, [2.5, 97.5])
        error = float(np.max(np.abs(np.asarray(historical) - reproduced)))
        if error > 1e-12:
            raise ValueError(f"hierarchical reproduction failed: {identity}, error={error}")
        rows.append(dict(identity=identity, iterations=iterations, seed=seed,
            historical_lower=historical[0], historical_upper=historical[1],
            reproduced_lower=reproduced[0], reproduced_upper=reproduced[1],
            block_lower=low, block_upper=high,
            lower_change=low-historical[0], upper_change=high-historical[1],
            historical_width=historical[1]-historical[0], block_width=high-low,
            width_change=(high-low)-(historical[1]-historical[0]),
            zero_in_historical=historical[0] <= 0 <= historical[1],
            zero_in_block=low <= 0 <= high, reproduction_error=error))

    def verify_blocks(values, columns, ids):
        primary = read("primary_block_metrics.csv")
        pivot = primary.pivot(index="primary_block_id", columns=columns,
                              values="roc_auc").sort_index().reindex(columns=ids)
        expected = values.mean(axis=(1, 2)).reshape(16, -1)
        np.testing.assert_allclose(pivot.to_numpy(), expected, rtol=0, atol=1e-14)

    if rq == "RQ1":
        config = read("friedman_test.json", True)["bootstrap"]
        iterations, seed = config["iterations"], config["seed"]
        ids = list(STUDY_FORMULA_IDS)
        values = tensor(read("fold_metrics.csv.gz"), "formula_id", ids)
        verify_blocks(values, "formula_id", ids)
        history = read("statistical_comparison.csv").set_index("formula_id")
        ref = ids.index("harmonic_mean")
        result = hierarchical_budget_bootstrap(values, ref, iterations, seed)
        draws = block_draws(values.mean(axis=(1, 2)), iterations, seed)
        for i, formula in enumerate(ids):
            row = history.loc[formula]
            record("auc/"+formula, (row.roc_auc_ci_lower, row.roc_auc_ci_upper),
                   (result[0][i], result[1][i]), draws[:, i], seed, iterations)
            record("reference_minus_candidate/"+formula,
                   (row.delta_vs_leader_ci_lower, row.delta_vs_leader_ci_upper),
                   (result[2][i], result[3][i]), draws[:, ref]-draws[:, i], seed, iterations)
    elif rq == "RQ2":
        config = read("statistical_tests.json", True)
        iterations, seed = config["bootstrap_iterations"], config["bootstrap_seed"]
        primary = read("primary_block_metrics.csv")
        ids = sorted(primary.method_id.unique())
        values = tensor(read("fold_metrics.csv.gz"), "method_id", ids)
        verify_blocks(values, "method_id", ids)
        history = read("statistical_comparison.csv").set_index("method_id")
        ref = ids.index("harmonic_mean")
        for i, method in enumerate(ids):
            if i == ref:
                continue
            pair_seed = seed+i  # Original enumerate includes the skipped reference.
            result = paired_hierarchical_bootstrap(values[..., ref], values[..., i], iterations, pair_seed)
            draws = block_draws(values[..., [ref, i]].mean(axis=(1, 2)), iterations, pair_seed)
            row = history.loc[method]
            record("reference_auc/"+method, (row.reference_ci_low, row.reference_ci_high),
                   result[:2], draws[:, 0], pair_seed, iterations)
            record("candidate_minus_reference/"+method, (row.gap_ci_low, row.gap_ci_high),
                   result[2:], draws[:, 1]-draws[:, 0], pair_seed, iterations)
    else:
        config = read("statistical_tests.json", True)["bootstrap"]
        iterations, seed = config["iterations"], config["seed"]
        sizes, ids = [0.01, 0.02, 0.05], list(STUDY_FORMULA_IDS)
        columns = pd.MultiIndex.from_product([sizes, ids], names=["voxel_size", "formula_id"])
        values = tensor(read("fold_metrics_by_size.csv.gz"), ["voxel_size", "formula_id"], columns)
        values = values.reshape(16, 2, 10, 3, len(ids))
        verify_blocks(values, ["voxel_size", "formula_id"], columns)
        ref = ids.index("harmonic_mean")
        result = matched_size_bootstrap(values, 1, ref, iterations, seed)
        draws = block_draws(values.mean(axis=(1, 2)), iterations, seed)
        deltas = draws-draws[:, [1], :]
        interactions = deltas-deltas[:, :, [ref]]
        scores = read("rankings_by_size.csv").set_index(["voxel_size", "formula_id"])
        effects = read("technique_size_effects.csv").set_index(["voxel_size", "formula_id"])
        effects_i = read("technique_size_interactions.csv").set_index(["voxel_size", "formula_id"])
        for s, size in enumerate(sizes):
            for i, formula in enumerate(ids):
                row = scores.loc[(size, formula)]
                identity = f"{size}/{formula}"
                record("auc/"+identity, (row.roc_auc_ci_lower, row.roc_auc_ci_upper),
                       (result.score_lower[s, i], result.score_upper[s, i]), draws[:, s, i], seed, iterations)
                if s == 1:
                    continue
                row = effects.loc[(size, formula)]
                record("size_minus_primary/"+identity, (row.delta_ci_lower, row.delta_ci_upper),
                       (result.delta_lower[s, i], result.delta_upper[s, i]), deltas[:, s, i], seed, iterations)
                if i == ref:
                    continue
                row = effects_i.loc[(size, formula)]
                record("interaction/"+identity, (row.interaction_ci_lower, row.interaction_ci_upper),
                       (result.interaction_lower[s, i], result.interaction_upper[s, i]),
                       interactions[:, s, i], seed, iterations)
    output = aggregate / "bootstrap_block_sensitivity.csv"
    frame = pd.DataFrame(rows)
    frame.to_csv(output, index=False)
    manifest = dict(rq=rq, intervals=len(rows), iterations=iterations,
        method="paired complete-block percentile bootstrap", blocks=16,
        confidence_level=.95, simultaneous=False, holm_adjusted=False,
        maximum_reproduction_error=float(frame.reproduction_error.max()),
        zero_inclusion_changes=int((frame.zero_in_historical != frame.zero_in_block).sum()),
        source_sha256=sources, output_sha256={output.name: hashlib.sha256(output.read_bytes()).hexdigest()},
        numpy_version=np.__version__, pandas_version=pd.__version__)
    (aggregate / "bootstrap_block_sensitivity_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)+"\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rq", choices=["RQ1", "RQ2", "RQ3"], action="append")
    args = parser.parse_args()
    for rq in args.rq or ["RQ1", "RQ2", "RQ3"]:
        print(json.dumps(audit(rq), indent=2))


if __name__ == "__main__":
    main()
