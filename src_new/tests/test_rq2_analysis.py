import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from experiments.RQ2.analysis import (
    matched_size_bootstrap,
    spearman_rank_correlation,
)
from experiments.RQ2.run_experiment import (
    REFERENCE_FORMULA,
    RQ1_ANALYSIS,
    RQ1_PROTOCOL,
    RQ1_RUNNER,
    VOXEL_SIZES,
    aggregate_cross_size,
    build_cell_command,
    sha256_file,
    size_slug,
    validate_matched_fold_identities,
)
from vla_sbfl.formulas import FORMULA_REGISTRY, STUDY_FORMULA_IDS


class RQ2AnalysisTests(unittest.TestCase):
    def test_matched_bootstrap_preserves_constant_size_and_interaction_deltas(self):
        values = np.empty((3, 2, 4, 3, 2), dtype=float)
        for block in range(3):
            for budget in range(2):
                for fold in range(4):
                    baseline = 0.55 + 0.01 * block + 0.005 * budget + 0.001 * fold
                    values[block, budget, fold, :, 0] = [
                        baseline - 0.02,
                        baseline,
                        baseline + 0.03,
                    ]
                    values[block, budget, fold, :, 1] = [
                        baseline - 0.11,
                        baseline - 0.10,
                        baseline - 0.05,
                    ]
        result = matched_size_bootstrap(
            values,
            primary_size_index=1,
            reference_technique_index=0,
            iterations=200,
            seed=11,
        )
        self.assertEqual(result.score_lower.shape, (3, 2))
        np.testing.assert_allclose(result.delta_lower[1], 0.0)
        np.testing.assert_allclose(result.delta_upper[1], 0.0)
        np.testing.assert_allclose(result.delta_lower[0], [-0.02, -0.01])
        np.testing.assert_allclose(result.delta_upper[2], [0.03, 0.05])
        np.testing.assert_allclose(result.interaction_lower[:, 0], 0.0)
        np.testing.assert_allclose(result.interaction_upper[:, 0], 0.0)
        np.testing.assert_allclose(result.interaction_lower[0, 1], 0.01)
        np.testing.assert_allclose(result.interaction_upper[2, 1], 0.02)

    def test_spearman_rank_correlation_is_tie_aware(self):
        self.assertAlmostEqual(
            spearman_rank_correlation([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertAlmostEqual(
            spearman_rank_correlation([1, 2, 3, 4], [40, 30, 20, 10]), -1.0)
        tied = spearman_rank_correlation([1, 1, 3, 4], [2, 2, 5, 8])
        self.assertAlmostEqual(tied, 1.0)

    def test_fold_identity_validation_rejects_changed_outcome(self):
        rows = pd.DataFrame([{
            "configuration_id": "EO1__DH__200",
            "vla": "EO1",
            "condition": "DH",
            "budget": 200,
            "test_run": "run_1",
            "formula_id": "harmonic_mean",
            "technique": "Harmonic Mean",
            "episodes": 200,
            "failures": 100,
            "failure_prevalence": 0.5,
        }])
        validate_matched_fold_identities({
            0.01: rows.copy(),
            0.02: rows.copy(),
            0.05: rows.copy(),
        })
        changed = rows.copy()
        changed.loc[0, "failures"] = 101
        with self.assertRaises(ValueError):
            validate_matched_fold_identities({
                0.01: changed,
                0.02: rows.copy(),
                0.05: rows.copy(),
            })

    def test_command_builder_and_size_slug(self):
        self.assertEqual(size_slug(0.01), "size_0p01")
        command = build_cell_command(
            input_root=Path("input"),
            output_root=Path("output"),
            voxel_size=0.05,
            origin=(-1.0, -1.0, 0.0),
            chunk_size=100,
            bootstrap_iterations=200,
            bootstrap_seed=7,
            resume=True,
            smoke=True,
        )
        self.assertIn("--voxel-size", command)
        self.assertIn("0.05", command)
        self.assertIn("--resume", command)
        self.assertIn("--max-folds", command)
        self.assertIn("EO1__DH__200", command)

    def test_full_cross_size_aggregation_contract(self):
        vlas = ("EO1", "Groot", "Pi0", "SpatialVLA")
        conditions = ("DH", "DH_HybridDT", "DH_HybridRF", "DH_WeightedRF")
        budgets = (200, 400)
        runs = tuple("run_" + str(index) for index in range(1, 11))
        formula_ids = list(STUDY_FORMULA_IDS)
        reference_index = formula_ids.index(REFERENCE_FORMULA)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sources = {}
            for voxel_size in VOXEL_SIZES:
                source = root / size_slug(voxel_size)
                sources[voxel_size] = source
                (source / "aggregate").mkdir(parents=True)
                run = {
                    "status": "complete",
                    "experiment_signature": "synthetic-" + str(voxel_size),
                    "signature_details": {
                        "protocol_sha256": sha256_file(RQ1_PROTOCOL),
                        "runner_sha256": sha256_file(RQ1_RUNNER),
                        "analysis_sha256": sha256_file(RQ1_ANALYSIS),
                        "grid": {
                            "voxel_size": voxel_size,
                            "origin": [-1.0, -1.0, 0.0],
                            "coordinate_frame": "robot_base",
                        },
                    },
                }
                (source / "run.json").write_text(json.dumps(run), encoding="utf-8")
                rows = []
                size_effect = {0.01: -0.01, 0.02: 0.0, 0.05: -0.02}[voxel_size]
                for vla_index, vla in enumerate(vlas):
                    for condition_index, condition in enumerate(conditions):
                        block_effect = 0.002 * vla_index + 0.001 * condition_index
                        for budget in budgets:
                            configuration_id = "{}__{}__{}".format(
                                vla, condition, budget)
                            for run_index, run_id in enumerate(runs):
                                fold_effect = 0.0001 * run_index
                                for formula_index, formula_id in enumerate(formula_ids):
                                    technique_effect = -0.001 * abs(
                                        formula_index - reference_index)
                                    auc = (
                                        0.72 + block_effect + fold_effect
                                        + size_effect + technique_effect)
                                    rows.append({
                                        "configuration_id": configuration_id,
                                        "vla": vla,
                                        "condition": condition,
                                        "budget": budget,
                                        "test_run": run_id,
                                        "formula_id": formula_id,
                                        "technique": FORMULA_REGISTRY[formula_id].name,
                                        "roc_auc": auc,
                                        "average_precision": auc + 0.05,
                                        "episodes": 100,
                                        "failures": 50,
                                        "failure_prevalence": 0.5,
                                        "mean_known_fraction": 0.98,
                                        "all_unknown_trajectories": 0,
                                        "score_tie_fraction": 0.4,
                                        "fit_seconds": 0.1,
                                        "score_seconds": 0.2,
                                        "model_artifact_bytes": 1000,
                                    })
                pd.DataFrame(rows).to_csv(
                    source / "aggregate" / "fold_metrics.csv.gz",
                    index=False,
                    compression="gzip",
                )

            def synthetic_voxel_counts(_source, voxel_size):
                rows = []
                for vla in vlas:
                    for condition in conditions:
                        for budget in budgets:
                            configuration_id = "{}__{}__{}".format(
                                vla, condition, budget)
                            for run_id in runs:
                                rows.append({
                                    "voxel_size": voxel_size,
                                    "configuration_id": configuration_id,
                                    "test_run": run_id,
                                    "training_voxels": int(100 / voxel_size),
                                })
                return pd.DataFrame(rows)

            output = root / "output"
            with patch(
                "experiments.RQ2.run_experiment.load_training_voxel_counts",
                side_effect=synthetic_voxel_counts,
            ):
                summary = aggregate_cross_size(
                    cell_sources=sources,
                    output_root=output,
                    origin=(-1.0, -1.0, 0.0),
                    bootstrap_iterations=100,
                    bootstrap_seed=19,
                )
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(summary["phase_a_size_leader"], 0.02)
            self.assertEqual(summary["meaningful_interactions"], 0)
            self.assertTrue((output / "aggregate" / "RESULTS.md").is_file())
            effects = pd.read_csv(
                output / "aggregate" / "technique_size_effects.csv")
            interactions = pd.read_csv(
                output / "aggregate" / "technique_size_interactions.csv")
            self.assertEqual(len(effects), 68)
            self.assertEqual(len(interactions), 66)


if __name__ == "__main__":
    unittest.main()
