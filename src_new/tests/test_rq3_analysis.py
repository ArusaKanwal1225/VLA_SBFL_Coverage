import unittest

import numpy as np
import pandas as pd

from experiments.RQ3.analysis import (
    coverage_statistics,
    matched_coverage_bootstrap,
    matched_high_low_effects,
    select_coverage_quantiles,
)
from experiments.RQ3.run_experiment import (
    balanced_sample_ids,
    choose_calibration_run,
    stable_seed,
)


class RQ3AnalysisTests(unittest.TestCase):
    def test_coverage_statistics_reports_unique_weighted_and_class_values(self):
        visits = {
            "pass": frozenset({(0, 0, 0), (1, 0, 0)}),
            "fail": frozenset({(1, 0, 0), (2, 0, 0), (3, 0, 0)}),
        }
        result = coverage_statistics(
            {(0, 0, 0), (1, 0, 0)}, visits, {"pass": 0, "fail": 1})
        self.assertAlmostEqual(result.unique_fraction, 0.5)
        self.assertAlmostEqual(result.mean_known_fraction, (1.0 + 1.0 / 3.0) / 2.0)
        self.assertAlmostEqual(result.pass_known_fraction, 1.0)
        self.assertAlmostEqual(result.failure_known_fraction, 1.0 / 3.0)
        self.assertEqual(result.all_unknown_trajectories, 0)

    def test_quantile_selection_is_distinct_and_near_requested_targets(self):
        candidates = pd.DataFrame({
            "candidate_index": np.arange(10),
            "subset_sha256": ["subset-" + str(index) for index in range(10)],
            "calibration_unique_coverage": np.linspace(0.1, 1.0, 10),
        })
        selected = select_coverage_quantiles(
            candidates, {"low": 0.1, "medium": 0.5, "high": 0.9},
            metric="calibration_unique_coverage")
        self.assertEqual(set(selected["coverage_tier"]), {"low", "medium", "high"})
        self.assertEqual(selected["subset_sha256"].nunique(), 3)
        ordered = selected.set_index("coverage_tier")["calibration_unique_coverage"]
        self.assertLess(ordered["low"], ordered["medium"])
        self.assertLess(ordered["medium"], ordered["high"])

    def test_balanced_sampling_is_reproducible_and_exact(self):
        episodes = pd.DataFrame({
            "episode_id": ["episode-" + str(index) for index in range(20)],
            "failure": [0] * 10 + [1] * 10,
        })
        first = balanced_sample_ids(episodes, 10, np.random.default_rng(17))
        second = balanced_sample_ids(episodes, 10, np.random.default_rng(17))
        self.assertEqual(first, second)
        selected = episodes[episodes["episode_id"].isin(first)]
        self.assertEqual(int(selected["failure"].sum()), 5)
        self.assertEqual(len(selected), 10)

    def test_seed_and_calibration_rotation_are_deterministic(self):
        runs = tuple("run_" + str(index) for index in range(1, 11))
        self.assertEqual(choose_calibration_run("run_1", runs), "run_2")
        self.assertEqual(choose_calibration_run("run_10", runs), "run_1")
        self.assertEqual(stable_seed(7, "x", 1), stable_seed(7, "x", 1))
        self.assertNotEqual(stable_seed(7, "x", 1), stable_seed(7, "x", 2))

    def test_matched_high_low_effects_preserve_pairing(self):
        rows = []
        for tier, coverage, auc in (("low", 0.2, 0.6), ("high", 0.8, 0.72)):
            rows.append({
                "voxel_size": 0.02,
                "configuration_id": "EO1__DH__200",
                "vla": "EO1",
                "condition": "DH",
                "budget": 200,
                "test_run": "run_1",
                "training_size": 10,
                "selection_replicate": 1,
                "formula_id": "harmonic_mean",
                "technique": "Harmonic Mean",
                "coverage_tier": tier,
                "roc_auc": auc,
                "average_precision": auc + 0.05,
                "calibration_mean_known_fraction": coverage,
                "test_mean_known_fraction": coverage - 0.05,
                "test_unique_coverage": coverage - 0.10,
            })
        effects = matched_high_low_effects(pd.DataFrame(rows))
        self.assertEqual(len(effects), 1)
        self.assertAlmostEqual(float(effects.iloc[0]["delta_roc_auc"]), 0.12)
        self.assertAlmostEqual(
            float(effects.iloc[0]["delta_calibration_mean_known_fraction"]), 0.6)

    def test_matched_coverage_bootstrap_preserves_constant_effects(self):
        values = np.zeros((3, 2, 4, 2, 3), dtype=np.float64)
        values[..., 0, 0] = 0.01
        values[..., 0, 1] = 0.03
        values[..., 0, 2] = -0.02
        values[..., 1, 0] = 0.02
        values[..., 1, 1] = 0.04
        values[..., 1, 2] = -0.01
        result = matched_coverage_bootstrap(
            values,
            reference_technique_index=0,
            iterations=100,
            seed=17,
        )
        np.testing.assert_allclose(
            result.effect_lower,
            np.asarray([[0.01, 0.03, -0.02], [0.02, 0.04, -0.01]]),
        )
        np.testing.assert_allclose(result.effect_lower, result.effect_upper)
        np.testing.assert_allclose(
            result.overall_effect_lower,
            np.asarray([0.015, 0.035, -0.015]),
        )
        np.testing.assert_allclose(
            result.overall_interaction_lower,
            np.asarray([0.0, 0.02, -0.03]),
        )
        np.testing.assert_allclose(
            result.overall_interaction_lower,
            result.overall_interaction_upper,
        )


if __name__ == "__main__":
    unittest.main()
