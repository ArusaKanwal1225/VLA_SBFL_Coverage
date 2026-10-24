import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from experiments.RQ3_v2.analysis import (
    fixed_effect_slope,
    nested_balanced_subsets,
    summarize_learning_curve,
)
from experiments.RQ3_v2.run_experiment import (
    adjacent_reference_tests,
    choose_calibration_run,
    conditional_coverage_analysis,
    sample_efficiency_tests,
    stable_seed,
)
from vla_sbfl.formulas import FORMULA_REGISTRY, TABLE4_FORMULA_IDS


class RQ3V2AnalysisTests(unittest.TestCase):
    def test_nested_balanced_subsets_are_reproducible_balanced_and_nested(self):
        episodes = pd.DataFrame({
            "episode_id": ["episode-{:02d}".format(index) for index in range(40)],
            "failure": [0] * 20 + [1] * 20,
        })
        sizes = (4, 10, 20)
        first = nested_balanced_subsets(
            episodes, sizes, np.random.default_rng(31))
        second = nested_balanced_subsets(
            episodes, sizes, np.random.default_rng(31))
        self.assertEqual(first, second)
        for size in sizes:
            selected = episodes[episodes["episode_id"].isin(first[size])]
            self.assertEqual(len(selected), size)
            self.assertEqual(int(selected["failure"].sum()), size // 2)
        self.assertLess(set(first[4]), set(first[10]))
        self.assertLess(set(first[10]), set(first[20]))

    def test_nested_balanced_subsets_reject_insufficient_class_pool(self):
        episodes = pd.DataFrame({
            "episode_id": ["p1", "p2", "f1"],
            "failure": [0, 0, 1],
        })
        with self.assertRaisesRegex(ValueError, "lacks 2 trajectories for class 1"):
            nested_balanced_subsets(episodes, (2, 4), np.random.default_rng(1))

    def test_learning_curve_summary_uses_log_scale_and_reports_target_size(self):
        sizes = (10, 20, 50, 100, 200)
        constant = summarize_learning_curve(sizes, (0.63,) * len(sizes))
        self.assertAlmostEqual(constant.normalized_log_aulc, 0.63)
        self.assertAlmostEqual(constant.gain, 0.0)
        self.assertEqual(constant.size_at_95_percent_improvement, 10)

        increasing = summarize_learning_curve(
            sizes, (0.50, 0.54, 0.57, 0.59, 0.60))
        self.assertAlmostEqual(increasing.gain, 0.10)
        self.assertEqual(increasing.size_at_95_percent_improvement, 200)
        self.assertGreater(increasing.normalized_log_aulc, 0.50)
        self.assertLess(increasing.normalized_log_aulc, 0.60)

        below_chance = summarize_learning_curve((10, 20), (0.40, 0.45))
        self.assertEqual(below_chance.size_at_95_percent_improvement, 20)

    def test_fixed_effect_slope_removes_stratum_intercepts(self):
        frame = pd.DataFrame({
            "stratum": ["a", "a", "a", "b", "b", "b"],
            "coverage": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "auc": [0.7, 0.9, 1.1, 2.8, 3.0, 3.2],
        })
        result = fixed_effect_slope(
            frame, "coverage", "auc", ["stratum"])
        self.assertAlmostEqual(result.slope, 2.0)
        self.assertAlmostEqual(result.effect_per_10_percentage_points, 0.2)
        self.assertEqual(result.observations, 6)
        self.assertEqual(result.informative_strata, 2)

    def test_fixed_effect_slope_requires_within_stratum_variation(self):
        frame = pd.DataFrame({
            "stratum": ["a", "a", "b", "b"],
            "coverage": [0.2, 0.2, 0.4, 0.4],
            "auc": [0.6, 0.7, 0.8, 0.9],
        })
        with self.assertRaisesRegex(ValueError, "no within-stratum variation"):
            fixed_effect_slope(frame, "coverage", "auc", ["stratum"])

    def test_seed_and_calibration_rotation_are_deterministic(self):
        runs = tuple("run_" + str(index) for index in range(1, 11))
        self.assertEqual(choose_calibration_run("run_1", runs), "run_2")
        self.assertEqual(choose_calibration_run("run_10", runs), "run_1")
        self.assertEqual(stable_seed(7, "x", 1), stable_seed(7, "x", 1))
        self.assertNotEqual(stable_seed(7, "x", 1), stable_seed(7, "x", 2))

    def test_complete_design_statistical_paths(self):
        vlas = ("EO1", "Groot", "Pi0", "SpatialVLA")
        conditions = ("DH", "DH_HybridDT", "DH_HybridRF", "DH_WeightedRF")
        sizes = (10, 20, 50, 100, 200)
        curve_rows = []
        efficiency_rows = []
        coverage_rows = []
        for vla_index, vla in enumerate(vlas):
            for condition_index, condition in enumerate(conditions):
                block = vla + "__" + condition
                block_offset = 0.0001 * (4 * vla_index + condition_index)
                for size_index, size in enumerate(sizes):
                    curve_rows.append({
                        "voxel_size": 0.02,
                        "primary_block_id": block,
                        "training_size": size,
                        "formula_id": "harmonic_mean",
                        "roc_auc": 0.55 + block_offset + 0.01 * size_index,
                    })
                for formula_index, formula_id in enumerate(TABLE4_FORMULA_IDS):
                    technique = FORMULA_REGISTRY[formula_id].name
                    efficiency_rows.append({
                        "voxel_size": 0.02,
                        "primary_block_id": block,
                        "formula_id": formula_id,
                        "technique": technique,
                        "normalized_log_aulc": 0.60 + block_offset + 0.0001 * formula_index,
                    })
                    for repetition, coverage in ((1, 0.20), (2, 0.30)):
                        coverage_rows.append({
                            "voxel_size": 0.02,
                            "vla": vla,
                            "condition": condition,
                            "formula_id": formula_id,
                            "technique": technique,
                            "configuration_id": block + "__200",
                            "test_run": "run_1",
                            "training_size": 20,
                            "sampling_repetition": repetition,
                            "calibration_unique_coverage": coverage,
                            "roc_auc": 0.50 + block_offset + 0.20 * coverage,
                        })

        adjacent, adjacent_metadata = adjacent_reference_tests(
            pd.DataFrame(curve_rows), sizes)
        self.assertEqual(adjacent_metadata["status"], "complete")
        self.assertEqual(len(adjacent), 4)
        np.testing.assert_allclose(adjacent["mean_delta_roc_auc"], 0.01)

        efficiency, efficiency_metadata = sample_efficiency_tests(
            pd.DataFrame(efficiency_rows))
        self.assertEqual(efficiency_metadata["status"], "complete")
        self.assertEqual(len(efficiency), 31)

        with TemporaryDirectory() as directory:
            effects, coverage_metadata = conditional_coverage_analysis(
                pd.DataFrame(coverage_rows), Path(directory))
            self.assertEqual(coverage_metadata["status"], "complete")
            self.assertEqual(len(effects), 16 * 31)
            np.testing.assert_allclose(effects["roc_auc_per_10pp_coverage"], 0.02)


if __name__ == "__main__":
    unittest.main()
