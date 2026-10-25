import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.RQ2_baselines.analysis import paired_hierarchical_bootstrap
from experiments.RQ2_baselines.catalogue import METRIC_BY_ID, METRIC_SPECS
from experiments.RQ2_baselines.run_experiment import load_metric_score, numeric_leaves


class RQ2BaselineTests(unittest.TestCase):
    def test_metric_catalogue_is_complete_and_directions_are_frozen(self):
        self.assertEqual(len(METRIC_SPECS), 13)
        self.assertEqual(len(METRIC_BY_ID), 13)
        self.assertEqual(METRIC_BY_ID["token_probability"].direction, -1)
        self.assertEqual(METRIC_BY_ID["pcs"].direction, -1)
        self.assertEqual(METRIC_BY_ID["tcp_acceleration_instability"].direction, 1)
        self.assertTrue(METRIC_BY_ID["optimal_trajectory"].task_aware)

    def test_numeric_leaves_flattens_nested_metric_payload(self):
        self.assertEqual(
            numeric_leaves([[1, 2], {"second": 4, "first": 3}]),
            [1.0, 2.0, 3.0, 4.0],
        )

    def test_metric_score_reports_empty_and_applies_direction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            empty = root / "empty.json"
            empty.write_text("[]", encoding="utf-8")
            self.assertEqual(load_metric_score(empty, 1), (None, 0, "empty"))

            values = root / "values.json"
            values.write_text(json.dumps([[1.0, 3.0], [5.0, 7.0]]), encoding="utf-8")
            score, count, status = load_metric_score(values, -1)
            self.assertEqual(score, -4.0)
            self.assertEqual(count, 4)
            self.assertEqual(status, "available")

    def test_paired_hierarchical_bootstrap_preserves_constant_gap(self):
        reference = np.arange(24, dtype=float).reshape(3, 2, 4)
        candidate = reference + 0.25
        ref_low, ref_high, gap_low, gap_high = paired_hierarchical_bootstrap(
            reference, candidate, iterations=200, seed=9
        )
        self.assertLess(ref_low, ref_high)
        self.assertTrue(np.isclose(gap_low, 0.25))
        self.assertTrue(np.isclose(gap_high, 0.25))


if __name__ == "__main__":
    unittest.main()
