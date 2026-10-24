import unittest

import numpy as np

from experiments.RQ1.analysis import (
    average_ranks,
    friedman_test,
    hierarchical_bootstrap,
    hierarchical_budget_bootstrap,
    holm_adjust,
    wilcoxon_signed_rank,
)


class RQ1AnalysisTests(unittest.TestCase):
    def test_average_ranks_are_tie_aware(self):
        np.testing.assert_allclose(average_ranks([3, 1, 1, 2]), [4, 1.5, 1.5, 3])
        np.testing.assert_allclose(
            average_ranks([3, 1, 1, 2], descending=True), [1, 3.5, 3.5, 2])

    def test_friedman_detects_consistent_ordering(self):
        values = np.asarray([
            [0.9, 0.7, 0.5],
            [0.8, 0.6, 0.4],
            [0.85, 0.65, 0.45],
            [0.95, 0.75, 0.55],
            [0.88, 0.68, 0.48],
        ])
        result = friedman_test(values)
        self.assertAlmostEqual(result.statistic, 10.0)
        self.assertLess(result.p_value, 0.01)

    def test_wilcoxon_and_holm(self):
        result = wilcoxon_signed_rank([4, 5, 6, 7, 8], [1, 2, 3, 4, 5])
        self.assertGreater(result.rank_biserial, 0)
        self.assertLess(result.p_value, 0.1)
        adjusted = holm_adjust({"a": 0.01, "b": 0.03, "c": 0.2})
        self.assertEqual(adjusted, {"a": 0.03, "b": 0.06, "c": 0.2})

    def test_hierarchical_bootstrap_shapes_and_leader_delta(self):
        scores = np.asarray([
            [[0.8, 0.6], [0.9, 0.7]],
            [[0.7, 0.5], [0.8, 0.6]],
        ])
        lower, upper, delta_lower, delta_upper = hierarchical_bootstrap(
            scores, leader_index=0, iterations=200, seed=7)
        self.assertEqual(lower.shape, (2,))
        self.assertTrue(np.all(lower <= upper))
        self.assertEqual(delta_lower[0], 0.0)
        self.assertEqual(delta_upper[0], 0.0)
        self.assertGreater(delta_lower[1], 0.0)

    def test_budget_bootstrap_preserves_shapes_and_leader_delta(self):
        scores = np.asarray([
            [[[0.8, 0.6], [0.9, 0.7]], [[0.7, 0.5], [0.8, 0.6]]],
            [[[0.9, 0.7], [0.8, 0.6]], [[0.85, 0.65], [0.75, 0.55]]],
        ])
        lower, upper, delta_lower, delta_upper = hierarchical_budget_bootstrap(
            scores, leader_index=0, iterations=200, seed=7)
        self.assertEqual(lower.shape, (2,))
        self.assertTrue(np.all(lower <= upper))
        self.assertEqual(delta_lower[0], 0.0)
        self.assertEqual(delta_upper[0], 0.0)
        self.assertGreater(delta_lower[1], 0.0)


if __name__ == "__main__":
    unittest.main()
