import unittest

import numpy as np

from experiments.RQ1.analysis import (
    MAX_EXACT_NONZERO_PAIRS,
    all_pairwise_wilcoxon,
    average_ranks,
    competition_ranks,
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

    def test_competition_ranks_assign_the_minimum_rank_to_ties(self):
        np.testing.assert_array_equal(
            competition_ranks([3, 1, 1, 2]), [4, 1, 1, 3])
        np.testing.assert_array_equal(
            competition_ranks([3, 1, 1, 2], descending=True), [1, 3, 3, 2])

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

    def test_approximate_wilcoxon_reproduces_previous_result(self):
        result = wilcoxon_signed_rank([4, 5, 6, 7, 8], [1, 2, 3, 4, 5])
        self.assertEqual(result.statistic, 0.0)
        self.assertEqual(result.z_score, 2.0869967789998034)
        self.assertAlmostEqual(result.p_value, 0.036888425707049886, places=15)
        self.assertEqual(result.rank_biserial, 1.0)
        self.assertEqual(result.nonzero_pairs, 5)
        tied_with_zeros = wilcoxon_signed_rank(
            [1, 2, 3, 4, 5], [1, 1, 4, 2, 5])
        self.assertEqual(tied_with_zeros.statistic, 1.5)
        self.assertEqual(tied_with_zeros.z_score, 0.5443310539518174)
        self.assertAlmostEqual(tied_with_zeros.p_value, 0.5862136810731401, places=15)
        self.assertEqual(tied_with_zeros.rank_biserial, 0.5)
        self.assertEqual(tied_with_zeros.nonzero_pairs, 3)
        self.assertEqual(
            tied_with_zeros,
            wilcoxon_signed_rank(
                [1, 2, 3, 4, 5], [1, 1, 4, 2, 5], method="approximate"),
        )

    def test_exact_wilcoxon_all_positive_and_all_negative(self):
        positive = wilcoxon_signed_rank(
            [1, 2, 3, 4, 5], [0, 0, 0, 0, 0], method="exact")
        negative = wilcoxon_signed_rank(
            [0, 0, 0, 0, 0], [1, 2, 3, 4, 5], method="exact")
        self.assertEqual(positive.statistic, 0.0)
        self.assertEqual(positive.p_value, 0.0625)
        self.assertEqual(positive.rank_biserial, 1.0)
        self.assertEqual(negative.statistic, 0.0)
        self.assertEqual(negative.p_value, positive.p_value)
        self.assertEqual(negative.rank_biserial, -1.0)

    def test_exact_wilcoxon_matches_audited_sixteen_same_sign_case(self):
        result = wilcoxon_signed_rank(
            range(1, 17), [0] * 16, method="exact")
        self.assertEqual(result.p_value, 2.0 / (2**16))
        self.assertEqual(result.nonzero_pairs, 16)

    def test_exact_wilcoxon_all_zero(self):
        result = wilcoxon_signed_rank(
            [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], method="exact")
        self.assertEqual(result.statistic, 0.0)
        self.assertEqual(result.z_score, 0.0)
        self.assertEqual(result.p_value, 1.0)
        self.assertEqual(result.rank_biserial, 0.0)
        self.assertEqual(result.nonzero_pairs, 0)

    def test_exact_wilcoxon_discards_zero_differences(self):
        with_zeros = wilcoxon_signed_rank(
            [1.0, 4.0, 2.0, 8.0], [0.0, 4.0, 4.0, 8.0], method="exact")
        without_zeros = wilcoxon_signed_rank(
            [1.0, 2.0], [0.0, 4.0], method="exact")
        self.assertEqual(with_zeros, without_zeros)
        self.assertEqual(with_zeros.nonzero_pairs, 2)

    def test_exact_wilcoxon_uses_average_ranks_for_absolute_ties(self):
        result = wilcoxon_signed_rank(
            [1.0, -1.0, 2.0], [0.0, 0.0, 0.0], method="exact")
        self.assertEqual(result.statistic, 1.5)
        self.assertEqual(result.p_value, 0.75)
        self.assertEqual(result.rank_biserial, 0.5)

    def test_exact_wilcoxon_is_symmetric_when_samples_are_swapped(self):
        first = [3.0, 1.0, 7.0, 4.0, 9.0]
        second = [1.0, 2.0, 4.0, 4.0, 5.0]
        forward = wilcoxon_signed_rank(first, second, method="exact")
        reverse = wilcoxon_signed_rank(second, first, method="exact")
        self.assertEqual(forward.statistic, reverse.statistic)
        self.assertEqual(forward.z_score, reverse.z_score)
        self.assertEqual(forward.p_value, reverse.p_value)
        self.assertEqual(forward.rank_biserial, -reverse.rank_biserial)
        self.assertEqual(forward.nonzero_pairs, reverse.nonzero_pairs)

    def test_exact_wilcoxon_rejects_samples_above_enumeration_limit(self):
        count = MAX_EXACT_NONZERO_PAIRS + 1
        with self.assertRaisesRegex(ValueError, "at most"):
            wilcoxon_signed_rank(
                range(1, count + 1), [0] * count, method="exact")

    def test_wilcoxon_rejects_unknown_method(self):
        with self.assertRaisesRegex(ValueError, "approximate.*exact"):
            wilcoxon_signed_rank([1.0], [0.0], method="automatic")

    def test_holm(self):
        adjusted = holm_adjust({"a": 0.01, "b": 0.03, "c": 0.2})
        self.assertEqual(adjusted, {"a": 0.03, "b": 0.06, "c": 0.2})

    def test_all_pairwise_wilcoxon_uses_one_complete_holm_family(self):
        values = np.asarray([
            [0.9, 0.7, 0.5],
            [0.8, 0.6, 0.4],
            [0.7, 0.5, 0.3],
            [0.6, 0.4, 0.2],
        ])
        results = all_pairwise_wilcoxon(
            values, ("first", "second", "third"), method="exact")
        self.assertEqual(len(results), 3)
        self.assertEqual(
            {(item.first_id, item.second_id) for item in results},
            {("first", "second"), ("first", "third"), ("second", "third")},
        )
        self.assertTrue(all(item.p_value == 0.125 for item in results))
        self.assertTrue(all(item.holm_adjusted_p_value == 0.375 for item in results))

    def test_all_pairwise_wilcoxon_has_561_pairs_for_34_techniques(self):
        values = np.zeros((16, 34), dtype=np.float64)
        results = all_pairwise_wilcoxon(
            values, tuple("technique_{}".format(index) for index in range(34)),
            method="exact",
        )
        self.assertEqual(len(results), 561)
        self.assertTrue(all(item.p_value == 1.0 for item in results))
        self.assertTrue(all(item.holm_adjusted_p_value == 1.0 for item in results))

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
