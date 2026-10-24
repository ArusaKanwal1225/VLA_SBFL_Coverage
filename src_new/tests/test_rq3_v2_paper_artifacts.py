import unittest

import numpy as np
import pandas as pd

from experiments.RQ3_v2.generate_plots import rank_matrix
from experiments.RQ3_v2.generate_tables import (
    TRAINING_SIZES,
    VLA_ORDER,
    build_coverage_performance_data,
    competitive_formula_ids,
)


class RQ3V2PaperArtifactTests(unittest.TestCase):
    def test_coverage_performance_table_is_vla_stratified_and_keeps_ties(self):
        rows = []
        for vla in VLA_ORDER:
            for size_index, size in enumerate(TRAINING_SIZES):
                coverage = 0.10 + 0.05 * size_index
                rows.extend(
                    [
                        {
                            "vla": vla,
                            "training_size": size,
                            "formula_id": "harmonic_mean",
                            "technique": "Harmonic Mean",
                            "roc_auc": 0.60 + 0.01 * size_index,
                            "roc_auc_rank": 1.5,
                            "calibration_unique_coverage": coverage,
                            "test_unique_coverage": coverage + 0.01,
                        },
                        {
                            "vla": vla,
                            "training_size": size,
                            "formula_id": "rogot2",
                            "technique": "Rogot2",
                            "roc_auc": 0.60 + 0.01 * size_index,
                            "roc_auc_rank": 1.5,
                            "calibration_unique_coverage": coverage,
                            "test_unique_coverage": coverage + 0.01,
                        },
                    ]
                )
        table = build_coverage_performance_data(pd.DataFrame(rows))
        self.assertEqual(len(table), len(VLA_ORDER) * len(TRAINING_SIZES))
        self.assertEqual(set(table["vla"]), set(VLA_ORDER))
        self.assertTrue((table["leading_techniques"] == "Harmonic Mean / Rogot2").all())
        np.testing.assert_allclose(
            table["test_unique_coverage"] - table["calibration_unique_coverage"],
            0.01,
        )

    def test_competitive_set_is_union_of_vla_top_ranks_plus_reference(self):
        rows = []
        techniques = (
            ("harmonic_mean", "Harmonic Mean"),
            ("dennis", "Dennis"),
            ("zoltar", "Zoltar"),
        )
        leaders = {
            "EO1": "dennis",
            "Groot": "dennis",
            "Pi0": "harmonic_mean",
            "SpatialVLA": "zoltar",
        }
        for vla in VLA_ORDER:
            for formula_id, technique in techniques:
                rank = 1.0 if leaders[vla] == formula_id else 2.0
                rows.append(
                    {
                        "vla": vla,
                        "formula_id": formula_id,
                        "technique": technique,
                        "normalized_log_aulc": 0.70 - 0.01 * rank,
                        "naulc_rank": rank,
                    }
                )
        selected = competitive_formula_ids(pd.DataFrame(rows), cutoff=1)
        self.assertEqual(set(selected), {"dennis", "harmonic_mean", "zoltar"})

    def test_rank_matrix_uses_requested_technique_and_size_order(self):
        rows = []
        for formula_index, formula_id in enumerate(("a", "b"), start=1):
            for size_index, size in enumerate(TRAINING_SIZES):
                rows.append(
                    {
                        "voxel_size": 0.02,
                        "vla": "EO1",
                        "training_size": size,
                        "formula_id": formula_id,
                        "roc_auc_rank": formula_index + size_index,
                    }
                )
        matrix = rank_matrix(pd.DataFrame(rows), "EO1", ("b", "a"))
        self.assertEqual(matrix.shape, (2, len(TRAINING_SIZES)))
        np.testing.assert_allclose(matrix[0], [2, 3, 4, 5, 6])
        np.testing.assert_allclose(matrix[1], [1, 2, 3, 4, 5])


if __name__ == "__main__":
    unittest.main()
