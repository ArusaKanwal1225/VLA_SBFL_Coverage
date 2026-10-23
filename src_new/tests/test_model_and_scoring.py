import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from vla_sbfl.formulas import TABLE4_FORMULA_IDS
from vla_sbfl.model import SBFLModel
from vla_sbfl.scoring import TrajectoryScorer, empirical_percentiles
from vla_sbfl.voxelizer import VoxelGrid


class ModelAndScoringTests(unittest.TestCase):
    def setUp(self):
        self.grid = VoxelGrid(0.5, (0.0, 0.0, 0.0), "test_frame")
        self.visits = {
            "failed": frozenset({(0, 0, 0), (1, 0, 0)}),
            "passed_a": frozenset({(1, 0, 0)}),
            "passed_b": frozenset({(2, 0, 0)}),
        }
        self.labels = {"failed": 1, "passed_a": 0, "passed_b": 0}
        self.model = SBFLModel.fit(
            self.visits, self.labels, self.grid, TABLE4_FORMULA_IDS,
            metadata={"purpose": "unit_test"},
        )

    def test_contingency_counts(self):
        rows = {
            tuple(voxel): tuple(counts)
            for voxel, counts in zip(self.model.voxels, self.model.counts)
        }
        self.assertEqual(rows[(0, 0, 0)], (1, 0, 0, 2))
        self.assertEqual(rows[(1, 0, 0)], (1, 1, 0, 1))
        self.assertEqual(rows[(2, 0, 0)], (0, 1, 1, 1))

    def test_artifact_round_trip(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "artifact"
            self.model.save(destination)
            loaded = SBFLModel.load(destination)
            self.assertEqual(loaded.formula_ids, self.model.formula_ids)
            np.testing.assert_array_equal(loaded.voxels, self.model.voxels)
            np.testing.assert_array_equal(loaded.counts, self.model.counts)
            self.assertEqual(loaded.training_episode_ids, self.model.training_episode_ids)
            for formula_id in loaded.formula_ids:
                np.testing.assert_allclose(loaded.scores[formula_id], self.model.scores[formula_id])
            self.assertTrue((destination / "metadata.json").is_file())
            self.assertTrue((destination / "formula_catalog.csv").is_file())
            self.assertTrue((destination / "training_episode_ids.csv.gz").is_file())
            self.assertTrue((destination / "formula_edge_cases.csv.gz").is_file())

    def test_scoring_known_partial_and_unknown(self):
        episodes = pd.DataFrame({
            "episode_id": ["known", "partial", "unknown"],
            "failure": [0, 1, 1],
        })
        visits = {
            "known": frozenset({(0, 0, 0)}),
            "partial": frozenset({(0, 0, 0), (99, 99, 99)}),
            "unknown": frozenset({(99, 99, 99)}),
        }
        result = TrajectoryScorer(self.model).score(visits, episodes).set_index("episode_id")
        self.assertEqual(result.loc["known", "scoring_status"], "scored")
        self.assertEqual(result.loc["partial", "known_fraction"], 0.5)
        self.assertEqual(result.loc["unknown", "scoring_status"], "no_training_support")
        score_columns = [column for column in result.columns if column.startswith("score__")]
        np.testing.assert_allclose(result.loc["unknown", score_columns].astype(float), 0.5)
        self.assertTrue(np.isfinite(result[score_columns].to_numpy(dtype=float)).all())

    def test_percentiles_are_tie_aware(self):
        values = empirical_percentiles(np.asarray([1.0, 2.0, 2.0, 4.0]))
        np.testing.assert_allclose(values, [0.0, 0.5, 0.5, 1.0])
        np.testing.assert_allclose(empirical_percentiles(np.asarray([3.0])), [0.5])

    def test_single_class_training_is_finite_and_explicit(self):
        model = SBFLModel.fit(
            {"failed": frozenset({(0, 0, 0), (1, 0, 0)})},
            {"failed": 1},
            self.grid,
            TABLE4_FORMULA_IDS,
        )
        self.assertFalse(model.metadata["training_has_both_classes"])
        self.assertEqual(model.metadata["training_failures"], 1)
        self.assertEqual(model.metadata["training_passes"], 0)
        for formula_id in model.formula_ids:
            self.assertTrue(np.isfinite(model.scores[formula_id]).all())


if __name__ == "__main__":
    unittest.main()
