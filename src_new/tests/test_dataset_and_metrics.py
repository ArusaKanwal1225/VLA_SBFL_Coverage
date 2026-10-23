import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from vla_sbfl.config import DatasetSelection
from vla_sbfl.dataset import load_trajectory_dataset
from vla_sbfl.metrics import average_precision, evaluate_predictions, roc_auc
from vla_sbfl.voxelizer import VoxelGrid


class DatasetAndMetricTests(unittest.TestCase):
    def test_voxelization_and_manifest_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "run_1"
            run.mkdir()
            pd.DataFrame({
                "episode_id": ["train_fail", "test_pass"],
                "failure": [1, 0],
                "num_steps": [2, 2],
                "algorithm": ["DH", "DH"],
            }).to_csv(run / "episodes.csv", index=False)
            pd.DataFrame({
                "episode_id": ["train_fail", "train_fail", "test_pass", "test_pass"],
                "step": [0, 1, 0, 1],
                "x": [0.1, 0.9, -0.1, -0.9],
                "y": [0.1, 0.1, 0.1, 0.1],
                "z": [0.1, 0.1, 0.1, 0.1],
            }).to_csv(run / "trajectories.csv", index=False)
            manifest = root / "manifest.csv"
            pd.DataFrame({
                "episode_id": ["train_fail", "test_pass"],
                "failure": [1, 0],
                "seed": [11, 11],
                "split": ["train", "test"],
            }).to_csv(manifest, index=False)
            dataset = load_trajectory_dataset(
                [root],
                grid=VoxelGrid(1.0, (0.0, 0.0, 0.0)),
                selection=DatasetSelection(manifest=manifest, split="test", seed=11),
                require_labels=True,
                chunk_size=2,
            )
            self.assertEqual(list(dataset.episodes["episode_id"]), ["test_pass"])
            self.assertEqual(dataset.visits["test_pass"], frozenset({(-1, 0, 0)}))

    def test_metrics_perfect_and_tied(self):
        labels = [0, 0, 1, 1]
        self.assertEqual(roc_auc(labels, [0.1, 0.2, 0.8, 0.9]), 1.0)
        self.assertEqual(average_precision(labels, [0.1, 0.2, 0.8, 0.9]), 1.0)
        self.assertEqual(roc_auc(labels, [0.5, 0.5, 0.5, 0.5]), 0.5)
        self.assertEqual(average_precision(labels, [0.5, 0.5, 0.5, 0.5]), 0.5)

    def test_prediction_evaluation(self):
        predictions = pd.DataFrame({
            "failure": [0, 0, 1, 1],
            "score__one": [0.1, 0.2, 0.8, 0.9],
            "score__two": [0.5, 0.5, 0.5, 0.5],
        })
        metrics = evaluate_predictions(predictions, ("one", "two"))
        self.assertEqual(list(metrics["formula_id"]), ["one", "two"])
        self.assertTrue(np.isfinite(metrics[["roc_auc", "average_precision"]]).all().all())


if __name__ == "__main__":
    unittest.main()

