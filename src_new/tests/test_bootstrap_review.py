from replication.references import reference_path
import hashlib
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.bootstrap_review import block_draws


class BootstrapReviewTests(unittest.TestCase):
    def test_complete_blocks_preserve_pairing_and_seed(self):
        values = np.arange(16, dtype=float)
        paired = np.column_stack([values, values, values + 0.1])
        first = block_draws(paired, 100, 31)
        np.testing.assert_array_equal(first, block_draws(paired, 100, 31))
        np.testing.assert_array_equal(first[:, 0], first[:, 1])
        np.testing.assert_allclose(first[:, 2]-first[:, 0], 0.1, rtol=0, atol=1e-14)

    @unittest.skipUnless((Path(__file__).resolve().parents[2]/"data/DATASET.json").is_file(), "Install the Zenodo dataset for integration tests")
    def test_historical_reproduction_and_manifest_integrity(self):
        root = Path(__file__).resolve().parents[2] / "experiments"
        total, changed = 0, []
        for directory, count in [("RQ1/results/phase_a_primary", 68),
                                 ("RQ2_baselines/results/all_models", 16),
                                 ("RQ2/results/phase_a_granularity", 236)]:
            aggregate = reference_path(root.parent, directory) / "aggregate"
            frame = pd.read_csv(aggregate / "bootstrap_block_sensitivity.csv")
            manifest = json.loads((aggregate / "bootstrap_block_sensitivity_manifest.json").read_text())
            self.assertEqual(len(frame), count)
            self.assertLess(frame.reproduction_error.max(), 1e-12)
            self.assertEqual(set(frame.iterations), {10000})
            np.testing.assert_allclose(frame.block_width, frame.block_upper-frame.block_lower, atol=1e-14)
            np.testing.assert_allclose(frame.width_change, frame.block_width-frame.historical_width, atol=1e-14)
            for name, expected in {**manifest["source_sha256"], **manifest["output_sha256"]}.items():
                self.assertEqual(hashlib.sha256((aggregate/name).read_bytes()).hexdigest(), expected)
            changed.extend(frame.loc[frame.zero_in_historical != frame.zero_in_block, "identity"])
            total += len(frame)
        self.assertEqual(total, 320)
        self.assertEqual(changed, ["interaction/0.05/rogot2"])
