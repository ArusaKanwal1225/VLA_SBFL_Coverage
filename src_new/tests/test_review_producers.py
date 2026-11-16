from replication.references import reference_path
"""Exercise real producers with frozen aggregate inputs and temporary outputs."""
import contextlib
import importlib
import io
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src_new.tests.test_statistical_review_artifacts import ROOT, SPECS


@unittest.skipUnless((ROOT/"data/DATASET.json").is_file(), "Install the Zenodo dataset for integration tests")
class ReviewProducerTests(unittest.TestCase):
    def test_statistical_producers_and_exact_tables(self):
        jobs = [("RQ1", "RQ1/results/phase_a_primary"),
                ("RQ2_baselines", "RQ2_baselines/results/all_models"),
                ("RQ2", "RQ2/results/phase_a_granularity"),
                ("RQ3_v2", "RQ3_v2/results/learning_curves")]
        for directory, results in jobs:
            source = reference_path(ROOT, results) / "aggregate"
            runner = importlib.import_module(f"experiments.{directory}.run_experiment")
            tables = importlib.import_module(f"experiments.{directory}.generate_tables")
            with self.subTest(rq=directory), tempfile.TemporaryDirectory() as tmp:
                aggregate = Path(tmp) / "aggregate"
                shutil.copytree(source, aggregate)
                # Remove review outputs: producers must reconstruct p-values from blocks.
                for path in aggregate.iterdir():
                    if "_exact" in path.name or "_approximate" in path.name:
                        path.unlink()
                for method in (("exact",) if directory == "RQ1" else ("approximate", "exact")):
                    runner.regenerate_statistical_review(aggregate, method=method)
                    for path in aggregate.glob(f"*_{method}.csv"):
                        expected = pd.read_csv(source / path.name)
                        actual = pd.read_csv(path)
                        if "formula_id" in actual:
                            keys = [c for c in ("voxel_size", "formula_id") if c in actual]
                            actual = actual.sort_values(keys).reset_index(drop=True)
                            expected = expected.sort_values(keys).reset_index(drop=True)
                        # RQ4 adds the historical primary_blocks column formerly omitted.
                        pd.testing.assert_frame_equal(actual[expected.columns], expected,
                            check_exact=False, rtol=0, atol=1e-14)
                output = Path(tmp) / "tables"
                with patch("sys.argv", ["tables", "--aggregate-dir", str(aggregate),
                    "--output-dir", str(output), "--statistical-review-method", "exact"]), \
                    contextlib.redirect_stdout(io.StringIO()):
                    tables.main()
                for path in output.glob("*.tex"):
                    expected = (source.parent / "tables" / path.name).read_text()
                    self.assertEqual(path.read_text(), expected)
                for path in output.glob("*.json"):
                    self.assertNotIn(str(ROOT), path.read_text())

    def test_bootstrap_recomputes_intervals_from_inputs(self):
        from experiments import bootstrap_review
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for directory in ("RQ1/results/phase_a_primary", "RQ2_baselines/results/all_models",
                              "RQ2/results/phase_a_granularity"):
                source = reference_path(ROOT, directory) / "aggregate"
                target = root / "experiments" / directory / "aggregate"
                shutil.copytree(source, target)
                for path in target.glob("bootstrap_block_sensitivity*"):
                    path.unlink()
            with patch.object(bootstrap_review, "ROOT", root):
                for rq in ("RQ1", "RQ2", "RQ3"):
                    self.assertLess(bootstrap_review.audit(rq)["maximum_reproduction_error"], 1e-12)

    def test_missing_metadata_writes_no_tables(self):
        for directory, results in (("RQ2_baselines", "RQ2_baselines/results/all_models"),
                                   ("RQ2", "RQ2/results/phase_a_granularity"),
                                   ("RQ3_v2", "RQ3_v2/results/learning_curves")):
            module = importlib.import_module(f"experiments.{directory}.generate_tables")
            with tempfile.TemporaryDirectory() as tmp:
                aggregate, output = Path(tmp)/"aggregate", Path(tmp)/"tables"
                shutil.copytree(reference_path(ROOT, results)/"aggregate", aggregate)
                (aggregate/"statistical_review_exact.json").unlink()
                with patch("sys.argv", ["tables", "--aggregate-dir", str(aggregate),
                    "--output-dir", str(output), "--statistical-review-method", "exact"]):
                    with self.assertRaises(FileNotFoundError):
                        module.main()
                self.assertFalse(output.exists())
