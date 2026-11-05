"""Regression checks for the frozen Phase D aggregate-only reviews."""
import hashlib
import json
import unittest
from unittest.mock import patch
import importlib
import contextlib
import io
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SPECS = (
    ("RQ2_baselines/results/all_models", "statistical_comparison", ["method_id"], "raw_p", "holm_p", 8, "rq2_baseline_statistics"),
    ("RQ2/results/phase_a_granularity", "technique_size_effects", ["voxel_size", "formula_id"], "wilcoxon_p_value", "holm_adjusted_p_value", 68, "rq3_technique_size_effects"),
    ("RQ2/results/phase_a_granularity", "technique_size_interactions", ["voxel_size", "formula_id"], "wilcoxon_p_value", "holm_adjusted_p_value", 66, "rq3_technique_size_interactions"),
    ("RQ3_v2/results/learning_curves", "adjacent_size_tests", ["lower_training_size", "upper_training_size"], "wilcoxon_p_value", "holm_adjusted_p_value", 4, "rq3v2_adjacent_size_tests"),
    ("RQ3_v2/results/learning_curves", "sample_efficiency_tests", ["formula_id"], "wilcoxon_p_value", "holm_adjusted_p_value", 33, "rq3v2_sample_efficiency_tests"),
    ("RQ3_v2/results/learning_curves", "conditional_coverage_tests", ["formula_id"], "wilcoxon_p_value", "holm_adjusted_p_value", 34, "rq3v2_conditional_coverage_tests"),
)


class StatisticalReviewArtifactsTests(unittest.TestCase):
    def test_review_cli_returns_before_experiment_dispatch(self):
        for module_name, output in (
            ("experiments.RQ2_baselines.run_experiment", "RQ2_baselines/results/all_models"),
            ("experiments.RQ2.run_experiment", "RQ2/results/phase_a_granularity"),
            ("experiments.RQ3_v2.run_experiment", "RQ3_v2/results/learning_curves"),
        ):
            module = importlib.import_module(module_name)
            with patch("sys.argv", [module_name, "--statistical-review-only", "--output-root",
                                   str(ROOT / "experiments" / output)]), \
                 patch.object(module, "regenerate_statistical_review", return_value={}) as review, \
                 patch.object(module, "aggregate_cross_size" if module_name == "experiments.RQ2.run_experiment" else "aggregate_results", side_effect=AssertionError("experiment aggregation called")), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(module.main(), 0)
                review.assert_called_once()

    def test_historical_reproduction_full_families_and_tables(self):
        changes = []
        for directory, stem, keys, raw, adjusted, family, table in SPECS:
            root = ROOT / "experiments" / directory
            old = pd.read_csv(root / "aggregate" / (stem + ".csv")).set_index(keys).sort_index()
            for method in ("approximate", "exact"):
                with self.subTest(stem=stem, method=method):
                    frame = pd.read_csv(root / "aggregate" / f"{stem}_{method}.csv").set_index(keys).sort_index()
                    self.assertTrue(old.index.equals(frame.index))
                    self.assertEqual(set(frame.wilcoxon_method), {method})
                    self.assertEqual(set(frame.holm_family_size), {family})
                    self.assertNotIn("z_score", frame.columns)
                    if method == "approximate":
                        np.testing.assert_allclose(old[[raw, adjusted]], frame[[raw, adjusted]], rtol=0, atol=1e-14)
                    candidates = frame
                    if family == 33:
                        candidates = frame.drop(index="harmonic_mean")
                    self.assertEqual(len(candidates), family)
                    ordered = candidates.sort_values(raw)
                    expected = np.minimum(1, np.maximum.accumulate(
                        ordered[raw].to_numpy() * np.arange(family, 0, -1)))
                    np.testing.assert_allclose(ordered[adjusted], expected, rtol=0, atol=1e-14)
                    if method == "approximate":
                        continue  # Regression uses aggregates; approximate tables are optional.
                    rendered = pd.read_csv(root / "tables" / f"{table}_{method}.csv").set_index(keys).sort_index()
                    pd.testing.assert_frame_equal(frame, rendered, check_exact=False, atol=1e-14, rtol=0)
                    self.assertTrue((root / "tables" / f"{table}_{method}.tex").is_file())
                    latex = (root / "tables" / f"{table}_{method}.tex").read_text()
                    precision = ".4f" if family == 8 else (".6g" if family in (68, 66) else ".8g")
                    for value in frame[adjusted]:
                        self.assertIn(format(value, precision), latex)
                    if method == "exact":
                        self.assertLessEqual(int(frame.nonzero_pairs.max()), 20)
                        changes.extend((stem, str(key)) for key in frame.index[(old[adjusted] < .05) != (frame[adjusted] < .05)])
        self.assertEqual(len(changes), 4)
        self.assertTrue(all(stem == "technique_size_effects" for stem, key in changes))
        for formula in ("rogot1", "scott", "zoltar", "mountford"):
            self.assertTrue(any(formula in key for stem, key in changes))

    def test_review_source_and_output_hashes(self):
        for directory in sorted({item[0] for item in SPECS}):
            aggregate = ROOT / "experiments" / directory / "aggregate"
            for method in ("approximate", "exact"):
                payload = json.loads((aggregate / f"statistical_review_{method}.json").read_text())
                sources = payload.get("source_sha256", payload.get("sources_sha256"))
                outputs = payload["output_sha256"]
                if isinstance(outputs, str):
                    outputs = {payload["output"]: outputs}
                for name, expected in {**sources, **outputs}.items():
                    self.assertEqual(hashlib.sha256((aggregate / name).read_bytes()).hexdigest(), expected)
                if method == "approximate":
                    continue
                tables = aggregate.parent / "tables"
                manifest = json.loads((tables / f"statistical_review_{method}_manifest.json").read_text())
                outputs = manifest.get("outputs", manifest.get("generated_files"))
                if isinstance(outputs, dict):
                    outputs = [{"path": name, "sha256": value} for name, value in outputs.items()]
                for entry in outputs:
                    path = Path(entry["path"])
                    path = ROOT / path if len(path.parts) > 1 else tables / path
                    self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), entry["sha256"])


if __name__ == "__main__":
    unittest.main()
