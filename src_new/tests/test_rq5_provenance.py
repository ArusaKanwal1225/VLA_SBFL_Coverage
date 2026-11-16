from replication.references import reference_path
"""Compatibility gates for frozen RQ5 maps after statistical-only updates."""
import json
import contextlib
import io
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.RQ5.provenance import ROOT, verify_compatibility, verify_frozen_maps


@unittest.skipUnless((ROOT/"data/DATASET.json").is_file(), "Install the Zenodo dataset for integration tests")
class RQ5ProvenanceTests(unittest.TestCase):
    def test_combined_baseline_reporting_preserves_original_performance(self):
        import pandas as pd
        from experiments.RQ2_baselines import generate_tables
        aggregate = reference_path(ROOT, 'RQ2_baselines/results/all_models')/'aggregate'
        rq1 = reference_path(ROOT, 'RQ1/results/phase_a_primary')/'aggregate'
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'tables'
            local_aggregate, local_rq1 = Path(tmp)/'baseline', Path(tmp)/'rq1'
            shutil.copytree(aggregate, local_aggregate)
            shutil.copytree(rq1, local_rq1)
            with patch('sys.argv', ['tables', '--aggregate-dir', str(local_aggregate),
                '--rq1-aggregate-dir', str(local_rq1), '--output-dir', str(output)]), \
                patch.object(generate_tables, 'repository_root', return_value=Path(tmp)), \
                contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(generate_tables.main(), 0)
            pd.testing.assert_frame_equal(pd.read_csv(output/'rq2_baseline_performance.csv'),
                pd.read_csv(aggregate.parent/'tables/rq2_baseline_performance.csv'))
            self.assertTrue((output/'rq2_baseline_performance_top5_rq1.tex').is_file())
            combined = pd.read_csv(output/'rq2_baseline_performance_top5_rq1.csv')
            self.assertTrue({'Harmonic Mean', 'Rogot2'}.issubset(set(combined.Method)))
            self.assertTrue({'Optimal Trajectory', 'TCP Acceleration Instability'}.issubset(set(combined.Method)))
            self.assertNotIn(str(ROOT), (output/'manifest.json').read_text())

    def test_historical_signature_and_scientific_compatibility(self):
        self.assertEqual(verify_compatibility(),
            '910e24f9b9d91fdd8863ed3353c12dd082a81de824292f140fc39e5435eeda71')

    def test_statistical_updates_allowed_and_scientific_mutations_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for relative in ('src_new/vla_sbfl', 'experiments/RQ1'):
                shutil.copytree(ROOT/relative, root/relative)
            shutil.copytree(ROOT/'data/reference/rq1', root/'data/reference/rq1')
            analysis = root/'experiments/RQ1/analysis.py'
            analysis.write_text(analysis.read_text()+'\n# Statistical reporting revision\n')
            verify_compatibility(root)
            for relative, old, new in (
                ('src_new/vla_sbfl/voxelizer.py', '0', '1'),
                ('src_new/vla_sbfl/scoring.py', 'np.max(', 'np.min('),
                ('src_new/vla_sbfl/model.py', 'ARTIFACT_VERSION = 1', 'ARTIFACT_VERSION = 2'),
            ):
                path = root/relative
                source = path.read_text()
                path.write_text(source.replace(old, new, 1))
                with self.assertRaisesRegex(ValueError, 'compatibility mismatch'):
                    verify_compatibility(root)
                path.write_text(source)
            runner = root/'experiments/RQ1/run_experiment.py'
            source = runner.read_text()
            mutated = source.replace('wanted = {str(value) for value in episode_ids}', 'wanted = set()', 1)
            self.assertNotEqual(source, mutated)
            runner.write_text(mutated)
            with self.assertRaisesRegex(ValueError, 'training/helper compatibility'):
                verify_compatibility(root)
            runner.write_text(source)
            run = root/'data/reference/rq1/run.json'
            record = json.loads(run.read_text()); record['experiment_signature'] = 'invalid'
            run.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, 'Historical signature'):
                verify_compatibility(root)

    def test_published_maps_and_corruption_detection(self):
        maps = ROOT/'data/rq5/reconstructed_pickup_maps'
        self.assertEqual(verify_frozen_maps(maps), 320)
        original = Path.read_bytes
        def corrupted(path):
            data = original(path)
            return data+b' ' if path.name == 'metadata.json' and maps in path.parents else data
        with patch.object(Path, 'read_bytes', corrupted):
            with self.assertRaisesRegex(ValueError, 'Frozen map integrity'):
                verify_frozen_maps(maps)
