#!/usr/bin/env python3
"""Verify the published RQ5 bundle, reproduce its analysis, or regenerate tables."""
import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / 'experiments/RQ5'
DEFAULT_DATA = SCRIPTS / 'data'
MODELS = ('eo1', 'gr00t', 'pi0', 'spatialvla-4b')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def verify(data):
    data = data.resolve()
    entries = rows(data / 'files_manifest.csv')
    seen = set()
    total = 0
    for record in entries:
        relative = Path(record['path'])
        path = (data / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(data) or relative.as_posix() in seen:
            raise ValueError('Invalid or duplicate manifest path: ' + str(relative))
        seen.add(relative.as_posix())
        if path.suffix.lower() in ('.mp4', '.jpg', '.jpeg', '.png'):
            raise ValueError('Unexpected media file: ' + str(path))
        if path.stat().st_size != int(record['bytes']) or sha(path) != record['sha256']:
            raise ValueError('File integrity mismatch: ' + str(path))
        total += int(record['bytes'])
    actual = {p.relative_to(data).as_posix() for p in data.rglob('*')
              if p.is_file() and p.name != 'files_manifest.csv'}
    if actual != seen:
        raise ValueError('Bundle has missing or unlisted files: ' + str(sorted(actual ^ seen)[:5]))
    metadata = json.loads((data / 'scenes_504/scene_metadata_504.json').read_text())
    selection = rows(data / 'reference/selection/sample_manifest.csv')
    summaries = {}
    for model in MODELS:
        folder = data / 'move_near_504' / model
        episodes = rows(folder / 'episodes.csv')
        if len(episodes) != 504 or {e['eval_id'] for e in episodes} != {str(i) for i in range(504)}:
            raise ValueError('Expected 504 unique scene IDs for ' + model)
        if len({e['episode_id'] for e in episodes}) != 504:
            raise ValueError('Duplicate episode IDs for ' + model)
        by_id = {e['episode_id']: e for e in episodes}
        for episode in episodes:
            if episode['model'] != model or episode['failure'] not in ('0', '1'):
                raise ValueError('Invalid episode label or model')
            log = json.loads((folder / 'allMetrics' / episode['eval_id'] / 'log.json').read_text())
            outcome = log[max(log, key=int)]['success']
            if isinstance(outcome, str):
                outcome = {'true': True, 'false': False}.get(outcome.strip().lower())
            if not isinstance(outcome, bool) or int(not outcome) != int(episode['failure']):
                raise ValueError('Log/CSV mismatch: ' + episode['episode_id'])
        selected = [e for e in selection if e['model'] == model]
        if len({e['episode_id'] for e in selected}) != len(selected):
            raise ValueError('Duplicate selected episodes for ' + model)
        for e in selected:
            current = by_id[e['episode_id']]
            if (e['eval_id'], e['failure']) != (current['eval_id'], current['failure']):
                raise ValueError('Selection mismatch: ' + e['episode_id'])
        classes = Counter(e['failure'] for e in selected)
        if classes['0'] != 100 or classes['1'] < 100:
            raise ValueError('Expected 100 passes and at least 100 failures for ' + model)
        cells = Counter((metadata[e['eval_id']]['cell_x'], metadata[e['eval_id']]['cell_y'])
                        for e in selected)
        if len(cells) != 72 or max(cells.values()) - min(cells.values()) > 1:
            raise ValueError('Invalid final cell quotas for ' + model)
        config = json.loads((folder / 'run_config.json').read_text())
        for key, name in [('tasks_sha256', 't-move_near_504.json'),
                          ('prompts_sha256', 'move_near_prompts_504.json')]:
            if config[key] != sha(data / 'scenes_504' / name):
                raise ValueError('Scene/prompt fingerprint mismatch for ' + model)
        summaries[model] = {'executions': len(episodes), 'selected': len(selected),
                            'selected_pass': classes['0'], 'selected_fail': classes['1']}
    print(json.dumps({'verified_files': len(entries), 'bytes': total,
                      'models': summaries}, indent=2), flush=True)
    return summaries


def command(script, *arguments):
    subprocess.run([sys.executable, str(SCRIPTS / script),
                    *map(str, arguments)], cwd=ROOT, check=True)


def compare_csv(reference, actual, keys, numeric):
    def indexed(path):
        records = rows(path)
        index = {tuple(r[k] for k in keys): r for r in records}
        if len(index) != len(records):
            raise ValueError('Duplicate metric keys in ' + str(path))
        return index
    expected, observed = indexed(reference), indexed(actual)
    if expected.keys() != observed.keys():
        raise ValueError('Metric keys differ: ' + actual.name)
    for key, row in expected.items():
        for col in numeric:
            left, right = row[col], observed[key][col]
            if left == right:
                continue
            try:
                a, b = float(left), float(right)
            except ValueError:
                raise ValueError('Metric availability differs: ' + str((key, col)))
            if math.isnan(a) and math.isnan(b):
                continue
            if not math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError('Metric differs: ' + str((key, col, left, right)))
    return {'file': actual.name, 'rows': len(expected), 'numeric_columns': numeric}


def run(data, output):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Choose an empty reproduction output directory: ' + str(output))
    summaries = verify(data)
    output.mkdir(parents=True, exist_ok=True)
    selection, sbfl, baselines = (output / name for name in ('selection', 'sbfl', 'baselines'))
    command('rq5_select_rq1_ratio.py', '--target', data / 'move_near_504',
            '--scene-metadata', data / 'scenes_504/scene_metadata_504.json',
            '--rq1-audit', ROOT / 'experiments/RQ1/results/phase_a_primary/audit/episode_manifest.csv.gz',
            '--fixed-pass', '100', '--min-per-class', '100', '--seed', '20261028',
            '--output', selection)
    identity = lambda path: sorted((r['model'], r['episode_id'], r['eval_id'], r['failure'])
                                  for r in rows(path))
    if identity(selection / 'sample_manifest.csv') != identity(data / 'reference/selection/sample_manifest.csv'):
        raise ValueError('Regenerated selection differs from the published sample')
    command('rq5_transfer.py', 'evaluate', '--maps', data / 'reconstructed_pickup_maps',
            '--target', data / 'move_near_504', '--sample-manifest', selection / 'sample_manifest.csv',
            '--output', sbfl)
    command('rq5_baselines.py', '--sample', selection / 'sample_manifest.csv',
            '--target', data / 'move_near_504', '--output', baselines, '--sbfl-results', sbfl)
    checks = []
    for name, keys, numeric in [
        ('fold_metrics.csv', ['configuration_id', 'rq1_test_run', 'formula_id',
                              'map_metadata_sha256', 'training_source', 'current_input_sha256'],
         ['roc_auc', 'average_precision', 'mean_known_fraction', 'all_unknown_trajectories']),
        ('configuration_metrics.csv', ['vla', 'condition', 'budget', 'formula_id'],
         ['roc_auc', 'average_precision', 'mean_known_fraction', 'all_unknown_trajectories']),
        ('model_metrics.csv', ['vla', 'formula_id'],
         ['roc_auc', 'average_precision', 'mean_known_fraction', 'all_unknown_trajectories'])]:
        checks.append(compare_csv(data / 'reference/sbfl' / name, sbfl / name, keys, numeric))
    checks.append(compare_csv(data / 'reference/baselines/baseline_availability.csv',
                              baselines / 'baseline_availability.csv', ['vla', 'metric_id'],
                              ['available', 'episodes', 'availability_fraction', 'roc_auc', 'average_precision']))
    report = {'status': 'matched_reference', 'models': summaries,
              'selection_matches_reference': True, 'map_metadata_hashes_match_reference': True,
              'metric_checks': checks,
              'map_origin': 'reconstructed_from_current_pickup_using_frozen_RQ1_protocol',
              'videos_required': False, 'pickup_inputs_required': False}
    (output / 'verification_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print('RQ5 reproduction matches the published reference metrics.', flush=True)
    print('SBFL table:', sbfl / 'roc_auc_table.tex')
    print('Baselines plus top-five SBFL table:', baselines / 'baseline_top5_sbfl.tex')


def tables(data, output):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Choose an empty table output directory: ' + str(output))
    command('generate_transfer_table.py', '--input', data / 'reference/sbfl/model_metrics.csv',
            '--output-dir', output / 'sbfl')
    command('generate_baseline_sbfl_table.py', '--baseline-dir', data / 'reference/baselines',
            '--sbfl-dir', data / 'reference/sbfl', '--output-dir', output / 'baselines')
    print('Tables generated from reference metrics; no episode rescoring performed.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['verify', 'run', 'tables'])
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--output', type=Path, help='Empty output directory, required for run/tables')
    args = parser.parse_args()
    if args.command != 'verify' and args.output is None:
        parser.error('--output is required for run/tables')
    data = args.data.expanduser().resolve()
    if args.command == 'verify':
        verify(data)
    elif args.command == 'run':
        run(data, args.output.expanduser().resolve())
    else:
        tables(data, args.output.expanduser().resolve())


if __name__ == '__main__':
    main()
