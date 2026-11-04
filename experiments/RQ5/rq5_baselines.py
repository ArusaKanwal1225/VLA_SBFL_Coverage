#!/usr/bin/env python3
"""Evaluate the frozen RQ2 baseline catalogue on the existing RQ5 sample."""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src_new')]
import pandas as pd
from experiments.RQ2_baselines.catalogue import METRIC_SPECS
from experiments.RQ2_baselines.run_experiment import load_metric_score
from experiments.RQ1.generate_tables import tie_aware_ranks, latex_escape
from vla_sbfl.metrics import roc_auc, average_precision

MODELS = [('eo1', 'EO1'), ('gr00t', 'GR00T'), ('pi0', 'Pi0'), ('spatialvla-4b', 'SpatialVLA')]
DEFAULT_ROOT = Path('/home/ubuntu/Deephyperion_VLA/results')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def load_manifest(path, target):
    frame = pd.read_csv(path, dtype={'episode_id': str, 'model': str, 'eval_id': str})
    required = {'model', 'episode_id', 'eval_id', 'failure'}
    if frame.empty or not required <= set(frame.columns) or frame.episode_id.duplicated().any():
        raise ValueError('Invalid or duplicate RQ5 sample manifest')
    frame['failure'] = pd.to_numeric(frame.failure, errors='raise')
    if not frame.failure.isin([0, 1]).all(): raise ValueError('Invalid failure labels')
    known = {m for m, _ in MODELS}
    if not set(frame.model) <= known: raise ValueError('Unknown model in manifest')
    models = [(m, label) for m, label in MODELS if m in set(frame.model)]
    for model, _ in models:
        sample = frame[frame.model == model]
        if set(sample.failure) != {0, 1}:
            raise ValueError(model + ': sample must contain pass and fail episodes')
        if sample.eval_id.duplicated().any() or not sample.eval_id.str.fullmatch(r'\d+').all():
            raise ValueError(model + ': invalid or duplicate scene IDs')
        episodes = pd.read_csv(target / model / 'episodes.csv', dtype={'episode_id': str, 'eval_id': str})
        if episodes.episode_id.duplicated().any() or episodes.eval_id.duplicated().any():
            raise ValueError(model + ': duplicate episode or scene in current CSV')
        current = episodes.set_index('episode_id')
        for row in sample.itertuples():
            if row.episode_id not in current.index: raise ValueError('Sample episode missing: ' + row.episode_id)
            ref = current.loc[row.episode_id]
            if str(ref.eval_id) != str(row.eval_id) or int(ref.failure) != int(row.failure):
                raise ValueError('Sample disagrees with current result: ' + row.episode_id)
            logpath = target / model / 'allMetrics' / row.eval_id / 'log.json'
            log = json.loads(logpath.read_text(encoding='utf-8'))
            success = log[max(log, key=int)]['success']
            if isinstance(success, str): success = {'true': True, 'false': False}.get(success.strip().lower())
            if not isinstance(success, bool) or int(not success) != int(row.failure):
                raise ValueError('Sample label disagrees with final log: ' + str(logpath))
    return frame, models


def render_tables(results, models, output):
    complete = results[results.complete_all_models].copy()
    if complete.empty:
        raise ValueError('No baseline is available for every selected episode and model')
    keys = [label for _, label in models]
    pivot = complete.pivot(index=['metric_id', 'method', 'family', 'task_aware'], columns='vla', values='roc_auc').reset_index()
    pivot['Overall'] = pivot[keys].mean(axis=1)
    ranks = tie_aware_ranks(dict(zip(pivot.metric_id, pivot.Overall)))
    pivot['Rank'] = pivot.metric_id.map(ranks)
    pivot = pivot.sort_values(['Overall', 'metric_id'], ascending=[False, True])
    pivot.to_csv(output/'baseline_performance.csv', index=False, float_format='%.12g')
    columns = [*keys, 'Overall']
    best = {col: float(pivot[col].max()) for col in columns}
    second = {}
    for col in columns:
        vals = sorted(pivot[col].unique(), reverse=True)
        second[col] = next((float(v) for v in vals if not math.isclose(v, best[col], rel_tol=1e-12, abs_tol=1e-12)), None)
    b = chr(92); partial = len(models) != 4
    caption = ('RQ5 baseline trajectory-level failure-discrimination ROC-AUC on the selected RQ5 sample. '
               'Overall is the equal-weight mean across the ' + ('evaluated models only (partial result)' if partial else 'four models') +
               '. Bold denotes the best value and underlining the second-best value per column. '
               '$^{'+b+'dagger}$ indicates a task-aware metric using object/goal state.')
    lines = [b+'begin{table*}[t]', b+'centering', b+'caption{'+caption+'}', b+'label{tab:rq5-baseline-performance}',
             '{'+b+'scriptsize', b+'setlength{'+b+'tabcolsep}{4pt}', b+'begin{tabular}{l'+'c'*(len(columns)+1)+'}',
             b+'toprule', 'Method & '+' & '.join(columns)+' & Rank '+b*2, b+'midrule']
    for _, row in pivot.iterrows():
        label = latex_escape(row['method']) + (('$^{'+b+'dagger}$') if row['task_aware'] else '')
        cells = []
        for col in columns:
            value = float(row[col]); text = f'{value:.4f}'
            if math.isclose(value, best[col], rel_tol=1e-12, abs_tol=1e-12): text = b+'textbf{'+text+'}'
            elif second[col] is not None and math.isclose(value, second[col], rel_tol=1e-12, abs_tol=1e-12): text = b+'underline{'+text+'}'
            cells.append(text)
        lines.append(label+' & '+' & '.join(cells)+' & '+str(int(row['Rank']))+' '+b*2)
    lines += [b+'bottomrule', b+'end{tabular}', '}', b+'end{table*}']
    (output/'baseline_performance.tex').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample', type=Path, default=DEFAULT_ROOT/'rq5_transfer_final/sample_manifest.csv')
    parser.add_argument('--target', type=Path, default=DEFAULT_ROOT/'move_near_504')
    parser.add_argument('--output', type=Path, default=DEFAULT_ROOT/'rq5_baselines_final')
    parser.add_argument('--sbfl-results', type=Path, help='Optionally add top-five SBFL methods from a transfer result using the same sample')
    args = parser.parse_args()
    frame, models = load_manifest(args.sample, args.target)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Output directory is not empty: ' + str(args.output))
    args.output.mkdir(parents=True, exist_ok=True)
    detail, summary = [], []
    for model, vla in models:
        sample = frame[frame.model == model].sort_values('episode_id')
        for spec in METRIC_SPECS:
            entries = []
            for row in sample.itertuples():
                path = args.target / model / 'allMetrics' / row.eval_id / spec.filename
                score, count, status = load_metric_score(path, spec.direction)
                entry = {'model': model, 'vla': vla, 'eval_id': row.eval_id, 'episode_id': row.episode_id,
                         'failure': int(row.failure), 'metric_id': spec.metric_id, 'score': score,
                         'value_count': count, 'status': status}
                entries.append(entry); detail.append(entry)
            valid = [e for e in entries if e['status'] == 'available']
            row = {'vla': vla, 'metric_id': spec.metric_id, 'method': spec.name, 'family': spec.family,
                   'task_aware': spec.task_aware, 'available': len(valid), 'episodes': len(entries),
                   'availability_fraction': len(valid)/len(entries), 'roc_auc': None, 'average_precision': None}
            if len(valid) == len(entries):
                row['roc_auc'] = roc_auc([e['failure'] for e in valid], [e['score'] for e in valid])
                row['average_precision'] = average_precision([e['failure'] for e in valid], [e['score'] for e in valid])
            summary.append(row)
    availability = pd.DataFrame(summary)
    complete_ids = set(availability.groupby('metric_id').filter(lambda g: len(g) == len(models) and (g.availability_fraction == 1).all()).metric_id)
    availability['complete_all_models'] = availability.metric_id.isin(complete_ids)
    pd.DataFrame(detail).to_csv(args.output/'baseline_scores.csv', index=False)
    availability.to_csv(args.output/'baseline_availability.csv', index=False)
    render_tables(availability, models, args.output)
    config = {'sample_file': str(args.sample.resolve()), 'sample_sha256': digest(args.sample),
              'models': [m for m, _ in models], 'episodes_per_model': {m: int(len(frame[frame.model == m])) for m,_ in models},
              'selected_class_counts': {m: {str(k): int(v) for k,v in frame[frame.model == m].failure.value_counts().items()} for m,_ in models}, 'failure_is_positive': True,
              'aggregation': 'arithmetic mean of numeric JSON leaves; fixed direction from RQ2 catalogue',
              'missing_policy': 'no episode removal; metric requires availability for every selected episode in every selected model',
              'complete_baselines': sorted(complete_ids), 'metric_catalogue': [s.to_dict() for s in METRIC_SPECS],
              'note': 'Fixed evaluation sample; no additional baseline fitting or target calibration.'}
    (args.output/'analysis_config.json').write_text(json.dumps(config, indent=2)+'\n')
    if args.sbfl_results is not None:
        from generate_baseline_sbfl_table import generate
        generate(args.output, args.sbfl_results, args.output)
    print('Complete baselines:', ', '.join(sorted(complete_ids)))
    print('Results:', args.output.resolve())


if __name__ == '__main__': main()
