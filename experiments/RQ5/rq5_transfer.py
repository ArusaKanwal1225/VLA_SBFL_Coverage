#!/usr/bin/env python3
"""Reconstruct pick-up maps with the RQ1 protocol; transfer frozen maps to move-near."""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src_new')]
import numpy as np
import pandas as pd
from experiments.RQ1.run_experiment import (discover_configurations, subset_dataset,
    discover_run_directories, trajectory_hashes, experiment_signature, validate_configuration_metadata)
from vla_sbfl.dataset import load_trajectory_dataset
from vla_sbfl.formulas import STUDY_FORMULA_IDS, FORMULA_REGISTRY
from vla_sbfl.model import SBFLModel, episode_id_hash
from vla_sbfl.scoring import TrajectoryScorer
from vla_sbfl.metrics import evaluate_predictions
from vla_sbfl.voxelizer import VoxelGrid

GRID = VoxelGrid(0.02, (-1., -1., 0.), 'robot_base')
MODEL_NAMES = {'pi0': 'Pi0', 'gr00t': 'Groot', 'eo1': 'EO1', 'spatialvla-4b': 'SpatialVLA'}
RUNS = tuple('run_%d' % i for i in range(1, 11))
RQ1 = ROOT / 'experiments/RQ1/results/phase_a_primary'
DEFAULT_TARGET = Path('/home/ubuntu/Deephyperion_VLA/results/move_near_504')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def signature():
    current, _ = experiment_signature(GRID, ROOT / 'experiments/RQ1/RQ1.md')
    record = json.loads((RQ1 / 'run.json').read_text())
    if record.get('status') != 'complete' or current != record.get('experiment_signature'):
        raise ValueError('RQ1 frozen protocol/code signature does not match completed run')
    return current


def selected_configs(input_root, models):
    wanted = {MODEL_NAMES[m] for m in models}
    return [c for c in discover_configurations(input_root) if c.vla in wanted]



def frozen_configurations(models):
    """Recover the evaluation design from the frozen audit, without pickup inputs."""
    wanted = {MODEL_NAMES[m] for m in models}
    folds = pd.read_csv(RQ1 / 'audit/fold_manifest.csv', dtype={'test_run': str})
    subset = folds[folds.vla.isin(wanted)]
    configs = []
    for cid, group in subset.groupby('configuration_id', sort=True):
        columns = ['configuration_id', 'vla', 'condition', 'budget']
        rows = group[columns].drop_duplicates()
        if len(rows) != 1 or len(group) != len(RUNS) or set(group.test_run) != set(RUNS):
            raise ValueError('Invalid frozen RQ1 configuration: ' + cid)
        row = rows.iloc[0]
        configs.append(SimpleNamespace(configuration_id=str(cid), vla=str(row.vla),
                                       condition=str(row.condition), budget=int(row.budget)))
    if any(sum(c.vla == vla for c in configs) != 8 for vla in wanted):
        raise ValueError('Expected eight frozen RQ1 configurations per requested model')
    return configs


def map_provenance_report(needed):
    """Report the origins of the actual map files, without changing their bytes."""
    configs = {}
    for config, run, folder in needed:
        raw = json.loads((folder / 'metadata.json').read_text())
        training = raw['training']
        record = {'configuration_id': config.configuration_id, 'vla': config.vla,
                  'training_source': training.get('training_source', 'rq1_original_artifact'),
                  'current_input_sha256': training.get('current_input_sha256'),
                  'historical_audit_comparison': training.get('historical_audit_comparison')}
        previous = configs.setdefault(config.configuration_id, record)
        if previous != record:
            raise ValueError('Inconsistent map origins within ' + config.configuration_id)
    sources = sorted({r['training_source'] for r in configs.values()})
    reconstructed = sources == ['current_sbfl_pickup_input']
    return {'map_count': len(needed), 'configuration_count': len(configs),
            'training_sources': sources, 'reconstructed_from_current_pickup': reconstructed,
            'historical_original_map_identity_verified': False if reconstructed else None,
            'target_task_training_or_calibration': False,
            'interpretation': ('Maps reconstructed from current pickup inputs using the frozen RQ1 protocol; '
                               'historical original map files were not reused.' if reconstructed
                               else 'Consult per-configuration training sources.'),
            'configurations': list(configs.values())}


def compare_training_audit(dataset, config, audit, hashes):
    """Report historical discrepancies; the current input files are authoritative."""
    reference = audit[audit.configuration_id == config.configuration_id].copy()
    actual = dataset.episodes[['episode_id', 'run_id', 'eval_id', 'failure', 'object_name', 'num_steps']].copy()
    actual['trajectory_sha256'] = actual.episode_id.map(hashes)
    cols = ['episode_id', 'run_id', 'eval_id', 'failure', 'object_name', 'num_steps', 'trajectory_sha256']
    reference = reference[cols].astype(str).set_index('episode_id')
    actual = actual[cols].astype(str).set_index('episode_id')
    joined = reference.join(actual, how='outer', lsuffix='_audit', rsuffix='_current')
    differing = []
    for name in cols[1:]:
        different = joined[name + '_audit'].fillna('<missing>') != joined[name + '_current'].fillna('<missing>')
        differing.extend(joined.index[different].tolist())
    changed = sorted(set(differing))
    return {'reference_episodes': len(reference), 'current_episodes': len(actual),
            'different_episodes': len(changed), 'example_episode_ids': changed[:10]}


def input_fingerprint(configuration):
    """Fingerprint the current pickup CSVs so reused maps cannot silently go stale."""
    files = []
    for directory in discover_run_directories([configuration.directory]):
        for name in ('episodes.csv', 'trajectories.csv'):
            path = directory / name
            files.append({'path': str(path.relative_to(configuration.directory)), 'sha256': sha(path)})
    payload = json.dumps(files, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def prepare(args):
    sig = signature()
    audit = pd.read_csv(RQ1 / 'audit/episode_manifest.csv.gz', dtype={'episode_id': str})
    for c in selected_configs(args.input, args.models):
        print('Loading and auditing RQ1:', c.configuration_id, flush=True)
        dataset = load_trajectory_dataset([c.directory], GRID, require_labels=True)
        validate_configuration_metadata(c, dataset.episodes)
        hashes, info = trajectory_hashes(discover_run_directories([c.directory]))
        if info['duplicate_episode_steps']:
            raise ValueError('Duplicate trajectory steps: ' + c.configuration_id)
        comparison = compare_training_audit(dataset, c, audit, hashes)
        fingerprint = input_fingerprint(c)
        if comparison['different_episodes']:
            print('Historical RQ1 audit differs for {} episodes in {}; using current SBFL input.'.format(
                comparison['different_episodes'], c.configuration_id), flush=True)
        for run in RUNS:
            train = subset_dataset(dataset, dataset.episodes.loc[dataset.episodes.run_id.astype(str) != run, 'episode_id'])
            dest = args.maps / 'folds' / c.configuration_id / run / 'model'
            if (dest / 'metadata.json').exists():
                model = SBFLModel.load(dest)
                if model.metadata.get('experiment_signature') != sig or model.metadata.get('training_episode_ids_sha256') != episode_id_hash(tuple(train.labels)) or model.metadata.get('current_input_sha256') != fingerprint:
                    raise ValueError('Existing model provenance mismatch: ' + str(dest))
                continue
            if dest.exists() and any(dest.iterdir()):
                raise FileExistsError('Incomplete map folder; inspect before retrying: ' + str(dest))
            model = SBFLModel.fit(train.visits, train.labels, GRID, STUDY_FORMULA_IDS,
                {'experiment': 'RQ1_phase_a_primary', 'experiment_signature': sig,
                 'configuration_id': c.configuration_id, 'vla': c.vla, 'condition': c.condition,
                 'budget': c.budget, 'test_run': run,
                 'training_source': 'current_sbfl_pickup_input', 'current_input_sha256': fingerprint,
                 'historical_audit_comparison': comparison})
            model.save(dest)
            print('Saved', dest, flush=True)


def checked_success(value, path):
    if isinstance(value, str): value = {'true': True, 'false': False}.get(value.strip().lower())
    if not isinstance(value, bool): raise ValueError('Invalid success in ' + str(path))
    return value


def select_target(path, model, seed, sample_manifest=None):
    data = load_trajectory_dataset([path], GRID, require_labels=True)
    ep = data.episodes.copy()
    if len(ep) != 504 or set(pd.to_numeric(ep.eval_id)) != set(range(504)):
        raise ValueError(f'{model}: require all 504 completed final scenes before sampling; found {len(ep)}')
    if set(ep.model.astype(str)) != {model} or ep.eval_id.duplicated().any():
        raise ValueError(f'{model}: invalid model or duplicated scene ID')
    for row in ep.itertuples():
        logpath = path / 'allMetrics' / str(int(row.eval_id)) / 'log.json'
        log = json.loads(logpath.read_text())
        success = checked_success(log[max(log, key=int)]['success'], logpath)
        if int(not success) != int(row.failure):
            raise ValueError('CSV/log label mismatch: ' + str(logpath))
    counts = ep.failure.value_counts().to_dict()
    if sample_manifest is None:
        if min(counts.get(0, 0), counts.get(1, 0)) < 100:
            raise ValueError(f'{model}: need at least 100 pass and 100 fail; found {counts}')
        # Independent RNG per model; outcome-stratified sampling without replacement.
        model_seed = int.from_bytes(hashlib.sha256(f'{seed}:{model}'.encode()).digest()[:8], 'big')
        rng = random.Random(model_seed)
        chosen = []
        for label in (0, 1):
            ids = sorted(ep.loc[ep.failure == label, 'episode_id'].astype(str))
            chosen.extend(rng.sample(ids, 100))
    else:
        required = {'model', 'episode_id', 'eval_id', 'failure'}
        if not required <= set(sample_manifest.columns): raise ValueError('Sample manifest is missing required columns')
        sample = sample_manifest[sample_manifest.model.astype(str) == model].copy()
        if sample.empty or sample.episode_id.isna().any() or sample.episode_id.astype(str).duplicated().any():
            raise ValueError('Empty or duplicate sample for ' + model)
        sample['episode_id'] = sample.episode_id.astype(str)
        sample['failure'] = pd.to_numeric(sample.failure, errors='coerce')
        sample['eval_id'] = pd.to_numeric(sample.eval_id, errors='coerce')
        if sample[['failure','eval_id']].isna().any().any() or set(sample.failure) != {0, 1}:
            raise ValueError('Sample must contain valid IDs and both outcome classes for ' + model)
        check = sample[['episode_id','failure','eval_id']].merge(ep[['episode_id','failure','eval_id']],
                on='episode_id', how='left', suffixes=('_sample','_input'), validate='one_to_one')
        if check[['failure_input','eval_id_input']].isna().any().any() or not (check.failure_sample == check.failure_input).all() or not (check.eval_id_sample == check.eval_id_input).all():
            raise ValueError('Sample manifest disagrees with target episodes for ' + model)
        chosen = sample.episode_id.tolist()
    return subset_dataset(data, chosen), counts


def evaluate(args):
    from experiments.RQ5.provenance import verify_compatibility, verify_frozen_maps
    sig = verify_compatibility()
    verify_frozen_maps(args.maps)
    configs = frozen_configurations(args.models)
    needed = [(c, run, args.maps / 'folds' / c.configuration_id / run / 'model') for c in configs for run in RUNS]
    missing = [str(path) for _, _, path in needed if not (path / 'metadata.json').is_file()]
    if missing:
        raise FileNotFoundError(f'{len(missing)} RQ1 map artifacts missing. Run prepare first; example: {missing[0]}')
    provenance = map_provenance_report(needed)
    datasets, counts = {}, {}
    supplied_sample = None
    if args.sample_manifest is not None:
        supplied_sample = pd.read_csv(args.sample_manifest, dtype={'episode_id': str, 'model': str})
        if 'model' not in supplied_sample.columns or not set(args.models) <= set(supplied_sample.model.astype(str)):
            raise ValueError('Sample manifest lacks a selected model')
    configs_hash = {}
    for model in args.models:
        path = args.target / model
        config = json.loads((path / 'run_config.json').read_text())
        if config.get('model') != model or config.get('task') != 'google_robot_move_near_customizable':
            raise ValueError('Incorrect target run configuration: ' + model)
        configs_hash[model] = tuple(config.get(k) for k in ('task', 'seed', 'tasks_sha256', 'prompts_sha256'))
        if any(x is None for x in configs_hash[model]): raise ValueError('Incomplete target run configuration: '+model)
        datasets[model], counts[model] = select_target(path, model, args.seed, supplied_sample)
    if len(set(configs_hash.values())) != 1:
        raise ValueError('Target scenes, prompts, task or seed differ across models')
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Output directory is not empty: ' + str(args.output))
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = pd.concat([d.episodes.assign(target_model=m) for m, d in datasets.items()], ignore_index=True)
    manifest.drop(columns=['_run_directory'], errors='ignore').to_csv(args.output / 'sample_manifest.csv', index=False)
    metadata = {'seed': args.seed if supplied_sample is None else None, 'sample_per_class': 100 if supplied_sample is None else None, 'target_class_counts': counts,
                'source_sample_manifest': str(args.sample_manifest.resolve()) if supplied_sample is not None else None,
                'source_sample_sha256': sha(args.sample_manifest) if supplied_sample is not None else None,
                'selected_class_counts': {m: {str(k): int(v) for k, v in d.episodes.failure.value_counts().items()} for m, d in datasets.items()},
                'target_signatures': configs_hash, 'rq1_signature': sig,
                'compatibility_verification': {
                    'historical_signature_verified': True,
                    'frozen_artifact_integrity_verified': True,
                    'scientific_code_contract_verified': True,
                    'contract_sha256': sha(Path(__file__).with_name('compatibility_contract.json')),
                },
                'grid': GRID.to_dict(), 'models': args.models,
                'map_provenance_file': 'map_provenance.json', 'training_sources': provenance['training_sources'],
                'selection': ('100 per outcome per model without replacement' if supplied_sample is None else 'provided sample manifest') + '; same sample for every RQ1 map and formula'}
    (args.output / 'analysis_config.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (args.output / 'map_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    folds = pd.read_csv(RQ1 / 'audit/fold_manifest.csv', dtype={'test_run': str})
    metrics, predictions = [], []
    for c, run, path in needed:
        model = SBFLModel.load(path)
        if model.grid != GRID or tuple(model.formula_ids) != tuple(STUDY_FORMULA_IDS):
            raise ValueError('Map grid or formulas differ from RQ1: ' + str(path))
        if model.metadata.get('experiment_signature') != sig or model.metadata.get('configuration_id') != c.configuration_id or model.metadata.get('test_run') != run:
            raise ValueError('Map provenance mismatch: ' + str(path))
        expected = folds[(folds.configuration_id == c.configuration_id) & (folds.test_run == run)]
        if len(expected) != 1:
            raise ValueError('Missing frozen RQ1 fold metadata: ' + str(path))
        if model.metadata.get('training_source') != 'current_sbfl_pickup_input':
            if episode_id_hash(model.training_episode_ids) != expected.iloc[0].training_episode_ids_sha256:
                raise ValueError('Map training episode IDs differ from frozen RQ1 fold: ' + str(path))
        elif not model.metadata.get('current_input_sha256'):
            raise ValueError('Current-input map lacks its input fingerprint: ' + str(path))
        target_model = next(k for k,v in MODEL_NAMES.items() if v == c.vla)
        target = datasets[target_model]
        if set(model.training_episode_ids) & set(target.labels): raise ValueError('Train/target episode IDs overlap')
        pred = TrajectoryScorer(model).score(target.visits, target.episodes)
        pred.insert(0, 'configuration_id', c.configuration_id); pred.insert(1, 'rq1_test_run', run)
        pred.insert(2, 'vla', c.vla)
        result = evaluate_predictions(pred, model.formula_ids)
        result.insert(0, 'configuration_id', c.configuration_id); result.insert(1, 'vla', c.vla)
        result.insert(2, 'condition', c.condition); result.insert(3, 'budget', c.budget); result.insert(4, 'rq1_test_run', run)
        result['technique'] = result.formula_id.map({f: FORMULA_REGISTRY[f].name for f in model.formula_ids})
        result['mean_known_fraction'] = pred.known_fraction.mean()
        result['all_unknown_trajectories'] = int((pred.scoring_status == 'no_training_support').sum())
        result['map_metadata_sha256'] = sha(path / 'metadata.json')
        result['training_source'] = model.metadata.get('training_source', 'rq1_original_artifact')
        result['current_input_sha256'] = model.metadata.get('current_input_sha256', '')
        result['historical_audit_different_episodes'] = model.metadata.get('historical_audit_comparison', {}).get('different_episodes', '')
        metrics.append(result); predictions.append(pred.drop(columns=['_run_directory'], errors='ignore'))
        print('Evaluated', c.configuration_id, run, flush=True)
    result = pd.concat(metrics, ignore_index=True)
    result.to_csv(args.output / 'fold_metrics.csv', index=False)
    pd.concat(predictions, ignore_index=True).to_csv(args.output / 'predictions.csv.gz', index=False, compression='gzip')
    means = result.groupby(['vla','condition','budget','formula_id','technique'], as_index=False).agg(
        roc_auc=('roc_auc','mean'), average_precision=('average_precision','mean'),
        mean_known_fraction=('mean_known_fraction','mean'), all_unknown_trajectories=('all_unknown_trajectories','mean'))
    means.to_csv(args.output / 'configuration_metrics.csv', index=False)
    summary = means.groupby(['vla','formula_id','technique'], as_index=False).agg(
        roc_auc=('roc_auc','mean'), average_precision=('average_precision','mean'),
        mean_known_fraction=('mean_known_fraction','mean'), all_unknown_trajectories=('all_unknown_trajectories','mean'))
    summary.sort_values(['vla','roc_auc','formula_id'], ascending=[True,False,True]).to_csv(args.output / 'model_metrics.csv', index=False)
    from generate_transfer_table import generate
    generate(args.output / 'model_metrics.csv', args.output)
    print('Results written to', args.output, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('prepare','evaluate'):
        p = sub.add_parser(command)
        p.add_argument('--input', type=Path, default=ROOT/'input', help='Pick-up input root for prepare; evaluate uses the frozen fold audit')
        p.add_argument('--maps', type=Path, required=True, help='RQ1 map artifact root')
        p.add_argument('--models', choices=tuple(MODEL_NAMES), nargs='+', default=list(MODEL_NAMES))
        if command == 'evaluate':
            p.add_argument('--target', type=Path, default=DEFAULT_TARGET)
            p.add_argument('--output', type=Path, required=True)
            p.add_argument('--seed', type=int, default=20261028)
            p.add_argument('--sample-manifest', type=Path, help='Use a fixed sample rather than selecting 100 per class')
    args = parser.parse_args()
    if len(set(args.models)) != len(args.models): parser.error('duplicate models')
    (prepare if args.command == 'prepare' else evaluate)(args)

if __name__ == '__main__': main()
