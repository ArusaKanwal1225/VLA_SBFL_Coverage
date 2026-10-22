#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RQ1 Step 04 - deterministic TCP XYZ voxel coverage, Pi0, 200 scenes.

Run from a repository containing input/, src/, and output/:
    python src/04_rq1_voxelize_pi0_200.py

Requirements: numpy, pandas, matplotlib; Python 3.9+.

Input (created by prior extraction and split scripts):
    input/Pi0/<algorithm>/run_<1..10>/episodes.csv
    input/Pi0/<algorithm>/run_<1..10>/trajectories.csv
    output/RQ1/pi0_200/step03_grouped_splits/split_manifest.csv

Outputs in output/RQ1/pi0_200/step04_voxel_coverage/:
    episode_voxel_visits.csv.gz              1 row per episode/visited voxel
    per_episode_voxel_statistics.csv         counts and trajectory lengths
    algorithm_voxel_summary.csv              overall descriptive counts
    split_voxel_coverage.csv                 per-algorithm/seed train/test coverage
    voxelization_settings.json               fixed grid and reproducibility metadata
    validation_report.json                   data integrity status
    figures/*.png, *.pdf                     plots saved with CSV source values

Important: overall-algorithm statistics are DESCRIPTIVE ONLY. All training
quantities for SBFL must be learned ONLY from the training IDs in the frozen
split manifest. Do not compute suspiciousness from all 8,000 episodes.
"""

from __future__ import annotations
import argparse
import csv
import gzip
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ALGORITHMS = (
    'DH_200scenes', 'DH_HybridDT_200scenes',
    'DH_HybridRF_200scenes', 'DH_WeightedRF_200scenes'
)
SEEDS = (11, 23, 37, 41, 53, 67, 79, 83, 97, 101)
SPLIT_KEY = 'scene_group_exact_candidate'


def save_csv(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data).to_csv(path, index=False, encoding='utf-8')


def save_figure(fig, directory, name):
    directory.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(directory / (name + '.png'), dpi=200, bbox_inches='tight')
    fig.savefig(directory / (name + '.pdf'), bbox_inches='tight')
    plt.close(fig)


def load_run(root, algorithm, run, size, origin):
    base = root / 'input' / 'Pi0' / algorithm / run
    ep_path = base / 'episodes.csv'
    tr_path = base / 'trajectories.csv'
    if not ep_path.is_file() or not tr_path.is_file():
        raise ValueError('Missing run data: ' + str(base))

    with ep_path.open('r', encoding='utf-8-sig', newline='') as fh:
        reader = csv.DictReader(fh)
        required = {'episode_id', 'run_id', 'failure', 'num_steps'}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f'{ep_path}: missing {required - set(reader.fieldnames or ())}')
        episodes = {}
        for row in reader:
            eid = row['episode_id']
            if not eid or eid in episodes:
                raise ValueError(f'{ep_path}: blank or duplicated episode_id {eid!r}')
            if row['run_id'] != run or row['failure'] not in ('0', '1'):
                raise ValueError(f'{ep_path}: invalid run/outcome for {eid}')
            n_steps = int(row['num_steps'])
            if n_steps < 1:
                raise ValueError(f'{ep_path}: empty episode {eid}')
            episodes[eid] = {'label': row['failure'], 'n_steps': n_steps,
                             'run_id': run, 'voxels': set(), 'steps': set(),
                             'length_m': 0.0, 'last_xyz': None}

    if len(episodes) != 200:
        raise ValueError(f'{base}: expected 200 episodes, got {len(episodes)}')

    with tr_path.open('r', encoding='utf-8-sig', newline='') as fh:
        reader = csv.DictReader(fh)
        required = {'episode_id', 'step', 'x', 'y', 'z'}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f'{tr_path}: missing {required - set(reader.fieldnames or ())}')
        for row in reader:
            eid = row['episode_id']
            if eid not in episodes:
                raise ValueError(f'{tr_path}: unknown episode_id: {eid}')
            ep = episodes[eid]
            step = int(row['step'])
            if step < 0 or step in ep['steps']:
                raise ValueError(f'{tr_path}: duplicate/negative step for {eid}: {step}')
            ep['steps'].add(step)
            xyz = (float(row['x']), float(row['y']), float(row['z']))
            if not all(math.isfinite(x) for x in xyz):
                raise ValueError(f'{tr_path}: non-finite XYZ for {eid}, step={step}')
            idx = tuple(math.floor((xyz[i] - origin[i]) / size) for i in range(3))
            ep['voxels'].add(idx)
            # The extracted rows are expected in step order; verify order explicitly.
            if ep['last_xyz'] is not None:
                previous_step, previous_xyz = ep['last_xyz']
                if step <= previous_step:
                    raise ValueError(f'{tr_path}: steps out of order for {eid}')
                ep['length_m'] += math.dist(xyz, previous_xyz)
            ep['last_xyz'] = (step, xyz)

    for eid, ep in episodes.items():
        if len(ep['steps']) != ep['n_steps'] or ep['steps'] != set(range(ep['n_steps'])):
            raise ValueError(f'{base}: missing/extra trajectory rows for {eid}')
        if len(ep['voxels']) == 0:
            raise ValueError(f'{base}: no visited voxels for {eid}')
        del ep['steps']
        del ep['last_xyz']
    return episodes


def load_validated_splits(root, by_alg):
    path = root / 'output' / 'RQ1' / 'pi0_200' / 'step03_grouped_splits' / 'split_manifest.csv'
    if not path.is_file():
        raise ValueError('Missing frozen splits: ' + str(path))
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding='utf-8-sig')
    need = {'algorithm', 'seed', 'episode_id', 'run_id', 'failure', 'split', SPLIT_KEY}
    if not need.issubset(df.columns):
        raise ValueError('Split manifest missing columns: ' + str(need - set(df.columns)))
    if len(df) != 80000:
        raise ValueError(f'Expected 80000 split records (4 x 10 x 2000), got {len(df)}')
    if df[['algorithm','seed','episode_id']].duplicated().any():
        raise ValueError('Duplicate episode within an algorithm/seed split')
    if not df.split.isin(['train','test']).all():
        raise ValueError('Unexpected split label')
    for alg in ALGORITHMS:
        if alg not in by_alg:
            raise ValueError('Missing algorithm from trajectory data: ' + alg)
        epids = set(by_alg[alg])
        a = df[df.algorithm == alg]
        if len(a) != 20000:
            raise ValueError(f'{alg}: expected 20000 split rows')
        for seed in SEEDS:
            s = a[a.seed == str(seed)]
            if len(s) != 2000 or set(s.episode_id) != epids:
                raise ValueError(f'{alg}/{seed}: episodes differ from extracted input')
            for row in s.itertuples(index=False):
                info = by_alg[alg][row.episode_id]
                if row.failure != info['label'] or row.run_id != info['run_id']:
                    raise ValueError(f'{alg}/{seed}: label/run mismatch for {row.episode_id}')
            train = s[s.split == 'train']
            test = s[s.split == 'test']
            if not (580 <= len(test) <= 620 and len(train) + len(test) == 2000):
                raise ValueError(f'{alg}/{seed}: invalid split sizes')
            if train.failure.nunique() < 2 or test.failure.nunique() < 2:
                raise ValueError(f'{alg}/{seed}: one-class train/test set')
            if set(train[SPLIT_KEY]).intersection(set(test[SPLIT_KEY])):
                raise ValueError(f'{alg}/{seed}: overlapping scene groups')
    if set(df.algorithm) != set(ALGORITHMS):
        raise ValueError('Unexpected set of algorithms in split manifest')
    if set(df.seed) != set(map(str, SEEDS)):
        raise ValueError('Unexpected seed set')
    return df, path


def produce_outputs(root, size, origin):
    by_alg = {}
    episode_rows = []
    algorithm_rows = []
    for alg in ALGORITHMS:
        merged = {}
        for n in range(1, 11):
            run = f'run_{n}'
            run_eps = load_run(root, alg, run, size, origin)
            if set(merged).intersection(run_eps):
                raise ValueError(f'{alg}: duplicate globally unique episode_id across runs')
            merged.update(run_eps)
            print(f'[CHECKED] {alg}/{run}: {len(run_eps)} episodes')
        if len(merged) != 2000:
            raise ValueError(f'{alg}: expected 2000 episodes, got {len(merged)}')
        by_alg[alg] = merged
        all_vox = set().union(*(e['voxels'] for e in merged.values()))
        failures = sum(int(e['label']) for e in merged.values())
        algorithm_rows.append({
            'algorithm': alg, 'episodes': len(merged),
            'failures': failures, 'successes': len(merged) - failures,
            'unique_voxels_full_data_descriptive_only': len(all_vox),
            'mean_unique_voxels_per_episode': float(np.mean([len(e['voxels']) for e in merged.values()])),
            'median_unique_voxels_per_episode': float(np.median([len(e['voxels']) for e in merged.values()])),
        })
        for eid, ep in merged.items():
            episode_rows.append({
                'algorithm': alg, 'run_id': ep['run_id'], 'episode_id': eid,
                'failure': int(ep['label']), 'sample_count': ep['n_steps'],
                'unique_voxel_count': len(ep['voxels']),
                'tcp_path_length_m': ep['length_m'],
            })
    print('All 8000 trajectories validated. Loading existing Step 03 split manifest...')
    split_df, split_path = load_validated_splits(root, by_alg)

    out = root / 'output' / 'RQ1' / 'pi0_200' / 'step04_voxel_coverage'
    figs = out / 'figures'
    out.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    # A deterministic, reusable, binary episode-voxel spectrum. This file
    # does not include any label-derived suspiciousness score.
    spectrum_path = out / 'episode_voxel_visits.csv.gz'
    with gzip.open(spectrum_path, 'wt', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['algorithm', 'episode_id', 'voxel_i', 'voxel_j', 'voxel_k'])
        for alg in ALGORITHMS:
            for eid in sorted(by_alg[alg]):
                for i, j, k in sorted(by_alg[alg][eid]['voxels']):
                    writer.writerow([alg, eid, i, j, k])

    split_rows = []
    for alg in ALGORITHMS:
        a = split_df[split_df.algorithm == alg]
        for seed in SEEDS:
            s = a[a.seed == str(seed)]
            train_ids = list(s.loc[s.split == 'train', 'episode_id'])
            test_ids = list(s.loc[s.split == 'test', 'episode_id'])
            train_vox = set().union(*(by_alg[alg][eid]['voxels'] for eid in train_ids))
            test_vox = set().union(*(by_alg[alg][eid]['voxels'] for eid in test_ids))
            common = train_vox & test_vox
            test_visits = sum(len(by_alg[alg][eid]['voxels']) for eid in test_ids)
            known_test_visits = sum(len(by_alg[alg][eid]['voxels'] & train_vox) for eid in test_ids)
            train_binary_visits = sum(len(by_alg[alg][eid]['voxels']) for eid in train_ids)
            split_rows.append({
                'algorithm': alg, 'seed': seed,
                'train_episodes': len(train_ids), 'test_episodes': len(test_ids),
                'train_unique_voxels': len(train_vox),
                'test_unique_voxels': len(test_vox),
                'shared_train_test_voxels': len(common),
                'test_only_unique_voxels': len(test_vox - train_vox),
                'test_unique_voxel_known_fraction': len(common) / len(test_vox) if test_vox else 0.0,
                'test_episode_voxel_visit_known_fraction': known_test_visits / test_visits if test_visits else 0.0,
                'train_binary_episode_voxel_visits': train_binary_visits,
                'test_binary_episode_voxel_visits': test_visits,
            })
            print(f'[COVERAGE] {alg} seed={seed}: train_vox={len(train_vox)} '
                  f'test_vox={len(test_vox)} test_known={len(common)/len(test_vox):.3f}')
    per_episode = pd.DataFrame(episode_rows)
    per_algorithm = pd.DataFrame(algorithm_rows)
    by_split = pd.DataFrame(split_rows)
    save_csv(per_episode, out / 'per_episode_voxel_statistics.csv')
    save_csv(per_algorithm, out / 'algorithm_voxel_summary.csv')
    save_csv(by_split, out / 'split_voxel_coverage.csv')

    labels = [a.replace('_200scenes','') for a in ALGORITHMS]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    xs = np.arange(len(ALGORITHMS))
    ax.bar(xs, per_algorithm.unique_voxels_full_data_descriptive_only)
    ax.set_xticks(xs, labels, rotation=15)
    ax.set_ylabel('Distinct visited voxels (all episodes; descriptive only)')
    ax.set_title('Pi0 200 - overall spatial exploration by algorithm')
    save_figure(fig, figs, '01_total_distinct_voxels_descriptive')

    fig, ax = plt.subplots(figsize=(9, 4.8))
    grouped = [per_episode.loc[per_episode.algorithm == a, 'unique_voxel_count'].to_numpy()
               for a in ALGORITHMS]
    try:
        ax.boxplot(grouped, tick_labels=labels, showfliers=False)
    except TypeError:
        ax.boxplot(grouped, labels=labels, showfliers=False)
    ax.tick_params(axis='x', labelrotation=15)
    ax.set_ylabel('Distinct voxels visited by one episode')
    ax.set_title('Pi0 200 - voxel visits per episode')
    save_figure(fig, figs, '02_voxels_per_episode')

    fig, ax = plt.subplots(figsize=(9, 4.8))
    for alg in ALGORITHMS:
        sub = by_split.loc[by_split.algorithm == alg].sort_values('seed')
        ax.plot(sub.seed, sub.train_unique_voxels, marker='o', label=alg.replace('_200scenes',''))
    ax.set_xlabel('Fixed split seed')
    ax.set_ylabel('Distinct TRAINING voxels')
    ax.set_title('Pi0 200 - spatial coverage of 70% training sets')
    ax.legend()
    save_figure(fig, figs, '03_train_unique_voxels_by_seed')

    fig, ax = plt.subplots(figsize=(9, 4.8))
    for alg in ALGORITHMS:
        sub = by_split.loc[by_split.algorithm == alg].sort_values('seed')
        ax.plot(sub.seed, sub.test_episode_voxel_visit_known_fraction,
                marker='o', label=alg.replace('_200scenes',''))
    ax.set_xlabel('Fixed split seed')
    ax.set_ylabel('Fraction of test episode-voxel visits known in training')
    ax.set_ylim(0, 1)
    ax.set_title('Pi0 200 - held-out voxel familiarity (diagnostic)')
    ax.legend()
    save_figure(fig, figs, '04_test_voxel_familiarity_by_seed')

    settings = {
        'step': '04_deterministic_tcp_xyz_voxelization',
        'model': 'Pi0', 'budget': 200, 'algorithms': list(ALGORITHMS),
        'voxel_size_m': size, 'origin_m': list(origin),
        'voxel_rule': 'floor((tcp_xyz - origin_xyz)/voxel_size)',
        'binary_rule': 'each distinct voxel counted once per episode',
        'split_manifest': str(split_path.relative_to(root)),
        'split_seeds': list(SEEDS),
        'scene_key_status': 'candidate scene key confirmed by previous script; full simulator identity not independently proven',
        'full_episode_voxel_summary': 'descriptive only; must not be used to learn test-dependent suspiciousness',
        'training_coverage_definition': 'absolute count of unique training-visited voxels',
    }
    (out / 'voxelization_settings.json').write_text(json.dumps(settings, indent=2), encoding='utf-8')
    report = {
        'status': 'PASS', 'runs_validated': 40, 'episodes_validated': sum(map(len, by_alg.values())),
        'expected_splits': 40, 'splits_analyzed': len(by_split),
        'binary_coverage_spectrum': str(spectrum_path.relative_to(root)),
        'output_folder': str(out),
        'note': 'No suspiciousness trained yet; Step 05 will use training-only failure/pass counts.',
    }
    (out / 'validation_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('\nRQ1 STEP 04 - VOXEL COVERAGE')
    print(f"Runs: {report['runs_validated']}/40 | Episodes: {report['episodes_validated']}/8000")
    print(f"Splits: {len(by_split)}/40 | Status: {report['status']}")
    print('Saved:', out)
    print('Next: Step 05 compute train-only e_f, e_p, n_f, n_p for each voxel/seed.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parent.parent)
    p.add_argument('--voxel-size', type=float, default=0.02)
    p.add_argument('--origin', type=str, default='-1,-1,0')
    args = p.parse_args()
    try:
        origin = tuple(float(x.strip()) for x in args.origin.split(','))
        if len(origin) != 3 or not all(math.isfinite(v) for v in origin):
            raise ValueError('origin must be 3 finite comma-separated coordinates')
        if not (math.isfinite(args.voxel_size) and args.voxel_size > 0):
            raise ValueError('voxel-size must be positive')
        produce_outputs(args.repo_root.resolve(), args.voxel_size, origin)
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print('BLOCKED:', str(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
