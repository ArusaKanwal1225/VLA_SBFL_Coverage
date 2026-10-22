#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RQ1 Step 03 | Review scene keys; optionally make grouped 70/30 splits.

Review first (safe; does NOT split):
  python src/03_rq1_review_and_split_pi0_200.py

After verifying that object_name, orientation_id, x, y represent the scene
configuration for YOUR experiment and reviewing the duplicate/near-duplicate
CSV outputs, explicitly enable splits:
  python src/03_rq1_review_and_split_pi0_200.py --confirm-scene-key

All plots are saved as both PNG and PDF; underlying values are saved as CSV.
No original input is modified. Compatible with Python 3.9+.
"""
from __future__ import annotations
import argparse
import json
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
KEY = 'scene_group_exact_candidate'
NEAR_KEY = 'scene_group_4dp_diagnostic'
REQUIRED = (
    'algorithm', 'run_id', 'eval_id', 'episode_id', 'failure',
    'object_name', 'orientation_id', 'x', 'y', KEY, NEAR_KEY
)


def save_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding='utf-8')


def save_fig(fig, folder, stem):
    fig.tight_layout()
    fig.savefig(folder / (stem + '.png'), dpi=200, bbox_inches='tight')
    fig.savefig(folder / (stem + '.pdf'), bbox_inches='tight')
    plt.close(fig)


def validate_dataset(root):
    path = root / 'input' / 'Pi0' / 'scene_metadata_candidates_200.csv'
    if not path.is_file():
        raise ValueError('Missing scene metadata: ' + str(path) + '. Run Step 02 first.')
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding='utf-8-sig')
    missing = set(REQUIRED) - set(df.columns)
    if missing:
        raise ValueError('Missing columns: ' + ', '.join(sorted(missing)))
    if len(df) != 8000 or df.episode_id.duplicated().any():
        raise ValueError('Expected 8,000 globally unique episode IDs')
    if df[list(REQUIRED)].eq('').any().any():
        raise ValueError('Blank required metadata field')
    if not df.failure.isin(['0', '1']).all():
        raise ValueError('Invalid failure labels')
    if set(df.algorithm) != set(ALGORITHMS):
        raise ValueError('Unexpected set of algorithms in candidate manifest')
    for alg in ALGORITHMS:
        subset = df[df.algorithm == alg]
        if len(subset) != 2000:
            raise ValueError(f'Expected 2000 episodes for {alg}, got {len(subset)}')
        for ri in range(1, 11):
            run = 'run_' + str(ri)
            ss = subset[subset.run_id == run]
            if len(ss) != 200 or ss.eval_id.nunique() != 200:
                raise ValueError(f'Invalid run count or duplicate eval_id for {alg}/{run}')
            episodes_file = root / 'input' / 'Pi0' / alg / run / 'episodes.csv'
            if not episodes_file.is_file():
                raise ValueError('Missing input episodes file: ' + str(episodes_file))
            eps = pd.read_csv(episodes_file, dtype=str, keep_default_na=False,
                              usecols=['episode_id', 'failure'])
            both = ss[['episode_id', 'failure']].merge(
                eps, on='episode_id', how='outer', suffixes=('_meta', '_input'), indicator=True)
            if len(both) != 200 or (both._merge != 'both').any() or \
                    (both.failure_meta != both.failure_input).any():
                raise ValueError(f'Episode/label mismatch with input for {alg}/{run}')

    # A hash must always refer to the same specified scene variables. A hash
    # cannot, however, prove that ALL scene properties were captured.
    config_counts = (df.groupby(['algorithm', KEY], sort=False)
                       [['object_name', 'orientation_id', 'x', 'y']]
                       .nunique(dropna=False))
    if (config_counts > 1).any().any():
        raise ValueError('Inconsistent scene fields within an exact candidate group')
    return df


def inspect_groups(df, out, figures):
    sizes = (df.groupby(['algorithm', KEY], sort=False)
             .agg(group_size=('episode_id', 'size'),
                  distinct_runs=('run_id', 'nunique'),
                  failures=('failure', lambda s: int((s == '1').sum())),
                  successes=('failure', lambda s: int((s == '0').sum())))
             .reset_index())
    duplicates = sizes[sizes.group_size > 1].copy()
    duplicated_eps = df.merge(duplicates[['algorithm', KEY]],
                               on=['algorithm', KEY], how='inner')
    near_groups = (df.groupby(['algorithm', NEAR_KEY])
                     .agg(exact_groups=(KEY, 'nunique'),
                          episodes=('episode_id', 'size'))
                     .reset_index())
    near_groups = near_groups[near_groups.exact_groups > 1].copy()
    near_eps = df.merge(near_groups[['algorithm', NEAR_KEY]],
                         on=['algorithm', NEAR_KEY], how='inner')
    summary = []
    for alg in ALGORITHMS:
        s = sizes[sizes.algorithm == alg]
        near = near_groups[near_groups.algorithm == alg]
        summary.append({
            'algorithm': alg, 'episodes': 2000,
            'exact_groups': len(s),
            'repeated_exact_groups': int((s.group_size > 1).sum()),
            'episodes_in_repeated_groups': int(s.loc[s.group_size > 1, 'group_size'].sum()),
            'cross_run_exact_groups': int((s.distinct_runs > 1).sum()),
            'rounded4_groups_containing_multiple_exact_groups': len(near),
        })
    summary = pd.DataFrame(summary)
    save_csv(sizes, out / 'all_exact_group_sizes.csv')
    save_csv(duplicates, out / 'repeated_exact_groups.csv')
    save_csv(duplicated_eps.sort_values(['algorithm', KEY, 'run_id']),
             out / 'repeated_scene_episodes.csv')
    save_csv(near_groups, out / 'near_duplicate_group_summary.csv')
    save_csv(near_eps.sort_values(['algorithm', NEAR_KEY, 'run_id']),
             out / 'near_duplicate_episodes.csv')
    save_csv(summary, out / 'scene_key_review_summary.csv')

    labels = [s.replace('_200scenes', '') for s in ALGORITHMS]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(labels, summary.episodes_in_repeated_groups)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                str(int(bar.get_height())), ha='center', va='bottom')
    ax.set_ylabel('Episodes in repeated exact scene groups')
    ax.set_ylim(0, max(3, summary.episodes_in_repeated_groups.max() * 1.25))
    ax.set_title('Pi0 200: repeated candidate scene configurations')
    ax.tick_params(axis='x', labelrotation=15)
    save_fig(fig, figures, '01_repeated_exact_groups')

    fig, ax = plt.subplots(figsize=(9, 4.5))
    widths = np.arange(len(summary))
    ax.bar(widths - .18, summary.repeated_exact_groups, width=.36,
           label='Repeated exact scene groups')
    ax.bar(widths + .18, summary.rounded4_groups_containing_multiple_exact_groups,
           width=.36, label='Near-duplicate groups (diagnostic)')
    ax.set_xticks(widths, labels, rotation=15)
    ax.set_ylabel('Number of groups')
    ax.set_title('Pi0 200: exact versus near-duplicate scene candidates')
    ax.legend()
    save_fig(fig, figures, '02_exact_and_near_duplicate_groups')

    print('\nSCENE GROUP REVIEW - Pi0 200')
    print(summary.to_string(index=False))
    if not duplicated_eps.empty:
        print('\nREPEATED EXACT EPISODES (for manual review):')
        print(duplicated_eps[['algorithm', 'run_id', 'eval_id', 'failure',
                               'object_name', 'orientation_id', 'x', 'y']]
              .to_string(index=False))
    if not near_groups.empty:
        print(f'\nATTENTION: {len(near_groups)} rounded-coordinate groups contain '
              'more than one exact scene key. Review near_duplicate_episodes.csv.')
    return summary, duplicates, near_groups


def assign_stratified_groups(subset, seed):
    """A deterministic scene-group split, stratified approximately by run and outcome.

    A repeated scene group is an indivisible unit. The lowest-numbered run
    provides its assignment stratum; all members inherit the group assignment.
    """
    rng = np.random.default_rng(seed)
    groups = []
    for key, g in subset.groupby(KEY, sort=True):
        labels = list(g['failure'])
        n_fail = labels.count('1')
        n_pass = labels.count('0')
        category = 'mixed' if n_fail and n_pass else ('fail' if n_fail else 'pass')
        primary_run = min(g['run_id'], key=lambda x: int(x.rsplit('_', 1)[1]))
        groups.append((key, len(g), n_fail, primary_run, category))
    strata = {}
    for record in groups:
        strata.setdefault((record[3], record[4]), []).append(record)
    test_groups = set()
    for stratum in sorted(strata):
        records = strata[stratum]
        n_pick = int(round(0.30 * len(records)))
        chosen = rng.choice(len(records), size=n_pick, replace=False)
        for j in chosen:
            test_groups.add(records[int(j)][0])
    split = subset[KEY].map(lambda key: 'test' if key in test_groups else 'train')
    return split


def produce_splits(df, out, figures, summary):
    manifest_chunks = []
    split_summary = []
    run_summary = []
    leakage_rows = []
    for alg in ALGORITHMS:
        subset = df[df.algorithm == alg].copy()
        for seed in SEEDS:
            split = assign_stratified_groups(subset, seed)
            result = subset[['algorithm', 'run_id', 'eval_id', 'episode_id',
                             'failure', 'object_name', KEY]].copy()
            result.insert(0, 'seed', seed)
            result['split'] = list(split)
            train = result[result.split == 'train']
            test = result[result.split == 'test']
            common = set(train[KEY]) & set(test[KEY])
            n_test = len(test)
            if common or not (580 <= n_test <= 620):
                raise ValueError(f'Leakage or count imbalance for {alg} seed={seed}: '
                                 f'test={n_test}, overlap={len(common)}')
            for part in (train, test):
                if part.failure.nunique() != 2:
                    raise ValueError(f'Single-class split: {alg} seed={seed} part={part.split.iloc[0]}')
            split_summary.append({
                'algorithm': alg, 'seed': seed,
                'train_episodes': len(train), 'test_episodes': n_test,
                'train_failures': int((train.failure == '1').sum()),
                'test_failures': int((test.failure == '1').sum()),
                'train_failure_rate': float((train.failure == '1').mean()),
                'test_failure_rate': float((test.failure == '1').mean()),
                'train_scene_groups': train[KEY].nunique(),
                'test_scene_groups': test[KEY].nunique(),
                'shared_scene_groups': len(common),
            })
            leakage_rows.append({'algorithm': alg, 'seed': seed,
                                 'shared_scene_groups': len(common),
                                 'shared_episode_ids': len(set(train.episode_id) & set(test.episode_id))})
            for rid, rg in result.groupby('run_id'):
                run_summary.append({'algorithm': alg, 'seed': seed, 'run_id': rid,
                                    'train_episodes': int((rg.split == 'train').sum()),
                                    'test_episodes': int((rg.split == 'test').sum()),
                                    'test_failure_rate': float((rg.loc[rg.split == 'test', 'failure'] == '1').mean())})
            manifest_chunks.append(result)
            print(f'[SPLIT] {alg} seed={seed}: train={len(train)} test={n_test} '
                  f'shared_scene_groups={len(common)}')
    full = pd.concat(manifest_chunks, ignore_index=True)
    totals = pd.DataFrame(split_summary)
    per_run = pd.DataFrame(run_summary)
    leakage = pd.DataFrame(leakage_rows)
    save_csv(full, out / 'split_manifest.csv')
    save_csv(totals, out / 'split_summary.csv')
    save_csv(per_run, out / 'per_run_split_summary.csv')
    save_csv(leakage, out / 'group_leakage_checks.csv')

    fig, ax = plt.subplots(figsize=(10, 4.5))
    a = totals.groupby('algorithm')[['train_episodes', 'test_episodes']].mean().reindex(ALGORITHMS)
    ix = np.arange(len(a))
    ax.bar(ix - .2, a.train_episodes, width=.4, label='Train')
    ax.bar(ix + .2, a.test_episodes, width=.4, label='Test')
    ax.set_xticks(ix, [s.replace('_200scenes', '') for s in ALGORITHMS], rotation=15)
    ax.set_ylabel('Mean number of episodes across 10 seeds')
    ax.set_title('Pi0 200: grouped 70/30 split sizes')
    ax.legend()
    save_fig(fig, figures, '03_train_test_sizes')

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for alg in ALGORITHMS:
        sub = totals[totals.algorithm == alg]
        ax.plot(sub.seed, sub.test_failure_rate, marker='o', label=alg.replace('_200scenes', ''))
    ax.set_xlabel('Split seed')
    ax.set_ylabel('Failure fraction in test episodes')
    ax.set_ylim(0, 1)
    ax.set_title('Pi0 200: held-out failure prevalence by split')
    ax.legend()
    save_fig(fig, figures, '04_test_failure_fraction_by_seed')

    fig, ax = plt.subplots(figsize=(10, 4.5))
    hm = per_run.groupby(['algorithm', 'run_id']).test_episodes.mean().unstack('run_id').reindex(ALGORITHMS)
    hm = hm.reindex(columns=[f'run_{i}' for i in range(1, 11)])
    im = ax.imshow(hm.to_numpy(), cmap='viridis', aspect='auto')
    ax.set_yticks(range(len(ALGORITHMS)), [s.replace('_200scenes', '') for s in ALGORITHMS])
    ax.set_xticks(range(10), [f'run_{i}' for i in range(1, 11)], rotation=45)
    ax.set_title('Pi0 200: average held-out episodes per run')
    fig.colorbar(im, ax=ax, label='Test episodes')
    save_fig(fig, figures, '05_test_episodes_by_run')
    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path,
                        default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--confirm-scene-key', action='store_true',
                        help='I verified that object_name, orientation_id, x, y '
                             'adequately identify scenes for the grouping protocol')
    args = parser.parse_args()
    root = args.repo_root.resolve()
    out = root / 'output' / 'RQ1' / 'pi0_200' / 'step03_grouped_splits'
    figures = out / 'figures'
    figures.mkdir(parents=True, exist_ok=True)
    try:
        df = validate_dataset(root)
        summary, duplicated, near = inspect_groups(df, out, figures)
    except (OSError, ValueError, KeyError) as exc:
        print('BLOCKED:', exc, file=sys.stderr)
        return 1

    report = {
        'step': '03_scene_key_review_and_grouped_splits',
        'model': 'Pi0', 'budget': 200, 'episodes_reviewed': len(df),
        'exact_candidate_fields': ['object_name', 'orientation_id', 'x', 'y'],
        'scene_key_scope': 'candidate configuration key, NOT independently verified simulator identity',
        'rounded_key_use': 'diagnostic only',
        'near_duplicate_groups_needing_review': int(len(near)),
        'explicit_key_confirmation': bool(args.confirm_scene_key),
        'split_seeds': list(SEEDS),
        'status': 'PENDING_KEY_REVIEW',
    }
    if args.confirm_scene_key:
        if len(near):
            print('BLOCKED: near-duplicate candidate groups exist. Inspect CSVs before deciding '\
                  'how to group them. No split generated.', file=sys.stderr)
            report['status'] = 'BLOCKED_NEAR_DUPLICATES'
            rc = 2
        else:
            try:
                summary_splits = produce_splits(df, out, figures, summary)
                report['status'] = 'GROUPED_SPLITS_CREATED_WITH_CONFIRMED_CANDIDATE_KEY'
                report['splits_created'] = len(summary_splits)
                print(f'\nSUCCESS: {len(summary_splits)} algorithm x seed splits created, '
                      'all group overlap checks passed.')
                rc = 0
            except (ValueError, OSError) as exc:
                report['status'] = 'FAILED_SPLIT_VALIDATION'
                report['error'] = str(exc)
                print('BLOCKED:', exc, file=sys.stderr)
                rc = 3
    else:
        print('\nREVIEW COMPLETE; NO TRAIN/TEST SPLITS CREATED.')
        print('Check repeated_scene_episodes.csv and near_duplicate_episodes.csv.')
        print('Verify object_name + orientation_id + x + y capture scene identity.')
        print('Only then rerun with --confirm-scene-key.')
        rc = 0
    (out / 'step03_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Output:', out)
    return rc


if __name__ == '__main__':
    sys.exit(main())
