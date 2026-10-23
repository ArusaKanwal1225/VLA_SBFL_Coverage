#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RQ1 Pi0 400 Step 05: compute train-only SBFL contingency counts per voxel.

Run from the SBFL-only repository:
    python src/RQ1_Pi0_400/05_rq1_train_voxel_counts_pi0_400.py --repo-root .

Reads the EXACT Step 03 frozen split assignments and Step 04 binary
trajectory-to-voxel coverage. For each algorithm and seed, computes:
    e_f: failing TRAIN episodes visiting voxel
    e_p: passing TRAIN episodes visiting voxel
    n_f: failing TRAIN episodes not visiting voxel
    n_p: passing TRAIN episodes not visiting voxel

No test episode's voxel visits or outcomes contribute to these counts.
This step does NOT compute suspiciousness; that is Step 06.

Dependencies: Python 3.9+, numpy, pandas, matplotlib.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ALGORITHMS = (
    'DH_400scenes', 'DH_HybridDT_400scenes',
    'DH_HybridRF_400scenes', 'DH_WeightedRF_400scenes',
)
SEEDS = (11, 23, 37, 41, 53, 67, 79, 83, 97, 101)
GROUP = 'scene_group_split'
EXACT_GROUP = 'scene_group_exact_candidate'
NEAR_GROUP = 'scene_group_4dp_diagnostic'
STEM = Path('output/RQ1/pi0_400')
VISITS_NAME = 'episode_voxel_visits.csv.gz'
COUNTS_NAME = 'train_voxel_contingency_counts.csv.gz'


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def figure_save(fig, folder: Path, stem: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for suffix in ('png', 'pdf'):
        fig.savefig(folder / f'{stem}.{suffix}', dpi=200, bbox_inches='tight')
    plt.close(fig)


def save_csv(df, filename: Path) -> None:
    filename.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(df).to_csv(filename, index=False, encoding='utf-8')


def check_inputs(root: Path):
    split_path = root / STEM / 'step03_grouped_splits/split_manifest.csv'
    step4 = root / STEM / 'step04_voxel_coverage'
    visits_path = step4 / VISITS_NAME
    epstats_path = step4 / 'per_episode_voxel_statistics.csv'
    coverage_path = step4 / 'split_voxel_coverage.csv'
    settings_path = step4 / 'voxelization_settings.json'
    report_path = step4 / 'validation_report.json'
    for p in (split_path, visits_path, epstats_path, coverage_path,
              settings_path, report_path):
        if not p.is_file():
            raise ValueError(f'Missing prerequisite: {p}')

    settings = json.loads(settings_path.read_text(encoding='utf-8'))
    report = json.loads(report_path.read_text(encoding='utf-8'))
    if report.get('status') != 'PASS' or report.get('splits_analyzed') != 40 or report.get('episodes_validated') != 16000:
        raise ValueError('Step 04 validation report is not PASS for all 40 splits')
    if settings.get('model') != 'Pi0' or settings.get('budget') != 400:
        raise ValueError('Step 04 settings are not for Pi0, 400 scenes')
    if not math.isclose(float(settings.get('voxel_size_m', -1)), 0.02,
                        abs_tol=1e-12, rel_tol=0):
        raise ValueError('Voxel size differs from the frozen 0.02m protocol')
    if [float(x) for x in settings.get('origin_m', [])] != [-1.0, -1.0, 0.0]:
        raise ValueError('Grid origin differs from the frozen [-1,-1,0] protocol')
    if settings.get('binary_rule') != 'each distinct voxel counted once per episode':
        raise ValueError('Step 04 is not the expected binary episode-voxel coverage')

    manifest = pd.read_csv(split_path, dtype=str, keep_default_na=False,
                           encoding='utf-8-sig')
    required = {'algorithm', 'seed', 'episode_id', 'run_id', 'failure',
                'split', GROUP, EXACT_GROUP, NEAR_GROUP}
    if not required.issubset(manifest.columns):
        raise ValueError(f'Missing split columns: {sorted(required-set(manifest.columns))}')
    if len(manifest) != 160000 or manifest[['algorithm','seed','episode_id']].duplicated().any():
        raise ValueError('Split manifest must have exactly 160000 unique algorithm/seed/episode rows')
    if set(manifest.algorithm) != set(ALGORITHMS) or set(manifest.seed) != set(map(str, SEEDS)):
        raise ValueError('Algorithm or seed mismatch in frozen splits')
    if not manifest.failure.isin(['0','1']).all() or not manifest.split.isin(['train','test']).all():
        raise ValueError('Invalid failure labels or split assignments')
    if (manifest.episode_id == '').any() or manifest[[GROUP, EXACT_GROUP, NEAR_GROUP]].eq('').any().any():
        raise ValueError('Missing episode or scene group identifier')

    stats = pd.read_csv(epstats_path, dtype={'algorithm':str, 'episode_id':str},
                        encoding='utf-8-sig')
    stats_required = {'algorithm', 'episode_id', 'failure', 'unique_voxel_count'}
    if len(stats) != 16000 or not stats_required.issubset(stats.columns):
        raise ValueError('Step 04 per-episode statistics missing or incomplete')
    if stats[['algorithm','episode_id']].duplicated().any():
        raise ValueError('Duplicated episode in Step 04 statistics')
    if (stats.unique_voxel_count <= 0).any():
        raise ValueError('Episode with no visited voxels in Step 04 statistics')

    expected_counts = {}
    stats_labels = {}
    for row in stats.itertuples(index=False):
        key = (row.algorithm, row.episode_id)
        expected_counts[key] = int(row.unique_voxel_count)
        stats_labels[key] = str(row.failure)

    train_masks, labels, info, groupings = {}, {}, {}, {}
    for algorithm in ALGORITHMS:
        subset = manifest[manifest.algorithm == algorithm]
        if len(subset) != 40000:
            raise ValueError(f'{algorithm}: expected 40000 manifest rows')
        first_seed = subset[subset.seed == str(SEEDS[0])]
        if len(first_seed) != 4000 or first_seed.episode_id.nunique() != 4000:
            raise ValueError(f'{algorithm}: expected exactly 4000 unique episodes')
        ids = set(first_seed.episode_id)
        mask = {eid: 0 for eid in ids}
        label_map = dict(zip(first_seed.episode_id, first_seed.failure))
        run_map = dict(zip(first_seed.episode_id, first_seed.run_id))
        group_map = dict(zip(first_seed.episode_id, first_seed[GROUP]))
        exact_map = dict(zip(first_seed.episode_id, first_seed[EXACT_GROUP]))
        near_map = dict(zip(first_seed.episode_id, first_seed[NEAR_GROUP]))
        for eid in ids:
            if (algorithm, eid) not in expected_counts:
                raise ValueError(f'Missing Step 04 episode statistic: {algorithm}/{eid}')
            if label_map[eid] != stats_labels[(algorithm, eid)]:
                raise ValueError(f'Outcome mismatch: {algorithm}/{eid}')
        label_set = set(label_map.values())
        if label_set != {'0','1'}:
            raise ValueError(f'{algorithm}: not both success and failure labels')
        train_stats = []
        for j, seed in enumerate(SEEDS):
            s = subset[subset.seed == str(seed)]
            if len(s) != 4000 or set(s.episode_id) != ids:
                raise ValueError(f'{algorithm}/seed={seed}: episode set differs')
            if any(label_map[eid] != lbl or run_map[eid] != run or
                   group_map[eid] != grp or exact_map[eid] != exact or
                   near_map[eid] != near
                   for eid, lbl, run, grp, exact, near in
                   zip(s.episode_id, s.failure, s.run_id, s[GROUP],
                       s[EXACT_GROUP], s[NEAR_GROUP])):
                raise ValueError(f'{algorithm}/seed={seed}: episode metadata differs')
            tr = s[s.split == 'train']
            te = s[s.split == 'test']
            if not (2750 <= len(tr) <= 2850 and 1150 <= len(te) <= 1250):
                raise ValueError(f'{algorithm}/seed={seed}: unexpected split size')
            if tr.failure.nunique() < 2 or te.failure.nunique() < 2:
                raise ValueError(f'{algorithm}/seed={seed}: one-class split')
            for group_name in (GROUP, EXACT_GROUP, NEAR_GROUP):
                if set(tr[group_name]) & set(te[group_name]):
                    raise ValueError(f'{algorithm}/seed={seed}: train/test overlap in {group_name}')
            for eid in tr.episode_id:
                mask[eid] |= 1 << j
            n_f = int((tr.failure == '1').sum())
            n_p = int((tr.failure == '0').sum())
            train_stats.append({
                'algorithm': algorithm, 'seed': seed,
                'training_episodes': len(tr), 'test_episodes': len(te),
                'training_failures': n_f, 'training_passes': n_p,
                'training_failure_prevalence': n_f / len(tr),
                'test_failures_diagnostic_only': int((te.failure == '1').sum()),
                'train_expected_binary_visits': sum(expected_counts[(algorithm, eid)] for eid in tr.episode_id),
            })
        if not all(m > 0 for m in mask.values()):
            # May be perfectly legitimate that one episode is in the test set
            # for every seed. Do not assume every episode must train somewhere.
            pass
        train_masks[algorithm] = mask
        labels[algorithm] = label_map
        info[algorithm] = train_stats
        groupings[algorithm] = group_map

    coverage = pd.read_csv(coverage_path, dtype={'algorithm':str, 'seed':int},
                           encoding='utf-8-sig')
    need = {'algorithm', 'seed', 'train_unique_voxels',
            'train_binary_episode_voxel_visits', 'train_episodes', 'test_episodes'}
    if len(coverage) != 40 or not need.issubset(coverage.columns):
        raise ValueError('Step 04 split coverage summary missing required entries')
    if coverage[['algorithm','seed']].duplicated().any():
        raise ValueError('Duplicate Step 04 split coverage row')
    coverage_map = {(r.algorithm,int(r.seed)):r for r in coverage.itertuples(index=False)}
    if set(coverage_map) != {(a,s) for a in ALGORITHMS for s in SEEDS}:
        raise ValueError('Step 04 split coverage lacks algorithm/seed combinations')
    return (visits_path, split_path, settings_path, train_masks,
            labels, info, expected_counts, coverage_map)


def support_bin(support: int) -> str:
    if support <= 2:
        return str(support)
    if support <= 5:
        return '3-5'
    if support <= 10:
        return '6-10'
    if support <= 20:
        return '11-20'
    if support <= 50:
        return '21-50'
    if support <= 100:
        return '51-100'
    return '101+'


def export_algorithm(alg, counts, labels, train_info, incidence, coverage_map,
                     writer, summary_rows, support_rows, example_rows):
    for j, seed in enumerate(SEEDS):
        training = train_info[j]
        F = training['training_failures']
        P = training['training_passes']
        index_fail, index_pass = 2*j, 2*j+1
        entries = []
        histogram = Counter()
        support_binned = Counter()
        for voxel, vec in counts.items():
            ef, ep = vec[index_fail], vec[index_pass]
            support = ef + ep
            if support == 0:
                continue  # Not covered by any training episode for this split
            nf, np_ = F - ef, P - ep
            if min(ef, ep, nf, np_) < 0 or (ef+nf != F) or (ep+np_ != P):
                raise ValueError(f'Invalid SBFL contingency values: {alg}/{seed}/{voxel}')
            writer.writerow([alg, seed, *voxel, ef, ep, nf, np_, support,
                             F, P])
            histogram[support] += 1
            support_binned[support_bin(support)] += 1
            entries.append((voxel, ef, ep, support))
        if not entries:
            raise ValueError(f'{alg}/seed={seed} has no covered training voxels')
        expected = coverage_map[(alg,seed)]
        total_visits = sum(e[3] for e in entries)
        if len(entries) != int(expected.train_unique_voxels):
            raise ValueError(f'{alg}/{seed}: voxel count disagrees with Step 04')
        if total_visits != int(expected.train_binary_episode_voxel_visits):
            raise ValueError(f'{alg}/{seed}: binary coverage incidence disagrees with Step 04')
        if total_visits != training['train_expected_binary_visits']:
            raise ValueError(f'{alg}/{seed}: train episode visits disagree with Step 04 episode statistics')
        if (F + P) != int(expected.train_episodes) or training['test_episodes'] != int(expected.test_episodes):
            raise ValueError(f'{alg}/{seed}: split size disagrees with Step 04')
        support_values = np.asarray([x[3] for x in entries], dtype=np.int32)
        only_f = sum(x[1] > 0 and x[2] == 0 for x in entries)
        only_p = sum(x[2] > 0 and x[1] == 0 for x in entries)
        both = sum(x[1] > 0 and x[2] > 0 for x in entries)
        summary_rows.append({
            **training,
            'training_covered_voxels': len(entries),
            'training_episode_voxel_visits': total_visits,
            'failure_episode_voxel_visits': sum(x[1] for x in entries),
            'passing_episode_voxel_visits': sum(x[2] for x in entries),
            'voxels_visited_only_by_failures': only_f,
            'voxels_visited_only_by_passes': only_p,
            'voxels_visited_by_both': both,
            'voxels_with_support_1': int(histogram[1]),
            'voxels_with_support_le_2': int(sum(n for k,n in histogram.items() if k <= 2)),
            'fraction_support_le_2': sum(n for k,n in histogram.items() if k <= 2)/len(entries),
            'mean_episode_support': float(support_values.mean()),
            'median_episode_support': float(np.median(support_values)),
            'max_episode_support': int(support_values.max()),
            'consistency': 'PASS',
        })
        for support, frequency in sorted(histogram.items()):
            support_rows.append({'algorithm':alg, 'seed':seed,
                                 'episode_support':support, 'voxel_count':frequency})
        if seed == SEEDS[0]:
            for voxel, ef, ep, supp in sorted(entries, key=lambda x: (-x[3], x[0]))[:15]:
                example_rows.append({'algorithm':alg, 'seed':seed,
                                     'voxel_i':voxel[0], 'voxel_j':voxel[1],
                                     'voxel_k':voxel[2], 'e_f':ef, 'e_p':ep,
                                     'n_f':F-ef, 'n_p':P-ep,
                                     'support':supp,
                                     'note':'high support only, NOT suspiciousness ranking'})
        print(f'[COUNTS] {alg} seed={seed}: train={F+P} (fail={F}, pass={P}) '
              f'voxels={len(entries)} sparse_le2={summary_rows[-1]["fraction_support_le_2"]:.1%}')


def generate_plots(summary, support_df, fig_dir, out):
    xs = np.array(SEEDS)
    labels = [a.replace('_400scenes', '') for a in ALGORITHMS]
    fig, ax = plt.subplots(figsize=(9.4, 4.7))
    for a in ALGORITHMS:
        sub = summary[summary.algorithm == a].sort_values('seed')
        ax.plot(sub.seed, sub.training_covered_voxels, marker='o', label=a.replace('_400scenes',''))
    ax.set_xticks(xs)
    ax.set_xlabel('Frozen split seed')
    ax.set_ylabel('Number of training-covered voxels')
    ax.set_title('Pi0 400: training voxel count across split seeds')
    ax.legend(fontsize=8)
    figure_save(fig, fig_dir, '01_train_covered_voxels_by_seed')

    fig, ax = plt.subplots(figsize=(9.4, 4.7))
    for a in ALGORITHMS:
        sub = summary[summary.algorithm == a].sort_values('seed')
        ax.plot(sub.seed, sub.fraction_support_le_2, marker='o',
                label=a.replace('_400scenes',''))
    ax.set_xticks(xs)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Frozen split seed')
    ax.set_ylabel('Fraction of voxels supported by <=2 training episodes')
    ax.set_title('Pi0 400: sparse training evidence by algorithm')
    ax.legend(fontsize=8)
    figure_save(fig, fig_dir, '02_sparse_voxel_fraction_by_seed')

    fig, ax = plt.subplots(figsize=(9.4, 4.7))
    width = 0.36
    idx = np.arange(4)
    mean_ef = [summary.loc[summary.algorithm == a, 'failure_episode_voxel_visits'].mean()
               for a in ALGORITHMS]
    mean_ep = [summary.loc[summary.algorithm == a, 'passing_episode_voxel_visits'].mean()
               for a in ALGORITHMS]
    ax.bar(idx-width/2, mean_ef, width=width, label='Failing train episode-voxel visits')
    ax.bar(idx+width/2, mean_ep, width=width, label='Passing train episode-voxel visits')
    ax.set_xticks(idx, labels, rotation=15)
    ax.set_ylabel('Mean binary incidences across 10 splits')
    ax.set_title('Pi0 400: failure/pass training coverage volume')
    ax.legend(fontsize=8)
    figure_save(fig, fig_dir, '03_train_binary_visits_by_outcome')

    wanted = ['1','2','3-5','6-10','11-20','21-50','51-100','101+']
    bins = []
    for a in ALGORITHMS:
        sub = support_df[(support_df.algorithm == a) & (support_df.seed == SEEDS[0])]
        total = int(sub.voxel_count.sum())
        d = Counter()
        for r in sub.itertuples(index=False):
            d[support_bin(int(r.episode_support))] += int(r.voxel_count)
        for bin_name in wanted:
            bins.append({'algorithm':a,'seed':SEEDS[0], 'support_bin':bin_name,
                         'voxel_count':d[bin_name],
                         'fraction_of_train_voxels':d[bin_name]/total if total else 0.0})
    bins_df = pd.DataFrame(bins)
    save_csv(bins_df, out / 'support_bin_distribution_seed11.csv')
    fig, ax = plt.subplots(figsize=(10, 4.9))
    positions = np.arange(len(wanted))
    width = .19
    for k, a in enumerate(ALGORITHMS):
        vals = bins_df[bins_df.algorithm == a].set_index('support_bin').loc[wanted, 'fraction_of_train_voxels']
        ax.bar(positions + (k-1.5)*width, vals, width, label=a.replace('_400scenes',''))
    ax.set_xticks(positions, wanted)
    ax.set_xlabel('Number of training episodes visiting a voxel (binned)')
    ax.set_ylabel('Fraction of training-covered voxels')
    ax.set_title('Pi0 400: voxel support distribution (seed 11)')
    ax.legend(fontsize=8)
    figure_save(fig, fig_dir, '04_voxel_support_distribution_seed11')


def run(root: Path) -> None:
    (visits_path, split_path, settings_path, train_masks, labels,
     info, expected_counts, coverage_map) = check_inputs(root)
    out = root / STEM / 'step05_training_contingency'
    out.mkdir(parents=True, exist_ok=True)
    figures = out / 'figures'
    all_summaries, support_rows, example_rows = [], [], []
    last_alg = None
    last_key = None
    completed = []
    counts = {}
    episodes_with_visits = set()
    observed_voxel_per_episode = Counter()
    train_incidence = [0] * len(SEEDS)

    counts_path = out / COUNTS_NAME
    with gzip.open(visits_path, 'rt', newline='', encoding='utf-8-sig') as f, \
         gzip.open(counts_path, 'wt', newline='', encoding='utf-8') as outf:
        reader = csv.DictReader(f)
        required = {'algorithm','episode_id','voxel_i','voxel_j','voxel_k'}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError('Step 04 spectrum is missing column(s): ' + str(required-set(reader.fieldnames or ())))
        writer = csv.writer(outf)
        writer.writerow(['algorithm','seed','voxel_i','voxel_j','voxel_k',
                         'e_f','e_p','n_f','n_p','support',
                         'training_total_failures','training_total_passes'])
        for row in reader:
            alg = row['algorithm']
            if alg not in train_masks:
                raise ValueError('Unexpected algorithm in Step 04 spectrum: ' + alg)
            if alg != last_alg:
                if last_alg is not None:
                    if len(episodes_with_visits) != 4000:
                        raise ValueError(f'{last_alg}: expected 4000 covered episodes')
                    export_algorithm(last_alg, counts, labels[last_alg], info[last_alg],
                                     train_incidence, coverage_map, writer,
                                     all_summaries, support_rows, example_rows)
                    completed.append(last_alg)
                if alg in completed or alg != ALGORITHMS[len(completed)]:
                    raise ValueError('Step 04 spectrum algorithms are not in expected block order')
                last_alg = alg
                last_key = None
                counts = defaultdict(lambda:[0]* (2 * len(SEEDS)))
                episodes_with_visits = set()
                train_incidence = [0]*len(SEEDS)
            eid = row['episode_id']
            if eid not in train_masks[alg]:
                raise ValueError(f'{alg}: visited voxel for unrecognized episode_id {eid}')
            voxel = (int(row['voxel_i']), int(row['voxel_j']), int(row['voxel_k']))
            key = (eid, voxel)
            if last_key is not None and key <= last_key:
                raise ValueError(f'{alg}: duplicate/out-of-order (episode,voxel) in Step 04 spectrum')
            last_key = key
            episodes_with_visits.add(eid)
            observed_voxel_per_episode[(alg,eid)] += 1
            bitmask = train_masks[alg][eid]
            if bitmask:
                outcome_offset = 0 if labels[alg][eid] == '1' else 1
                vec = counts[voxel]
                for j in range(len(SEEDS)):
                    if bitmask & (1 << j):
                        vec[2*j+outcome_offset] += 1
                        train_incidence[j] += 1
        if last_alg is not None:
            if len(episodes_with_visits) != 4000:
                raise ValueError(f'{last_alg}: expected 4000 covered episodes')
            export_algorithm(last_alg, counts, labels[last_alg], info[last_alg],
                             train_incidence, coverage_map, writer,
                             all_summaries, support_rows, example_rows)
            completed.append(last_alg)
    if tuple(completed) != ALGORITHMS:
        raise ValueError(f'Missing algorithms from Step 04 spectrum: {completed}')
    if len(observed_voxel_per_episode) != 16000:
        raise ValueError('Step 04 spectrum had fewer than 16000 distinct episodes')
    for key, n in expected_counts.items():
        if observed_voxel_per_episode[key] != n:
            raise ValueError(f'{key}: binary voxel row count differs from Step 04 per-episode statistics')
    summary = pd.DataFrame(all_summaries)
    support_df = pd.DataFrame(support_rows)
    if len(summary) != 40 or not summary.consistency.eq('PASS').all():
        raise ValueError('Did not produce all 40 valid SBFL spectra')
    save_csv(summary, out / 'training_contingency_split_summary.csv')
    save_csv(support_df, out / 'voxel_support_histogram.csv')
    save_csv(example_rows, out / 'example_high_support_voxels_seed11.csv')
    generate_plots(summary, support_df, figures, out)
    metadata = {
        'status':'PASS', 'model':'Pi0', 'budget':400,
        'algorithms':list(ALGORITHMS), 'seeds':list(SEEDS),
        'total_spectra':40, 'total_episodes_covered_in_step04':len(observed_voxel_per_episode),
        'component':'3D TCP voxel',
        'counts_definitions':{
            'e_f':'failing TRAIN episodes covering the voxel',
            'e_p':'passing TRAIN episodes covering the voxel',
            'n_f':'failing TRAIN episodes not covering the voxel',
            'n_p':'passing TRAIN episodes not covering the voxel',
        },
        'test_outcomes_used_to_fit_scores':False,
        'candidate_scene_group_limitation':'Split groups include reviewed near-duplicate merging; full simulator scene identity is not independently proven.',
        'voxelization_settings_path':str(settings_path.relative_to(root)),
        'split_manifest_sha256':digest_file(split_path),
        'step04_binary_spectrum_sha256':digest_file(visits_path),
        'counts_file':str(counts_path.relative_to(root)),
        'note':'Unvisited training voxels are absent from counts file. Later risk scoring needs an explicit unseen-voxel policy.',
    }
    (out/'step05_validation_report.json').write_text(json.dumps(metadata, indent=2),encoding='utf-8')
    print('\nRQ1 STEP 05 - TRAINING-ONLY VOXEL CONTINGENCY COUNTS')
    print(f'Spectra: {len(summary)}/40 | Validated episodes: {len(observed_voxel_per_episode)}/16000')
    print('Test labels used for voxel counts: NO')
    print('Status: PASS')
    print('Outputs:', out)
    print('Next: Step 06 apply the 14 SBFL formulas to these frozen training counts.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path,
                        default=Path(__file__).resolve().parents[2],
                        help='Path containing input, src and output folders')
    args = parser.parse_args()
    try:
        run(args.repo_root.resolve())
        return 0
    except (OSError, ValueError, KeyError, TypeError) as e:
        print('BLOCKED:', e, file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
