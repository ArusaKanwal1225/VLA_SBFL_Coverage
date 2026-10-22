#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RQ1 Step 07: apply frozen SBFL maps to held-out Pi0 test episodes.

Run from the SBFL-only repository (Python 3.9+):
    python src/07_rq1_score_heldout_pi0_200.py

Inputs (existing, never modified):
    output/RQ1/pi0_200/step03_grouped_splits/split_manifest.csv
    output/RQ1/pi0_200/step04_voxel_coverage/episode_voxel_visits.csv.gz
    output/RQ1/pi0_200/step04_voxel_coverage/per_episode_voxel_statistics.csv
    output/RQ1/pi0_200/step04_voxel_coverage/split_voxel_coverage.csv
    output/RQ1/pi0_200/step04_voxel_coverage/validation_report.json
    output/RQ1/pi0_200/step05_training_contingency/training_contingency_split_summary.csv
    output/RQ1/pi0_200/step05_training_contingency/train_voxel_contingency_counts.csv.gz
    output/RQ1/pi0_200/step06_suspiciousness/voxel_suspiciousness_scores.csv.gz
    output/RQ1/pi0_200/step06_suspiciousness/step06_validation_report.json

Outputs (under output/RQ1/pi0_200/step07_heldout_scoring/):
    test_episode_scores.csv.gz  (wide: one row per held-out episode and split)
    test_scoring_split_summary.csv
    test_method_score_diagnostics.csv
    example_seed11_test_predictions.csv
    step07_validation_report.json
    figures/01_test_known_voxel_fraction_by_seed.png and .pdf
    figures/02_episodes_with_no_known_voxels_by_seed.png and .pdf
    figures/03_fraction_max_score_ties_seed11.png and .pdf
    figures/04_ochiai_heldout_risk_by_outcome_seed11.png and .pdf

Frozen rule:
  * Scores learned in Step 06 from TRAINING outcomes only.
  * For each TEST episode, use the maximum suspiciousness over TRAINING-KNOWN
    visited voxels. Ignore visited voxels absent from the training map.
  * If an episode visits no known voxel, assign a deterministic score below
    the minimum training score for that technique/algorithm/seed.
  * Truth labels are appended ONLY AFTER risk predictions are computed.
  * Step 08, not this step, calculates ROC-AUC and Average Precision.

Any warnings about unverified full scene identity from Step 03 remain in force.
All text in this file is ASCII to avoid Windows legacy-encoding problems.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import itertools
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ALGORITHMS = (
    'DH_200scenes',
    'DH_HybridDT_200scenes',
    'DH_HybridRF_200scenes',
    'DH_WeightedRF_200scenes',
)
SEEDS = (11, 23, 37, 41, 53, 67, 79, 83, 97, 101)
METHODS = (
    'Tarantula', 'Barinel', 'Kulczynski2', 'Wong2', 'Ochiai2',
    'Ochiai', 'DStar2', 'Jaccard', 'Wong3', 'GP13', 'OP2',
    'Wong1', 'Zoltar', 'Ample',
)
GROUP_COLUMN = 'scene_group_exact_candidate'
BASE = Path('output') / 'RQ1' / 'pi0_200'
STEP3 = BASE / 'step03_grouped_splits'
STEP4 = BASE / 'step04_voxel_coverage'
STEP5 = BASE / 'step05_training_contingency'
STEP6 = BASE / 'step06_suspiciousness'
STEP7 = BASE / 'step07_heldout_scoring'


def check(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def required_file(path):
    check(path.is_file(), 'Missing required input: ' + str(path))
    return path


def read_report(path):
    data = json.loads(required_file(path).read_text(encoding='utf-8-sig'))
    check(data.get('status') == 'PASS', str(path) + ': prerequisite not PASS')
    return data


def save_fig(fig, directory, name):
    directory.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(directory / (name + '.' + ext), dpi=170, bbox_inches='tight')
    plt.close(fig)


def load_inputs(root):
    paths = {
        'manifest': required_file(root / STEP3 / 'split_manifest.csv'),
        'visits': required_file(root / STEP4 / 'episode_voxel_visits.csv.gz'),
        'stats': required_file(root / STEP4 / 'per_episode_voxel_statistics.csv'),
        'coverage': required_file(root / STEP4 / 'split_voxel_coverage.csv'),
        'step4_report': required_file(root / STEP4 / 'validation_report.json'),
        'step5_summary': required_file(root / STEP5 / 'training_contingency_split_summary.csv'),
        'step5_counts': required_file(root / STEP5 / 'train_voxel_contingency_counts.csv.gz'),
        'scores': required_file(root / STEP6 / 'voxel_suspiciousness_scores.csv.gz'),
        'step6_report': required_file(root / STEP6 / 'step06_validation_report.json'),
    }
    r4 = read_report(paths['step4_report'])
    r6 = read_report(paths['step6_report'])
    check(r4.get('splits_analyzed') == 40 and r4.get('episodes_validated') == 8000,
          'Step04 did not validate all 40 splits and 8000 episodes')
    check(r6.get('total_method_spectra') == 560 and
          r6.get('test_outcomes_used_to_fit_scores') is False,
          'Step06 expected 560 maps trained without held-out labels')
    check(set(r6.get('methods', [])) == set(METHODS),
          'Step06 formula list does not match the 14-method protocol')
    check(r6.get('unseen_test_voxel_policy', '').find('below minimum') >= 0,
          'Step06 unseen-voxel policy differs from this script')
    check(r6.get('input_training_counts_sha256') == digest(paths['step5_counts']),
          'Step05 counts changed after Step06; do not mix frozen files')

    manifest = pd.read_csv(paths['manifest'], dtype=str, keep_default_na=False,
                           encoding='utf-8-sig')
    required = {'algorithm', 'seed', 'episode_id', 'run_id', 'failure',
                'split', GROUP_COLUMN}
    check(required.issubset(manifest.columns),
          'Step03 manifest missing: ' + str(sorted(required - set(manifest.columns))))
    check(len(manifest) == 80000 and
          not manifest[['algorithm','seed','episode_id']].duplicated().any(),
          'Expected 80000 unique split manifest rows')
    check(set(manifest.algorithm) == set(ALGORITHMS) and
          set(manifest.seed) == set(map(str, SEEDS)) and
          manifest.split.isin(['train', 'test']).all() and
          manifest.failure.isin(['0', '1']).all(),
          'Unexpected algorithm, split seed, split or failure label')
    check((manifest[GROUP_COLUMN] != '').all(), 'Blank scene group key')

    epstats = pd.read_csv(paths['stats'], dtype={'algorithm':str,'episode_id':str,
                                                 'run_id':str,'failure':str},
                          encoding='utf-8-sig')
    stcols = {'algorithm', 'episode_id', 'run_id', 'failure', 'unique_voxel_count'}
    check(stcols.issubset(epstats.columns) and len(epstats) == 8000 and
          not epstats[['algorithm','episode_id']].duplicated().any(),
          'Incomplete Step04 episode statistics')
    stats_map = {}
    for r in epstats.itertuples(index=False):
        n = int(r.unique_voxel_count)
        check(n > 0 and str(r.failure) in ('0', '1'),
              'Nonpositive voxel count or bad Step04 outcome')
        stats_map[(r.algorithm, r.episode_id)] = (n, str(r.failure), r.run_id)

    coverage = pd.read_csv(paths['coverage'])
    check(len(coverage) == 40 and
          {'algorithm','seed','train_unique_voxels','test_unique_voxels',
           'test_episode_voxel_visit_known_fraction',
           'test_unique_voxel_known_fraction'}.issubset(coverage.columns),
          'Step04 coverage summary missing expected fields')
    cov_map = {(r.algorithm, int(r.seed)): r for r in coverage.itertuples(index=False)}
    check(set(cov_map) == {(a, s) for a in ALGORITHMS for s in SEEDS},
          'Step04 coverage combinations mismatch')

    train_summary = pd.read_csv(paths['step5_summary'])
    check(len(train_summary) == 40 and
          {'algorithm','seed','training_covered_voxels','training_episodes',
           'test_episodes', 'consistency'}.issubset(train_summary.columns) and
          train_summary.consistency.eq('PASS').all(),
          'Step05 training summary invalid')
    train_map = {(r.algorithm,int(r.seed)):r for r in train_summary.itertuples(index=False)}
    check(set(train_map) == set(cov_map), 'Step05 summary combinations mismatch')

    split_map = {}
    for alg in ALGORITHMS:
        alg_df = manifest[manifest.algorithm == alg]
        check(len(alg_df) == 20000, alg + ': incomplete split rows')
        expected_ids = {eid for a,eid in stats_map if a == alg}
        check(len(expected_ids) == 2000, alg + ': incomplete episode statistics')
        for seed in SEEDS:
            rows = alg_df[alg_df.seed == str(seed)]
            check(len(rows) == 2000 and set(rows.episode_id) == expected_ids,
                  f'{alg}/{seed}: wrong episode IDs')
            tr = rows[rows.split == 'train']
            te = rows[rows.split == 'test']
            check(1380 <= len(tr) <= 1420 and 580 <= len(te) <= 620,
                  f'{alg}/{seed}: expected approximately 70/30 split')
            check(te.failure.nunique() == 2 and tr.failure.nunique() == 2,
                  f'{alg}/{seed}: needs both labels in train/test')
            check(not set(tr[GROUP_COLUMN]).intersection(set(te[GROUP_COLUMN])),
                  f'{alg}/{seed}: leaked scene group')
            for rec in rows.itertuples(index=False):
                exp = stats_map[(alg, rec.episode_id)]
                check(rec.failure == exp[1] and rec.run_id == exp[2],
                      f'{alg}/{seed}: episode label/run changed')
            split_map[(alg,seed)] = te[['episode_id','run_id','failure']].copy()
            check(int(train_map[(alg,seed)].training_episodes) == len(tr) and
                  int(train_map[(alg,seed)].test_episodes) == len(te),
                  f'{alg}/{seed}: Step05 split count mismatch')
    return paths, split_map, stats_map, cov_map, train_map


def load_visits(path, stats_map):
    by_algorithm = {alg: {} for alg in ALGORITHMS}
    with gzip.open(path, 'rt', encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        required = {'algorithm','episode_id','voxel_i','voxel_j','voxel_k'}
        check(required.issubset(reader.fieldnames or ()),
              'Step04 binary visits missing columns')
        for (alg, eid), group in itertools.groupby(
                reader, key=lambda row: (row['algorithm'],row['episode_id'])):
            check(alg in by_algorithm and (alg,eid) in stats_map,
                  'Step04 unexpected episode: ' + str((alg,eid)))
            check(eid not in by_algorithm[alg],
                  'Step04 visit file is not sorted/grouped by episode')
            voxels = set()
            for row in group:
                v = (int(row['voxel_i']),int(row['voxel_j']),int(row['voxel_k']))
                check(v not in voxels, f'{alg}/{eid}: duplicate binary voxel visit')
                voxels.add(v)
            check(len(voxels) == stats_map[(alg,eid)][0],
                  f'{alg}/{eid}: binary visits differ from Step04 stats')
            by_algorithm[alg][eid] = frozenset(voxels)
    for alg in ALGORITHMS:
        expected = {eid for a,eid in stats_map if a == alg}
        check(set(by_algorithm[alg]) == expected,
              alg + ': some episodes absent from the binary visits')
    return by_algorithm


def make_plots(splits, methods, example_seed11, figdir):
    # All plots are diagnostics; never tune scores or choose formulas here.
    labels = [a.replace('_200scenes','') for a in ALGORITHMS]
    fig, ax = plt.subplots(figsize=(9, 4.7))
    for alg in ALGORITHMS:
        sub = splits[splits.algorithm == alg].sort_values('seed')
        ax.plot(sub.seed, sub.known_voxel_visit_fraction, marker='o',
                label=alg.replace('_200scenes',''))
    ax.set_ylim(0, 1)
    ax.set_xlabel('Fixed split seed')
    ax.set_ylabel('Fraction of test episode-voxel visits known in training')
    ax.set_title('Pi0 200: held-out voxel familiarity by split')
    ax.legend(fontsize=8)
    save_fig(fig, figdir, '01_test_known_voxel_fraction_by_seed')

    fig, ax = plt.subplots(figsize=(9, 4.7))
    for alg in ALGORITHMS:
        sub = splits[splits.algorithm == alg].sort_values('seed')
        ax.plot(sub.seed, sub.no_known_episode_fraction, marker='o',
                label=alg.replace('_200scenes',''))
    ax.set_ylim(0, max(0.05, min(1, 1.15*float(splits.no_known_episode_fraction.max()))))
    ax.set_xlabel('Fixed split seed')
    ax.set_ylabel('Fraction of test episodes with zero training-known voxels')
    ax.set_title('Pi0 200: complete unseen-voxel fallback usage')
    ax.legend(fontsize=8)
    save_fig(fig, figdir, '02_episodes_with_no_known_voxels_by_seed')

    d = methods[methods.seed == 11]
    mat = np.asarray([[float(d[(d.algorithm == alg)&(d.method == met)]
                            .fraction_episodes_at_max_score.iloc[0])
                       for alg in ALGORITHMS] for met in METHODS])
    fig, ax = plt.subplots(figsize=(8.3, 6.0))
    img = ax.imshow(mat, aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(np.arange(4), labels, rotation=15)
    ax.set_yticks(np.arange(14), METHODS)
    ax.set_title('Pi0 200, seed 11: test episodes tied at max predicted risk')
    fig.colorbar(img, ax=ax, label='Fraction tied (0-1)')
    save_fig(fig, figdir, '03_fraction_max_score_ties_seed11')

    # Outcome is used only for this post-scoring descriptive plot; not fitting.
    categories, ticklabels = [], []
    for alg in ALGORITHMS:
        a = example_seed11[example_seed11.algorithm == alg]
        for failure,label in ((0,'Success'), (1,'Failure')):
            categories.append(a.loc[a.failure == failure, 'score_Ochiai'].to_numpy())
            ticklabels.append(alg.replace('_200scenes','')+'\n'+label)
    fig, ax = plt.subplots(figsize=(11, 5.2))
    try:
        ax.boxplot(categories, tick_labels=ticklabels, showfliers=False)
    except TypeError:
        ax.boxplot(categories, labels=ticklabels, showfliers=False)
    ax.tick_params(axis='x', labelrotation=20, labelsize=8)
    ax.set_ylabel('Raw Ochiai max known voxel suspiciousness')
    ax.set_title('Pi0 200, seed 11: held-out risk by observed outcome (descriptive)')
    save_fig(fig, figdir, '04_ochiai_heldout_risk_by_outcome_seed11')


def run(root):
    paths, splits, stats_map, cov_map, train_map = load_inputs(root)
    visits = load_visits(paths['visits'], stats_map)
    out = root / STEP7
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / 'test_episode_scores.csv.gz.incomplete'
    dest = out / 'test_episode_scores.csv.gz'
    if tmp.exists():
        tmp.unlink()
    fields = ['algorithm','seed','run_id','episode_id','failure',
              'episode_unique_voxels','known_voxels','unseen_voxels',
              'known_voxel_fraction','no_known_voxels'] + ['score_'+m for m in METHODS]
    split_rows = []
    method_rows = []
    example = []
    keys_seen = []
    total_predictions = 0
    fallback_total = 0
    score_voxel_count = 0
    try:
        with gzip.open(tmp, 'wt', encoding='utf-8', newline='', compresslevel=6) as outfile, \
             gzip.open(paths['scores'], 'rt', encoding='utf-8-sig', newline='') as infile:
            writer = csv.DictWriter(outfile, fieldnames=fields)
            writer.writeheader()
            reader = csv.DictReader(infile)
            needed = {'algorithm','seed','voxel_i','voxel_j','voxel_k','support'} | set(METHODS)
            check(needed.issubset(reader.fieldnames or ()),
                  'Step06 suspiciousness file missing expected columns')
            for (alg, seed_text), group in itertools.groupby(
                    reader, key=lambda r: (r['algorithm'],r['seed'])):
                check(seed_text.isdecimal(), 'Invalid Step06 seed: '+seed_text)
                seed = int(seed_text)
                key = (alg, seed)
                expected = [(a, s) for a in ALGORITHMS for s in SEEDS]
                check(len(keys_seen) < 40 and key == expected[len(keys_seen)],
                      f'Step06 score block order mismatch: {key}')
                keys_seen.append(key)
                voxel_map = {}
                for line in group:
                    voxel = (int(line['voxel_i']), int(line['voxel_j']),int(line['voxel_k']))
                    check(voxel not in voxel_map, f'{key}: duplicated training voxel')
                    check(int(line['support']) > 0, f'{key}: unsupported training voxel')
                    values = np.asarray([float(line[m]) for m in METHODS], dtype=np.float64)
                    check(np.isfinite(values).all(), f'{key}: nonfinite training score')
                    voxel_map[voxel] = values
                check(len(voxel_map) == int(train_map[key].training_covered_voxels) ==
                      int(cov_map[key].train_unique_voxels),
                      f'{key}: training voxel count mismatch')
                score_voxel_count += len(voxel_map)
                score_mat = np.vstack(list(voxel_map.values()))
                # Use an explicit margin below the minimum training score.
                # This survives CSV float parsing better than a one-ULP
                # nextafter sentinel, including when the minimum is zero.
                minimum = score_mat.min(axis=0)
                margin = 1e-6 * np.maximum(1.0, np.abs(minimum))
                fallback = minimum - margin
                check(np.isfinite(fallback).all() and (fallback < minimum).all(),
                      f'{key}: invalid below-minimum fallback')
                testing = splits[key].sort_values('episode_id', kind='stable')
                outputs = []
                union_test = set()
                known_visit_sum, visits_sum, none_known = 0, 0, 0
                for row in testing.itertuples(index=False):
                    vox = visits[alg][row.episode_id]
                    union_test.update(vox)
                    known = vox.intersection(voxel_map)
                    n_known = len(known)
                    n_total = len(vox)
                    check(n_total > 0, 'Unexpected empty test episode')
                    if n_known == 0:
                        pred = fallback.copy()
                        none_known += 1
                    else:
                        # Prediction computation uses ONLY voxel coordinates and
                        # the frozen training scores, never held-out outcomes.
                        pred = np.max(np.vstack([voxel_map[v] for v in known]),axis=0)
                    check(np.isfinite(pred).all(), f'{key}: invalid test prediction')
                    known_visit_sum += n_known
                    visits_sum += n_total
                    result = {
                        'algorithm':alg, 'seed':seed,
                        'run_id':row.run_id, 'episode_id':row.episode_id,
                        'episode_unique_voxels':n_total,
                        'known_voxels':n_known,
                        'unseen_voxels':n_total - n_known,
                        'known_voxel_fraction':n_known/n_total,
                        'no_known_voxels':int(n_known == 0),
                    }
                    for j,method in enumerate(METHODS):
                        result['score_'+method] = float(pred[j])
                    # Explicitly append truth last, after every risk prediction.
                    result['failure'] = int(row.failure)
                    writer.writerow(result)
                    outputs.append(result)
                    if seed == 11:
                        example.append(result.copy())
                total_predictions += len(outputs)
                fallback_total += none_known
                test_df = pd.DataFrame(outputs)
                overlap_unique = len(union_test.intersection(voxel_map))
                check(len(union_test) == int(cov_map[key].test_unique_voxels),
                      f'{key}: Step04 distinct test voxel count mismatch')
                known_unique_fraction = overlap_unique/len(union_test)
                known_visit_fraction = known_visit_sum/visits_sum
                check(math.isclose(known_visit_fraction,
                                   float(cov_map[key].test_episode_voxel_visit_known_fraction),
                                   abs_tol=1e-12, rel_tol=0),
                      f'{key}: Step04 known episode-voxel visit fraction differs')
                check(math.isclose(known_unique_fraction,
                                   float(cov_map[key].test_unique_voxel_known_fraction),
                                   abs_tol=1e-12, rel_tol=0),
                      f'{key}: Step04 known distinct voxel fraction differs')
                split_rows.append({
                    'algorithm':alg, 'seed':seed, 'test_episodes':len(outputs),
                    'test_failures':int(test_df.failure.sum()),
                    'test_successes':int((test_df.failure==0).sum()),
                    'train_known_voxels':len(voxel_map),
                    'unique_test_voxels':len(union_test),
                    'unique_test_voxels_known_in_training':overlap_unique,
                    'unique_test_voxels_known_fraction':known_unique_fraction,
                    'total_test_episode_voxel_visits':visits_sum,
                    'known_test_episode_voxel_visits':known_visit_sum,
                    'known_voxel_visit_fraction':known_visit_fraction,
                    'episodes_with_no_known_voxels':none_known,
                    'no_known_episode_fraction':none_known/len(outputs),
                    'mean_episode_known_voxel_fraction':float(test_df.known_voxel_fraction.mean()),
                })
                for j,method in enumerate(METHODS):
                    s = test_df['score_'+method].to_numpy(dtype=float)
                    mx = s.max()
                    method_rows.append({
                        'algorithm':alg, 'seed':seed, 'method':method,
                        'test_episodes':len(s), 'raw_risk_min':float(s.min()),
                        'raw_risk_max':float(mx),
                        'raw_risk_median':float(np.median(s)),
                        'distinct_predicted_scores':int(np.unique(s).size),
                        'episodes_at_max_score':int(np.sum(s == mx)),
                        'fraction_episodes_at_max_score':float(np.mean(s == mx)),
                        'episodes_with_unseen_only_voxels':none_known,
                        'below_min_training_fallback':float(fallback[j]),
                    })
                print(f'[SCORED] {alg} seed={seed}: test={len(outputs)}, '
                      f'known_visit_fraction={known_visit_fraction:.3f}, '
                      f'no_known_episodes={none_known}')
        check(len(keys_seen) == 40 and total_predictions == sum(len(x) for x in splits.values()),
              'Missing score maps or incomplete test predictions')
        check(len(method_rows) == 560 and len(split_rows) == 40,
              'Expected 40 split summaries and 560 method diagnostics')
        tmp.replace(dest)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise

    splits_df = pd.DataFrame(split_rows)
    method_df = pd.DataFrame(method_rows)
    example_df = pd.DataFrame(example)
    splits_df.to_csv(out/'test_scoring_split_summary.csv',index=False,encoding='utf-8')
    method_df.to_csv(out/'test_method_score_diagnostics.csv',index=False,encoding='utf-8')
    example_df.to_csv(out/'example_seed11_test_predictions.csv',index=False,encoding='utf-8')
    make_plots(splits_df,method_df,example_df,out/'figures')
    report = {
        'status':'PASS','step':'07_heldout_scoring',
        'model':'Pi0','budget':200,'algorithms':list(ALGORITHMS),
        'seeds':list(SEEDS),'methods':list(METHODS),
        'splits_scored':len(keys_seen), 'method_split_combinations':len(method_rows),
        'test_episode_predictions':total_predictions,
        'per_method_predictions':total_predictions*len(METHODS),
        'training_voxel_rows_consumed':score_voxel_count,
        'total_episodes_with_no_known_training_voxel_over_splits':fallback_total,
        'risk_aggregation':'max over training-known visited voxels',
        'unseen_voxel_policy':'ignore unseen; if zero known, use min(training score) - 1e-6*max(1,abs(min)) per method/split',
        'labels_used_to_construct_risk_scores':False,
        'truth_labels_in_predictions_file_for_step08':True,
        'candidate_scene_grouping_warning':'Full simulator-scene identity has not been independently verified.',
        'input_files_sha256':{k:digest(v) for k,v in paths.items()},
        'prediction_file_sha256':digest(dest),
        'output_files':{
            'predictions':str(dest.relative_to(root)),
            'summary':'test_scoring_split_summary.csv',
            'method_diagnostics':'test_method_score_diagnostics.csv',
            'example':'example_seed11_test_predictions.csv',
        },
    }
    (out/'step07_validation_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('\nRQ1 STEP 07 - HELD-OUT EPISODE SCORING')
    print(f'Splits: {len(keys_seen)}/40 | Techniques: {len(METHODS)} | '
          f'Method/split combinations: {len(method_rows)}/560')
    print(f'Test episode predictions: {total_predictions} | '
          f'Risk values: {total_predictions*len(METHODS)}')
    print(f'Episodes with no training-known voxels (across splits): {fallback_total}')
    print('Test outcomes used to calculate risk: NO')
    print('Status: PASS')
    print('Files saved to:',out)
    print('Next: Step08 calculate ROC-AUC and AP from these FROZEN test predictions.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root',type=Path,
                        default=Path(__file__).resolve().parent.parent,
                        help='Root folder containing input/, src/, output/')
    args=parser.parse_args()
    try:
        run(args.repo_root.resolve())
        return 0
    except (OSError, ValueError, KeyError, TypeError, AssertionError) as exc:
        print('BLOCKED:',str(exc),file=sys.stderr)
        return 1


if __name__=='__main__':
    raise SystemExit(main())
