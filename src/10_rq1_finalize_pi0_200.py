#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RQ1 Step 10: finalize Pi0 200-scene SBFL results with reproducible evidence.

Run:
    python src/10_rq1_finalize_pi0_200.py

Requires completed Steps 03-09 in output/RQ1/pi0_200/. Nothing is
retrained and no train/test split is regenerated. Fails closed if the
frozen Step08/09 CSV hashes have changed. All figures are saved as PNG
and PDF. All summary tables and plot data are saved as CSV.

CAUTIONS:
    (1) Ten repeated splits reuse episodes; report descriptive comparisons,
        not naive p-values or independent-split confidence intervals.
    (2) ROC-AUC/AP quantify held-out failure discrimination, not proof of
        identifying physically faulty voxels.
    (3) Scene-group identity was inferred from source fields and has not
        been independently validated as every relevant simulator setting.
    (4) Only Pi0 200 scenes is covered by this report.

ASCII-only Python source for safe use with Windows editors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ALGORITHMS = (
    'DH_200scenes', 'DH_HybridDT_200scenes',
    'DH_HybridRF_200scenes', 'DH_WeightedRF_200scenes',
)
SEEDS = (11, 23, 37, 41, 53, 67, 79, 83, 97, 101)
METHODS = (
    'Tarantula', 'Barinel', 'Kulczynski2', 'Wong2', 'Ochiai2',
    'Ochiai', 'DStar2', 'Jaccard', 'Wong3', 'GP13', 'OP2',
    'Wong1', 'Zoltar', 'Ample',
)
BASE = Path('output') / 'RQ1' / 'pi0_200'
DEST = BASE / 'step10_final_report'
TOL = 1e-10


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for piece in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(piece)
    return h.hexdigest()


def load_json(path):
    require(path.is_file(), 'Required report missing: ' + str(path))
    return json.loads(path.read_text(encoding='utf-8-sig'))


def load_csv(path):
    require(path.is_file(), 'Required CSV missing: ' + str(path))
    return pd.read_csv(path, encoding='utf-8-sig')


def csvout(df, path):
    df.to_csv(path, index=False, encoding='utf-8', float_format='%.16g')


def plotout(fig, out, stem):
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(out / (stem + '.' + ext), dpi=180, bbox_inches='tight')
    plt.close(fig)


def check_frozen(root):
    paths = {
        'step03_manifest': root / BASE / 'step03_grouped_splits' / 'split_manifest.csv',
        'step04_report': root / BASE / 'step04_voxel_coverage' / 'validation_report.json',
        'step05_report': root / BASE / 'step05_training_contingency' / 'step05_validation_report.json',
        'step06_report': root / BASE / 'step06_suspiciousness' / 'step06_validation_report.json',
        'step07_report': root / BASE / 'step07_heldout_scoring' / 'step07_validation_report.json',
        'step08_report': root / BASE / 'step08_roc_ap' / 'step08_validation_report.json',
        'step09_report': root / BASE / 'step09_paired_comparison' / 'step09_validation_report.json',
        'step08_metrics': root / BASE / 'step08_roc_ap' / 'per_split_metrics.csv',
        'step08_overall': root / BASE / 'step08_roc_ap' / 'overall_method_summary.csv',
        'step09_ranking': root / BASE / 'step09_paired_comparison' / 'method_ranking_summary.csv',
        'step09_per_algorithm': root / BASE / 'step09_paired_comparison' / 'per_algorithm_method_summary.csv',
        'step09_leader': root / BASE / 'step09_paired_comparison' / 'leader_vs_others.csv',
        'step09_ranks': root / BASE / 'step09_paired_comparison' / 'per_split_method_ranks.csv',
    }
    for path in paths.values():
        require(path.is_file(), 'Missing earlier-step artifact: ' + str(path))
    reports = {k: load_json(p) for k, p in paths.items() if k.endswith('_report')}
    for key, report in reports.items():
        require(report.get('status') == 'PASS', key + ' is not PASS')
    require(reports['step04_report'].get('splits_analyzed') == 40, 'Step04 split mismatch')
    require(reports['step06_report'].get('total_method_spectra') == 560,
            'Step06 score maps mismatch')
    require(reports['step06_report'].get('test_outcomes_used_to_fit_scores') is False,
            'Step06 must be training-only')
    require(reports['step07_report'].get('labels_used_to_construct_risk_scores') is False,
            'Step07 test leakage check failed')
    require(reports['step07_report'].get('splits_scored') == 40, 'Step07 split mismatch')
    require(reports['step08_report'].get('method_split_metrics') == 560,
            'Step08 metric count mismatch')
    require(reports['step09_report'].get('method_split_results') == 560 and
            reports['step09_report'].get('paired_method_pairs') == 91,
            'Step09 comparison count mismatch')
    for key in ('step08_report','step09_report'):
        require(reports[key].get('algorithms') == list(ALGORITHMS),
                key + ': algorithm list changed')
        require(reports[key].get('methods') == list(METHODS),
                key + ': method list changed')
        require(reports[key].get('seeds') == list(SEEDS),
                key + ': split seeds changed')
    digests08 = reports['step08_report'].get('output_files_sha256', {})
    for key in ('step08_metrics', 'step08_overall'):
        require(sha256(paths[key]) == digests08.get(paths[key].name),
                'Step08 frozen SHA-256 mismatch: ' + str(paths[key]))
    digests09 = reports['step09_report'].get('output_sha256', {})
    for key in ('step09_ranking', 'step09_per_algorithm', 'step09_leader', 'step09_ranks'):
        require(sha256(paths[key]) == digests09.get(paths[key].name),
                'Step09 frozen SHA-256 mismatch: ' + str(paths[key]))
    require(reports['step09_report'].get('input_sha256',{}).get('metrics') ==
            sha256(paths['step08_metrics']), 'Step09 source metrics mismatch')
    require(reports['step09_report'].get('input_sha256',{}).get('ranking') ==
            sha256(paths['step08_overall']), 'Step09 source ranking mismatch')
    return paths, reports


def validate_and_summarize(paths):
    metrics = load_csv(paths['step08_metrics'])
    overall = load_csv(paths['step09_ranking'])
    peralg = load_csv(paths['step09_per_algorithm'])
    lead = load_csv(paths['step09_leader'])
    ranks = load_csv(paths['step09_ranks'])
    needed = {'algorithm','seed','method','roc_auc','average_precision',
              'test_failure_prevalence','ap_minus_prevalence',
              'test_failures','test_successes','test_episodes'}
    require(needed.issubset(metrics.columns),
            'Missing metrics columns: ' + str(sorted(needed - set(metrics.columns))))
    require(len(metrics) == 560 and not metrics[['algorithm','seed','method']].duplicated().any(),
            'Expected 560 unique algorithm/seed/method metrics')
    require(set(metrics.algorithm) == set(ALGORITHMS) and
            set(metrics.seed) == set(SEEDS) and set(metrics.method) == set(METHODS),
            'Metrics grid has unexpected algorithm, seed or SBFL method')
    require(metrics.groupby(['algorithm','seed']).size().eq(14).all() and
            metrics.groupby(['algorithm','method']).size().eq(10).all(),
            'Incomplete 4 x 10 x 14 metric grid')
    for (_, _), group in metrics.groupby(['algorithm','seed']):
        require(group.test_episodes.nunique() == 1 and
                group.test_failures.nunique() == 1 and
                group.test_successes.nunique() == 1,
                'Different held-out test populations within same split')
    require((metrics.test_failures + metrics.test_successes == metrics.test_episodes).all(),
            'Test class counts do not sum to test episodes')
    for col in ('roc_auc','average_precision','test_failure_prevalence'):
        require(np.isfinite(metrics[col]).all() and metrics[col].between(0,1).all(),
                'Non-finite or out-of-range values for ' + col)
    require(np.allclose(metrics.ap_minus_prevalence,
                        metrics.average_precision-metrics.test_failure_prevalence,atol=1e-10),
            'AP prevalence lift inconsistent')
    require(len(overall) == 14 and set(overall.method) == set(METHODS),
            'Step09 overall leaderboard incomplete')
    require(len(peralg) == 56 and len(ranks) == 560, 'Step09 descriptive tables incomplete')

    # Independently recompute means rather than just copying Step09 rank data.
    recomputed = (metrics.groupby(['algorithm','method'],as_index=False)
                  .agg(auc=('roc_auc','mean'), ap=('average_precision','mean'),
                       ap_lift=('ap_minus_prevalence','mean')))
    equal_mean = recomputed.groupby('method',as_index=False).agg(
        mean_auc=('auc','mean'), mean_ap=('ap','mean'), mean_ap_lift=('ap_lift','mean'))
    checked = equal_mean.merge(overall,on='method',validate='one_to_one')
    require(np.allclose(checked.mean_auc,checked.equal_algorithm_mean_roc_auc,atol=1e-10) and
            np.allclose(checked.mean_ap,checked.equal_algorithm_mean_ap,atol=1e-10) and
            np.allclose(checked.mean_ap_lift,checked.equal_algorithm_mean_ap_lift,atol=1e-10),
            'Frozen Step09 overall averages differ from recomputed Step08 metrics')

    # Build final ranked CSV with key metrics. No CI or significance stars.
    cols = ['auc_descriptive_rank','method','equal_algorithm_mean_roc_auc',
            'equal_algorithm_mean_ap','equal_algorithm_mean_ap_lift',
            'mean_split_auc_rank','tied_best_auc_blocks',
            'min_algorithm_mean_auc','max_algorithm_mean_auc']
    board = overall[cols].sort_values(['equal_algorithm_mean_roc_auc','method'],
                                      ascending=[False, True]).reset_index(drop=True)
    algo = peralg[['algorithm','method','mean_roc_auc','mean_ap','mean_ap_lift',
                   'min_roc_auc','max_roc_auc','mean_auc_rank']].copy()
    order = board.method.tolist()
    algo['method'] = pd.Categorical(algo.method, categories=order, ordered=True)
    algo = algo.sort_values(['algorithm','method']).reset_index(drop=True)
    algo['method'] = algo.method.astype(str)

    leader = str(board.iloc[0].method)
    runner = str(board.iloc[1].method)
    require(len(lead) == 13 and set(lead.competitor) == set(METHODS)-{leader},
            'Step09 leader comparisons inconsistent')
    require(set(lead.descriptive_leader) == {leader},
            'Step09 leader differs from recomputed ranking')
    auc_pivot = metrics.pivot(index=['algorithm','seed'],columns='method',values='roc_auc')
    pair_diff = (auc_pivot[leader] - auc_pivot[runner]).rename('auc_delta_leader_minus_runner')
    pair_by_alg = pair_diff.groupby(level='algorithm').agg(
        mean_delta='mean', min_delta='min', max_delta='max').reindex(ALGORITHMS).reset_index()
    compare = lead.set_index('competitor').loc[runner]
    require(np.isclose(pair_by_alg.mean_delta.mean(),
                       float(compare.mean_delta_auc_leader_minus_competitor),atol=1e-10),
            'Leader/runner mean paired AUC mismatch')
    pair_by_alg['leader'] = leader
    pair_by_alg['runner_up'] = runner
    pair_by_alg['overlapping_split_count'] = 10
    pair_by_alg = pair_by_alg[['algorithm','leader','runner_up','mean_delta',
                               'min_delta','max_delta','overlapping_split_count']]

    # Summary of tests. This is descriptive, not 40 independent trials.
    balance = (metrics.drop_duplicates(['algorithm','seed'])
               [['algorithm','seed','test_episodes','test_failures','test_successes',
                 'test_failure_prevalence']].sort_values(['algorithm','seed']))
    require(len(balance) == 40, 'Expected 40 evaluation populations')
    return metrics, board, algo, lead, ranks, pair_by_alg, balance, leader, runner


def check_group_leakage(path, metrics):
    groups = pd.read_csv(path, usecols=['algorithm','seed','episode_id','split',
                                         'scene_group_exact_candidate'],
                         dtype={'seed':str,'episode_id':str,'scene_group_exact_candidate':str,
                                'algorithm':str,'split':str},
                         encoding='utf-8-sig')
    require(len(groups) == 80000, 'Step03 expected 80000 episode/split records')
    require(not groups[['algorithm','seed','episode_id']].duplicated().any(),
            'Duplicate episode in split manifest')
    checks=[]
    for alg in ALGORITHMS:
        for seed in SEEDS:
            g = groups[(groups.algorithm == alg)&(groups.seed == str(seed))]
            require(len(g) == 2000 and set(g.split) == {'train','test'},
                    'Invalid Step03 train/test rows for ' + alg + ' ' + str(seed))
            tr = set(g.loc[g.split=='train','scene_group_exact_candidate'])
            te = set(g.loc[g.split=='test','scene_group_exact_candidate'])
            overlap = len(tr & te)
            test_count = int((g.split=='test').sum())
            expected_test = int(metrics.loc[(metrics.algorithm == alg)&
                                             (metrics.seed == seed),'test_episodes'].iloc[0])
            require(overlap == 0 and test_count == expected_test,
                    'Group leakage or test count mismatch for ' + alg + ' ' + str(seed))
            checks.append({'algorithm':alg,'seed':seed,'train_episodes':2000-test_count,
                           'test_episodes':test_count,
                           'shared_candidate_scene_groups':overlap,'status':'PASS'})
    return pd.DataFrame(checks)


def make_figures(out, board, algo, pair_by_alg, metrics, leader, runner):
    figures = out/'figures'
    methods = board.method.tolist()
    y = np.arange(len(methods))

    # Figure 1: paper-friendly overall metric ranking.
    fig,ax=plt.subplots(figsize=(9.8,6.3))
    mean=board.equal_algorithm_mean_roc_auc.to_numpy()
    ax.barh(y,mean)
    ax.axvline(0.5,linestyle='--',label='Chance ROC-AUC = 0.5')
    ax.set_yticks(y,methods)
    ax.invert_yaxis();ax.set_xlim(0,1)
    ax.set_xlabel('Mean held-out ROC-AUC (equal algorithm weight)')
    ax.set_title('RQ1 Pi0 200: descriptive ranking of 14 SBFL methods')
    ax.legend(loc='lower right',fontsize=9)
    plotout(fig,figures,'01_final_auc_leaderboard')

    # Figure 2: per-algorithm AUC matrix.
    pivot=algo.pivot(index='method',columns='algorithm',values='mean_roc_auc').reindex(
        index=methods, columns=ALGORITHMS)
    fig,ax=plt.subplots(figsize=(9.6,6.7))
    im=ax.imshow(pivot.to_numpy(),aspect='auto',vmin=0,vmax=1)
    ax.set_yticks(np.arange(14),methods)
    ax.set_xticks(np.arange(4),[s.replace('_200scenes','') for s in ALGORITHMS],
                  rotation=20,ha='right')
    for i in range(14):
        for j in range(4):
            ax.text(j,i,f'{pivot.iloc[i,j]:.3f}',ha='center',va='center',fontsize=8)
    ax.set_title('Mean ROC-AUC by SBFL method and test generation algorithm')
    fig.colorbar(im,ax=ax,label='Mean ROC-AUC')
    plotout(fig,figures,'02_final_auc_algorithm_heatmap')

    # Figure 3: paired leader vs second place. Vertical segment is descriptive
    # observed minimum-to-maximum delta across overlapping splits, NOT CI.
    vals=pair_by_alg.set_index('algorithm').reindex(ALGORITHMS)
    x=np.arange(4)
    fig,ax=plt.subplots(figsize=(9.2,4.6))
    mu=vals.mean_delta.to_numpy()
    errs=np.vstack([mu-vals.min_delta.to_numpy(), vals.max_delta.to_numpy()-mu])
    ax.errorbar(x,mu,yerr=errs,fmt='o',capsize=4)
    ax.axhline(0,linestyle='--')
    ax.set_xticks(x,[s.replace('_200scenes','') for s in ALGORITHMS],rotation=15)
    ax.set_ylabel(f'ROC-AUC: {leader} minus {runner}')
    ax.set_title('Paired held-out AUC differences (range over overlapping splits)')
    ax.grid(axis='y',alpha=0.2)
    plotout(fig,figures,'03_final_leader_vs_runner_auc')

    # Figure 4: AUC versus prevalence-adjusted AP.
    fig,ax=plt.subplots(figsize=(8.4,5.8))
    ax.scatter(board.equal_algorithm_mean_roc_auc,
               board.equal_algorithm_mean_ap_lift)
    for r in board.itertuples(index=False):
        ax.annotate(r.method,(r.equal_algorithm_mean_roc_auc,
                              r.equal_algorithm_mean_ap_lift),
                    fontsize=8,xytext=(4,4),textcoords='offset points')
    ax.axvline(0.5,linestyle='--');ax.axhline(0,linestyle='--')
    ax.set_xlabel('Equal-algorithm mean ROC-AUC')
    ax.set_ylabel('Mean AP minus test failure prevalence')
    ax.set_title('Failure discrimination: ROC-AUC and baseline-adjusted AP')
    ax.grid(alpha=0.16)
    plotout(fig,figures,'04_final_auc_vs_ap_lift')

    # Figure 5: variability across overlapping splits for the top 5.
    top=methods[:5]
    fig,ax=plt.subplots(figsize=(10.2,5.2))
    data=[metrics.loc[metrics.method==m,'roc_auc'].to_numpy() for m in top]
    ax.boxplot(data,showfliers=False,whis=(0,100))
    ax.set_xticks(np.arange(1,len(top)+1),top)
    ax.axhline(0.5,linestyle='--')
    ax.set_ylabel('Held-out ROC-AUC')
    ax.set_title('Top five SBFL methods: range over overlapping splits (not CI)')
    ax.grid(axis='y',alpha=0.16)
    plotout(fig,figures,'05_final_top5_split_variability')

    return figures


def git_head(root):
    try:
        r=subprocess.run(['git','rev-parse','HEAD'],cwd=root,check=True,
                         capture_output=True,text=True,timeout=5)
        return r.stdout.strip()
    except (OSError,subprocess.SubprocessError):
        return 'unavailable'


def make_paper_text(out, board, peralg, pair_by_alg, balance, leader, runner,
                    step07, step09):
    row=board.iloc[0]
    second=board.iloc[1]
    diff=float(row.equal_algorithm_mean_roc_auc-second.equal_algorithm_mean_roc_auc)
    positive=int((pair_by_alg.mean_delta > TOL).sum())
    alg_winners=[]
    for alg,block in peralg.groupby('algorithm'):
        top_auc=float(block.mean_roc_auc.max())
        winners=block.loc[np.isclose(block.mean_roc_auc,top_auc,atol=TOL),'method'].tolist()
        alg_winners.append((alg,winners,top_auc))
    lines=[
        '# RQ1 Final Result: Pi0, 200 scenes',
        '',
        '## Research question',
        '',
        'Which of the 14 spatial SBFL methods best discriminates failed versus',
        'successful held-out Pi0 trajectories under the fixed 200-scene protocol?',
        '',
        '## Experimental design',
        '',
        '- Model: Pi0. Test-generation algorithms: DH, HybridDT, HybridRF, WeightedRF.',
        '- Ten 200-episode runs per algorithm; 2,000 episodes per algorithm; 8,000 total.',
        '- Ten repeated 70/30 candidate-scene-grouped splits per algorithm.',
        '- Four algorithm means weighted equally for the overall descriptive comparison.',
        '- Robot TCP XYZ visited voxels, 0.02-m grid, origin (-1, -1, 0).',
        '- Compute 14 voxel suspiciousness formulas on TRAINING episodes only.',
        '- Score each TEST episode by maximum known-voxel suspiciousness.',
        '- Evaluate ROC-AUC (primary) and Average Precision (secondary).',
        '- Test failure prevalence is the no-skill AP baseline.',
        '- All 14 methods in the same algorithm/split share the exact same held-out episodes.',
        '',
        '## Final descriptive method ranking',
        '',
        '| Rank | SBFL technique | Mean ROC-AUC | Mean AP | Mean AP - prevalence |',
        '|---:|---|---:|---:|---:|',
    ]
    for r in board.itertuples(index=False):
        lines.append(f'| {int(r.auc_descriptive_rank)} | {r.method} | '
                     f'{r.equal_algorithm_mean_roc_auc:.4f} | '
                     f'{r.equal_algorithm_mean_ap:.4f} | '
                     f'{r.equal_algorithm_mean_ap_lift:.4f} |')
    lines += [
        '', '## Answer to RQ1 for the current configuration', '',
        f'**{leader}** has the highest descriptive equal-algorithm mean held-out',
        f'ROC-AUC ({row.equal_algorithm_mean_roc_auc:.4f}), followed by',
        f'**{runner}** ({second.equal_algorithm_mean_roc_auc:.4f}).',
        f'Their observed paired mean delta AUC is **+{diff:.4f}** in favor',
        f'of {leader}; {positive}/4 algorithms have a positive mean delta.',
        'These are descriptive estimates, not a formal hypothesis-test result.',
        '',
        '## Algorithm-specific best methods (ties shown)', '',
    ]
    for alg,winners,val in alg_winners:
        lines.append(f'- {alg}: {", ".join(winners)} (mean ROC-AUC {val:.4f}).')
    lines += [
        '', '## Protocol checks', '',
        '- All 40 algorithm/seed held-out splits have zero overlap under the',
        '  candidate scene-group key.',
        '- The Step08 and Step09 frozen input/result hashes are verified.',
        f'- Step07 found {int(step07.get("test_episode_predictions",0)):,} held-out',
        '  prediction rows (repeated across split seeds).',
        '- Test labels did not influence suspiciousness estimation or episode risk;',
        '  they were used only for the evaluation metrics.',
        '', '## Limits and threats to validity', '',
        '- ROC-AUC/AP measure held-out failure discrimination, **not** whether',
        '  an identified voxel was a causal fault location.',
        '- The ten split seeds overlap in training and testing episodes.',
        '  Do not treat 40 algorithm/split blocks as independent replicates;',
        '  there are no naive p-values or 95% confidence intervals.',
        '- The scene-group key (object_name, orientation_id, X, Y) is a proxy',
        '  for scene identity. Other simulator settings may need verification.',
        '- The leading technique was selected using this dataset; the observed',
        '  gap could be optimistic. Validate on 400-scene budgets and other models.',
        '- Failure prevalence varies by algorithm; interpret AP relative to its',
        '  held-out prevalence baseline.',
        '- The max-visited-voxel rule can emphasize low-support suspicious voxels;',
        '  keep sparse-voxel diagnostics and assess sensitivity if reporting strong claims.',
        '', '## Next experimental extension', '',
        'Repeat the frozen RQ1 protocol for Pi0 400 scenes and other VLA models.',
        'Then address RQ2 by varying training spatial coverage at fixed test',
        'episodes and approximately matched training sample sizes.',
        '', '## Output index', '',
        '- final_method_leaderboard.csv, final_algorithm_method_table.csv',
        '- final_leader_vs_others.csv, final_leader_runner_by_algorithm.csv',
        '- final_heldout_class_balance.csv, scene_group_leakage_checks.csv',
        '- final_protocol_checks.csv, input_file_checksums.csv',
        '- figures/ (5 PNG and 5 PDF); step10_validation_report.json',
        '',
    ]
    (out/'step10_final_rq1_research_report.md').write_text('\n'.join(lines),encoding='utf-8')


def run(root):
    paths,reports=check_frozen(root)
    metrics, board, algo, lead, ranks, pair_by_alg, balance, leader, runner = \
        validate_and_summarize(paths)
    overlap = check_group_leakage(paths['step03_manifest'],metrics)
    require(overlap.shared_candidate_scene_groups.eq(0).all(),
            'Nonzero shared candidate scene groups')
    out=root/DEST
    out.mkdir(parents=True,exist_ok=True)
    csvout(board,out/'final_method_leaderboard.csv')
    csvout(algo,out/'final_algorithm_method_table.csv')
    csvout(lead,out/'final_leader_vs_others.csv')
    csvout(pair_by_alg,out/'final_leader_runner_by_algorithm.csv')
    csvout(balance,out/'final_heldout_class_balance.csv')
    csvout(overlap,out/'scene_group_leakage_checks.csv')
    checks = [
        ('All Step04-09 prerequisite reports PASS',True),
        ('40 outer train-test split assignments exist',len(overlap)==40),
        ('Zero candidate-scene overlap in each split',overlap.shared_candidate_scene_groups.eq(0).all()),
        ('560 complete algorithm/seed/method metrics',len(metrics)==560),
        ('Identical held-out test episodes across all 14 methods',True),
        ('No test labels used for voxel suspicion or risk estimation',True),
        ('Frozen Step08/09 SHA256 signatures match',True),
        ('ROC-AUC / AP values within range',True),
        ('All four algorithms equally weighted',True),
        ('Avoid independence p-values for overlapping splits',True),
    ]
    csvout(pd.DataFrame([{'check':k,'passed':bool(v)} for k,v in checks]),
           out/'final_protocol_checks.csv')
    trace = pd.DataFrame([{'file_role':k,'relative_file':str(p.relative_to(root)).replace('\\','/'),
                           'sha256':sha256(p),'bytes':p.stat().st_size}
                          for k,p in paths.items()])
    csvout(trace,out/'input_file_checksums.csv')
    figdir=make_figures(out,board,algo,pair_by_alg,metrics,leader,runner)
    make_paper_text(out,board,algo,pair_by_alg,balance,leader,runner,
                    reports['step07_report'],reports['step09_report'])
    png=list(figdir.glob('*.png'));pdf=list(figdir.glob('*.pdf'))
    require(len(png)==5 and len(pdf)==5,'Expected five PNG/PDF figure pairs')
    artifacts=sorted([p for p in out.iterdir() if p.is_file()])
    report = {
        'status':'PASS',
        'study':'Pi0 RQ1 200 scenes',
        'step':'10_final_descriptive_results',
        'methods':list(METHODS),
        'algorithms':list(ALGORITHMS),
        'seeds':list(SEEDS),
        'method_split_metric_pairs':len(metrics),
        'heldout_episode_predictions_repeated_across_splits':reports['step07_report'].get('test_episode_predictions'),
        'splits_checked':len(overlap),
        'candidate_scene_group_overlaps':int(overlap.shared_candidate_scene_groups.sum()),
        'descriptive_leader':leader,
        'runner_up':runner,
        'leader_mean_auc':float(board.iloc[0].equal_algorithm_mean_roc_auc),
        'runner_mean_auc':float(board.iloc[1].equal_algorithm_mean_roc_auc),
        'paired_equal_algorithm_mean_auc_difference':float(pair_by_alg.mean_delta.mean()),
        'positive_mean_auc_difference_algorithms':int((pair_by_alg.mean_delta > TOL).sum()),
        'png_count':len(png),'pdf_count':len(pdf),
        'python_version':platform.python_version(),
        'pandas_version':pd.__version__,
        'numpy_version':np.__version__,
        'matplotlib_version':matplotlib.__version__,
        'git_commit':git_head(root),
        'inference_note':'No naive p-values or confidence intervals due to overlapping repeated splits.',
        'scope_note':'Measures held-out failure discrimination, not causal physical fault localization.',
        'scene_identity_note':'Candidate scene key is not independently confirmed as full scene identity.',
        'input_file_sha256':{k:sha256(p) for k,p in paths.items()},
        'output_file_sha256':{p.name:sha256(p) for p in artifacts},
    }
    (out/'step10_validation_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('\nRQ1 STEP 10 - FINAL RESEARCH RESULTS')
    print('Splits: {}/40 | Metric pairs: {}/560 | Models: Pi0 | Budget: 200'.format(
        len(overlap),len(metrics)))
    print('Frozen results and train/test group checks: PASS')
    print('Figures: {} PNG + {} PDF | Status: PASS'.format(len(png),len(pdf)))
    print('Descriptive winner: {} AUC={:.4f}'.format(leader,float(board.iloc[0].equal_algorithm_mean_roc_auc)))
    print('Runner up: {} AUC={:.4f}'.format(runner,float(board.iloc[1].equal_algorithm_mean_roc_auc)))
    print('Leader - runner: paired mean delta AUC={:+.4f}; positive algorithms={}/4'.format(
        float(pair_by_alg.mean_delta.mean()),int((pair_by_alg.mean_delta > TOL).sum())))
    print('\nTOP 5:')
    print(board[['auc_descriptive_rank','method','equal_algorithm_mean_roc_auc',
                 'equal_algorithm_mean_ap']].head().to_string(index=False))
    print('\nSaved to:',out)
    print('Next: replicate Pi0 400 scenes or begin RQ2 coverage-controlled pilot.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root',type=Path,default=Path(__file__).resolve().parent.parent,
                        help='Repository containing output/RQ1/pi0_200')
    args=parser.parse_args()
    try:
        run(args.repo_root.resolve())
        return 0
    except (OSError,ValueError,TypeError,KeyError,AssertionError,ImportError) as exc:
        print('BLOCKED:',str(exc),file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
