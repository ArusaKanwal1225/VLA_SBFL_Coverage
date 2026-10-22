#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RQ1 Step 09: paired descriptive comparison of 14 SBFL methods on Pi0/200.

Run (Python 3.9+):
    python src/09_rq1_paired_method_comparison_pi0_200.py

Input, unchanged:
    output/RQ1/pi0_200/step08_roc_ap/per_split_metrics.csv
    output/RQ1/pi0_200/step08_roc_ap/overall_method_summary.csv
    output/RQ1/pi0_200/step08_roc_ap/step08_validation_report.json

Outputs saved to output/RQ1/pi0_200/step09_paired_comparison/:
    method_ranking_summary.csv
    per_algorithm_method_summary.csv
    per_split_method_ranks.csv
    paired_comparisons_by_split.csv.gz
    paired_comparisons_by_algorithm.csv
    paired_comparisons_overall.csv
    leader_vs_others.csv
    winner_counts_by_method.csv
    step09_research_summary.md
    step09_validation_report.json
    figures/*.png and figures/*.pdf (8 of each)

Methodological guardrail: ten repeated 70:30 splits share episodes. The
40 split blocks are NOT independent samples. Therefore do not apply naive
Wilcoxon/t-tests or report nominal p-values/95% confidence intervals here.
AUC and AP measure held-out FAILURE DISCRIMINATION rather than true fault
localization, and scene grouping remains subject to source-verification caveats.

ASCII-only file to avoid Windows source-code encoding errors.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
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
SOURCE = BASE / 'step08_roc_ap'
DEST = BASE / 'step09_paired_comparison'
TIE_TOL = 1e-10


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checksum(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for piece in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(piece)
    return digest.hexdigest()


def csvout(frame, path):
    frame.to_csv(path, index=False, encoding='utf-8', float_format='%.16g')


def plotout(fig, path_without_suffix):
    fig.tight_layout()
    for ext in ('.png', '.pdf'):
        fig.savefig(str(path_without_suffix) + ext, dpi=175, bbox_inches='tight')
    plt.close(fig)


def short_alg(name):
    return name.replace('_200scenes', '')


def read_and_validate(root):
    indir = root / SOURCE
    paths = {key: indir / name for key, name in {
        'metrics': 'per_split_metrics.csv',
        'ranking': 'overall_method_summary.csv',
        'report': 'step08_validation_report.json',
    }.items()}
    for path in paths.values():
        require(path.is_file(), 'Missing Step08 file: ' + str(path))
    report = json.loads(paths['report'].read_text(encoding='utf-8-sig'))
    require(report.get('status') == 'PASS', 'Step08 status is not PASS')
    require(report.get('method_split_metrics') == 560 and
            report.get('splits_evaluated') == 40, 'Step08 did not validate 560 / 40')
    require(report.get('algorithms') == list(ALGORITHMS), 'Algorithm mismatch')
    require(report.get('seeds') == list(SEEDS), 'Split-seed mismatch')
    require(report.get('methods') == list(METHODS), 'SBFL method mismatch')
    digests = report.get('output_files_sha256', {})
    for key, name in [('metrics', 'per_split_metrics.csv'),
                      ('ranking', 'overall_method_summary.csv')]:
        require(digests.get(name) == checksum(paths[key]),
                'Step08 frozen file hash mismatch: ' + name)

    table = pd.read_csv(paths['metrics'], encoding='utf-8-sig')
    existing = pd.read_csv(paths['ranking'], encoding='utf-8-sig')
    needed = {'algorithm', 'seed', 'method', 'roc_auc', 'average_precision',
              'test_episodes', 'test_failures', 'test_successes',
              'test_failure_prevalence', 'ap_minus_prevalence'}
    require(needed.issubset(table.columns), 'Missing metric columns: ' +
            str(sorted(needed.difference(table.columns))))
    require(len(table) == 560 and not table[['algorithm','seed','method']].duplicated().any(),
            'Step08 metrics must contain 560 unique algorithm/seed/method rows')
    expected = set(itertools.product(ALGORITHMS, SEEDS, METHODS))
    observed = set(zip(table.algorithm, table.seed, table.method))
    require(expected == observed, 'Algorithm/seed/method grid is incomplete')
    require(table[['roc_auc','average_precision','test_failure_prevalence',
                   'ap_minus_prevalence']].notna().all().all(),
            'Missing measured metric values')
    for col in ['roc_auc','average_precision','test_failure_prevalence']:
        require(np.isfinite(table[col]).all() and table[col].between(0,1).all(),
                'Invalid or non-finite values in ' + col)
    require(np.allclose(table.ap_minus_prevalence,
                        table.average_precision-table.test_failure_prevalence,
                        atol=1e-11), 'AP lift formula inconsistency')
    require((table.test_failures + table.test_successes == table.test_episodes).all(),
            'Bad failure/pass test episode counts')
    for (_, _), group in table.groupby(['algorithm','seed']):
        require(len(group) == 14 and group.test_episodes.nunique() == 1 and
                group.test_failures.nunique() == 1 and
                group.test_successes.nunique() == 1,
                'Within-split methods have different evaluation populations')
    require(len(existing) == 14 and set(existing.method) == set(METHODS),
            'Step08 overall ranking incomplete')
    return table, existing, paths, report


def compute_ranks_and_summaries(table, prior):
    ranks = table.copy()
    ranks['roc_auc_rank_within_split'] = ranks.groupby(['algorithm','seed']).roc_auc.rank(
        ascending=False, method='average')
    ranks['ap_rank_within_split'] = ranks.groupby(['algorithm','seed']).average_precision.rank(
        ascending=False, method='average')
    max_auc = ranks.groupby(['algorithm','seed']).roc_auc.transform('max')
    ranks['tied_for_best_roc_auc'] = (max_auc - ranks.roc_auc).abs() <= TIE_TOL
    max_ap = ranks.groupby(['algorithm','seed']).average_precision.transform('max')
    ranks['tied_for_best_ap'] = (max_ap - ranks.average_precision).abs() <= TIE_TOL
    ranks['within_top3_auc'] = ranks.roc_auc_rank_within_split <= 3.0 + TIE_TOL
    ranks['within_top3_ap'] = ranks.ap_rank_within_split <= 3.0 + TIE_TOL

    per_alg = (ranks.groupby(['algorithm','method'], as_index=False).agg(
        split_count=('seed','nunique'),
        mean_roc_auc=('roc_auc','mean'),
        min_roc_auc=('roc_auc','min'),
        max_roc_auc=('roc_auc','max'),
        sd_roc_auc_across_overlapping_splits=('roc_auc','std'),
        mean_ap=('average_precision','mean'),
        mean_ap_lift=('ap_minus_prevalence','mean'),
        mean_auc_rank=('roc_auc_rank_within_split','mean'),
        mean_ap_rank=('ap_rank_within_split','mean'),
        tied_best_auc_count=('tied_for_best_roc_auc','sum'),
        tied_best_ap_count=('tied_for_best_ap','sum'),
        top3_auc_count=('within_top3_auc','sum'),
        top3_ap_count=('within_top3_ap','sum')))
    require(len(per_alg) == 56 and per_alg.split_count.eq(10).all(),
            'Expected 56 algorithm/method summaries')
    per_alg['auc_rank_within_algorithm'] = per_alg.groupby('algorithm').mean_roc_auc.rank(
        ascending=False, method='average')

    overall = (per_alg.groupby('method', as_index=False).agg(
        algorithms=('algorithm','nunique'),
        equal_algorithm_mean_roc_auc=('mean_roc_auc','mean'),
        min_algorithm_mean_auc=('mean_roc_auc','min'),
        max_algorithm_mean_auc=('mean_roc_auc','max'),
        between_algorithm_sd_auc=('mean_roc_auc','std'),
        equal_algorithm_mean_ap=('mean_ap','mean'),
        equal_algorithm_mean_ap_lift=('mean_ap_lift','mean'),
        mean_split_auc_rank=('mean_auc_rank','mean'),
        mean_split_ap_rank=('mean_ap_rank','mean'),
        tied_best_auc_blocks=('tied_best_auc_count','sum'),
        tied_best_ap_blocks=('tied_best_ap_count','sum'),
        top3_auc_blocks=('top3_auc_count','sum'),
        top3_ap_blocks=('top3_ap_count','sum'),
        mean_algorithm_auc_rank=('auc_rank_within_algorithm','mean')))
    overall['auc_descriptive_rank'] = overall.equal_algorithm_mean_roc_auc.rank(
        ascending=False, method='min').astype(int)
    overall = overall.sort_values(['equal_algorithm_mean_roc_auc', 'equal_algorithm_mean_ap', 'method'],
                                  ascending=[False,False,True]).reset_index(drop=True)
    prior_scores = prior.set_index('method').equal_algorithm_mean_roc_auc
    for row in overall.itertuples(index=False):
        require(math.isclose(row.equal_algorithm_mean_roc_auc,
                             float(prior_scores[row.method]), abs_tol=1e-10),
                'Recomputed overall AUC does not match frozen Step08 for ' + row.method)
    return ranks, per_alg, overall


def paired_analysis(table, overall):
    # Each row refers to two methods evaluated on the SAME test episodes.
    pivot_auc = table.pivot(index=['algorithm','seed'], columns='method', values='roc_auc')
    pivot_ap = table.pivot(index=['algorithm','seed'], columns='method', values='average_precision')
    paired = []
    for a,b in itertools.combinations(sorted(METHODS),2):
        dauc = pivot_auc[a] - pivot_auc[b]
        dap = pivot_ap[a] - pivot_ap[b]
        for (algorithm,seed), value in dauc.items():
            ap_value = float(dap.loc[(algorithm,seed)])
            paired.append({'method_a':a, 'method_b':b,
                           'algorithm':algorithm, 'seed':int(seed),
                           'delta_auc_a_minus_b':float(value),
                           'delta_ap_a_minus_b':ap_value,
                           'auc_win_a':int(value > TIE_TOL),
                           'auc_tie':int(abs(value) <= TIE_TOL),
                           'auc_win_b':int(value < -TIE_TOL),
                           'ap_win_a':int(ap_value > TIE_TOL),
                           'ap_tie':int(abs(ap_value) <= TIE_TOL),
                           'ap_win_b':int(ap_value < -TIE_TOL)})
    pairs = pd.DataFrame(paired)
    require(len(pairs) == 91*40, 'Expected 91 method pairs x 40 paired blocks')

    by_alg = pairs.groupby(['method_a','method_b','algorithm'], as_index=False).agg(
        mean_delta_auc=('delta_auc_a_minus_b','mean'),
        median_delta_auc=('delta_auc_a_minus_b','median'),
        min_delta_auc=('delta_auc_a_minus_b','min'),
        max_delta_auc=('delta_auc_a_minus_b','max'),
        mean_delta_ap=('delta_ap_a_minus_b','mean'),
        wins_a_auc=('auc_win_a','sum'),
        ties_auc=('auc_tie','sum'),
        wins_b_auc=('auc_win_b','sum'),
        wins_a_ap=('ap_win_a','sum'),
        ties_ap=('ap_tie','sum'),
        wins_b_ap=('ap_win_b','sum'))
    require(len(by_alg) == 91*4, 'Expected 364 paired-by-algorithm comparisons')
    by_alg['overlapping_split_count'] = 10

    across = by_alg.groupby(['method_a','method_b'], as_index=False).agg(
        mean_delta_auc_equal_algorithms=('mean_delta_auc','mean'),
        min_algorithm_mean_delta_auc=('mean_delta_auc','min'),
        max_algorithm_mean_delta_auc=('mean_delta_auc','max'),
        mean_delta_ap_equal_algorithms=('mean_delta_ap','mean'),
        auc_positive_algorithms=('mean_delta_auc',lambda a: int((a>TIE_TOL).sum())),
        auc_tied_algorithms=('mean_delta_auc',lambda a: int((a.abs()<=TIE_TOL).sum())),
        auc_negative_algorithms=('mean_delta_auc',lambda a: int((a<-TIE_TOL).sum())),
        auc_wins_a_across_overlapping_splits=('wins_a_auc','sum'),
        auc_ties_across_overlapping_splits=('ties_auc','sum'),
        auc_wins_b_across_overlapping_splits=('wins_b_auc','sum'),
        ap_wins_a_across_overlapping_splits=('wins_a_ap','sum'),
        ap_ties_across_overlapping_splits=('ties_ap','sum'),
        ap_wins_b_across_overlapping_splits=('wins_b_ap','sum'))
    require(len(across) == 91, 'Expected 91 overall method pairs')
    leader = str(overall.iloc[0].method)
    lead_alg_auc = (table[table.method==leader]
                    .set_index(['algorithm','seed']).roc_auc)
    lead_alg_ap = (table[table.method==leader]
                   .set_index(['algorithm','seed']).average_precision)
    rows=[]
    for rival in METHODS:
        if rival == leader:
            continue
        rival_auc = table[table.method==rival].set_index(['algorithm','seed']).roc_auc
        rival_ap = table[table.method==rival].set_index(['algorithm','seed']).average_precision
        diff_auc = lead_alg_auc - rival_auc
        diff_ap = lead_alg_ap - rival_ap
        alg_diff_auc = diff_auc.groupby(level='algorithm').mean().reindex(ALGORITHMS)
        alg_diff_ap = diff_ap.groupby(level='algorithm').mean().reindex(ALGORITHMS)
        rows.append({
            'descriptive_leader':leader,'competitor':rival,
            'leader_mean_auc':float(overall.set_index('method').loc[leader,'equal_algorithm_mean_roc_auc']),
            'competitor_mean_auc':float(overall.set_index('method').loc[rival,'equal_algorithm_mean_roc_auc']),
            'mean_delta_auc_leader_minus_competitor':float(alg_diff_auc.mean()),
            'mean_delta_ap_leader_minus_competitor':float(alg_diff_ap.mean()),
            'positive_auc_algorithms':int((alg_diff_auc>TIE_TOL).sum()),
            'tied_auc_algorithms':int((alg_diff_auc.abs()<=TIE_TOL).sum()),
            'negative_auc_algorithms':int((alg_diff_auc<-TIE_TOL).sum()),
            'positive_auc_overlapping_splits':int((diff_auc>TIE_TOL).sum()),
            'tied_auc_overlapping_splits':int((diff_auc.abs()<=TIE_TOL).sum()),
            'negative_auc_overlapping_splits':int((diff_auc<-TIE_TOL).sum()),
            'positive_ap_overlapping_splits':int((diff_ap>TIE_TOL).sum()),
            'tied_ap_overlapping_splits':int((diff_ap.abs()<=TIE_TOL).sum()),
            'negative_ap_overlapping_splits':int((diff_ap<-TIE_TOL).sum()),
            **{'delta_auc_'+a:float(alg_diff_auc[a]) for a in ALGORITHMS},
        })
    leader_vs = pd.DataFrame(rows).sort_values('mean_delta_auc_leader_minus_competitor',
                                                ascending=False).reset_index(drop=True)
    require(len(leader_vs) == 13, 'Expected 13 leader comparisons')
    return pairs, by_alg, across, leader_vs, leader


def make_plots(out, ranks, per_alg, overall, leader_vs, leader, table):
    figdir = out/'figures'
    figdir.mkdir(parents=True, exist_ok=True)
    order = overall.method.tolist()
    labels = [m for m in order]
    y = np.arange(14)
    # 01: overall AUC with min/max across the four algorithm means.
    mean = overall.equal_algorithm_mean_roc_auc.to_numpy()
    lo = overall.min_algorithm_mean_auc.to_numpy()
    hi = overall.max_algorithm_mean_auc.to_numpy()
    fig,ax=plt.subplots(figsize=(10,6.8))
    ax.barh(y,mean)
    ax.errorbar(mean,y,xerr=np.vstack([mean-lo,hi-mean]),fmt='none',
                ecolor='black',elinewidth=1.2,capsize=3,
                label='Range of 4 algorithm means (NOT a CI)')
    ax.set_yticks(y,labels);ax.invert_yaxis()
    ax.axvline(.5,ls='--',color='gray',lw=1,label='Chance ranking AUC = 0.5')
    ax.set_xlim(0,1.02);ax.set_xlabel('Equal-algorithm mean held-out ROC-AUC')
    ax.set_title('RQ1: descriptive AUC ranking with algorithm range')
    ax.legend(fontsize=9,loc='lower right')
    plotout(fig,figdir/'01_descriptive_auc_with_algorithm_range')

    # 02: method x algorithm AUC heatmap.
    mat = per_alg.pivot(index='method',columns='algorithm',values='mean_roc_auc').reindex(
        index=order,columns=ALGORITHMS)
    fig,ax=plt.subplots(figsize=(9.7,7))
    im=ax.imshow(mat.to_numpy(),aspect='auto',vmin=0,vmax=1)
    ax.set_yticks(np.arange(14),order)
    ax.set_xticks(np.arange(4),[short_alg(a) for a in ALGORITHMS],rotation=20,ha='right')
    for i in range(14):
        for j in range(4):
            ax.text(j,i,f'{mat.iloc[i,j]:.3f}',ha='center',va='center',fontsize=8)
    ax.set_title('RQ1: mean ROC-AUC across 10 splits per algorithm')
    fig.colorbar(im,ax=ax,label='Mean ROC-AUC')
    plotout(fig,figdir/'02_algorithm_method_auc_heatmap')

    # 03: ranks across repeated splits (descriptive, not independent samples).
    fig,ax=plt.subplots(figsize=(11,5.3))
    data = [ranks.loc[ranks.method==m,'roc_auc_rank_within_split'].to_numpy() for m in order]
    ax.boxplot(data,positions=np.arange(1,15),showfliers=False,whis=(0,100))
    ax.set_xticks(np.arange(1,15),order,rotation=55,ha='right')
    ax.set_ylabel('ROC-AUC rank in algorithm/split (lower is better)')
    ax.set_title('Rank variability across overlapping splits (descriptive)')
    ax.grid(axis='y',alpha=.18)
    plotout(fig,figdir/'03_splitwise_auc_rank_variability')

    # 04: per-algorithm mean paired difference between leader and each rival.
    delta_cols = ['delta_auc_'+a for a in ALGORITHMS]
    view = leader_vs.set_index('competitor').reindex([m for m in order if m != leader])
    heat=view[delta_cols].to_numpy()
    lim=max(float(np.max(np.abs(heat))),0.01)
    fig,ax=plt.subplots(figsize=(9.7,6.7))
    im=ax.imshow(heat,aspect='auto',cmap='coolwarm',vmin=-lim,vmax=lim)
    ax.set_yticks(np.arange(13),view.index.tolist())
    ax.set_xticks(np.arange(4),[short_alg(a) for a in ALGORITHMS],rotation=20,ha='right')
    for i in range(13):
        for j in range(4):
            ax.text(j,i,f'{heat[i,j]:+.3f}',ha='center',va='center',fontsize=8)
    ax.set_title(f'Paired mean AUC differences: {leader} minus competitor')
    fig.colorbar(im,ax=ax,label='Delta AUC (positive favors leader)')
    plotout(fig,figdir/'04_leader_paired_auc_delta_heatmap')

    # 05: split-by-split paired difference between top two descriptive methods.
    competitor=order[1]
    pv=table.pivot(index=['algorithm','seed'],columns='method',values='roc_auc')
    fig,ax=plt.subplots(figsize=(9.5,5.0))
    for alg in ALGORITHMS:
        block=pv.xs(alg,level='algorithm')
        ax.plot(block.index.to_numpy(),(block[leader]-block[competitor]).to_numpy(),
                marker='o',lw=1.2,label=short_alg(alg))
    ax.axhline(0,color='gray',ls='--',lw=1)
    ax.set_xlabel('Repeated split seed (overlapping test sets)')
    ax.set_ylabel(f'Delta AUC: {leader} minus {competitor}')
    ax.set_title('Within-identical-test-set comparisons: leader vs runner-up')
    ax.legend(fontsize=8);ax.grid(alpha=.15)
    plotout(fig,figdir/'05_leader_runner_up_auc_by_split')

    # 06: mean AP lift over test prevalence, rather than raw AP alone.
    sorted_ap=overall.sort_values('equal_algorithm_mean_ap_lift',ascending=True)
    fig,ax=plt.subplots(figsize=(9.1,6.6))
    ax.barh(sorted_ap.method,sorted_ap.equal_algorithm_mean_ap_lift)
    ax.axvline(0,color='gray',ls='--',lw=1)
    ax.set_xlabel('Mean(AP - test failure prevalence), equal algorithm weight')
    ax.set_title('Average Precision improvement over each test-set baseline')
    plotout(fig,figdir/'06_average_precision_lift_ranking')

    # 07: compare AUC and prevalence-adjusted AP rankings.
    fig,ax=plt.subplots(figsize=(8.6,6))
    ax.scatter(overall.equal_algorithm_mean_roc_auc,
               overall.equal_algorithm_mean_ap_lift,s=42)
    for r in overall.itertuples(index=False):
        ax.annotate(r.method,(r.equal_algorithm_mean_roc_auc,r.equal_algorithm_mean_ap_lift),
                    xytext=(4,4),textcoords='offset points',fontsize=8)
    ax.axvline(.5,ls='--',color='gray',lw=1)
    ax.axhline(0,ls='--',color='gray',lw=1)
    ax.set_xlabel('Equal-algorithm mean ROC-AUC')
    ax.set_ylabel('Equal-algorithm mean (AP - prevalence)')
    ax.set_title('ROC-AUC versus prevalence-adjusted Average Precision')
    ax.grid(alpha=.15)
    plotout(fig,figdir/'07_auc_vs_ap_lift')

    # 08: fraction of algorithm/split blocks tied for best AUC.
    fig,ax=plt.subplots(figsize=(9.4,6.6))
    wins=overall.set_index('method').reindex(order).tied_best_auc_blocks.to_numpy()
    ax.barh(y,wins)
    ax.set_yticks(y,order);ax.invert_yaxis()
    ax.set_xlim(0,40);ax.set_xticks(np.arange(0,41,5))
    ax.set_xlabel('Out of 40 method-comparison blocks (ties counted for each method)')
    ax.set_title('Times each method tied for best ROC-AUC (descriptive)')
    plotout(fig,figdir/'08_auc_leader_count_overlapping_splits')
    return figdir


def make_summary(out, leader, overall, lead_vs, report):
    second = str(overall.iloc[1].method)
    pair=lead_vs.set_index('competitor').loc[second]
    def f(v):
        return f'{float(v):.4f}'
    text = [
        '# RQ1 Step 09 - Paired descriptive comparison',
        '',
        '## Experimental scope',
        '',
        '- Model: Pi0; scene budget: 200; four algorithms; ten repeated 70/30 splits each.',
        '- Methods: 14; observations: 560 algorithm x split x method ROC-AUC/AP results.',
        '- All methods within each algorithm/seed are evaluated on identical held-out episodes.',
        '- The repeated splits overlap: results are descriptive, not 40 independent replications.',
        '',
        '## Descriptive findings',
        '',
        f'- Highest equal-algorithm mean ROC-AUC: **{leader}** ({f(overall.iloc[0].equal_algorithm_mean_roc_auc)}).',
        f'- Second-highest: **{second}** ({f(overall.iloc[1].equal_algorithm_mean_roc_auc)}).',
        f'- Paired mean ROC-AUC difference, {leader} - {second}: **{f(pair.mean_delta_auc_leader_minus_competitor)}**.',
        '- Positive/negative algorithm mean differences (leader vs runner-up): '
        f'{int(pair.positive_auc_algorithms)}/{int(pair.negative_auc_algorithms)} '
        f'({int(pair.tied_auc_algorithms)} ties).',
        '- The figures and tables also report the prevalence-adjusted AP lift.',
        '',
        '## Interpretation limitations',
        '',
        '- No naive p-values, significance stars or confidence intervals are reported here.',
        '- Using overlapping 70/30 splits as independent observations would overstate evidence.',
        '- These results evaluate held-out **failure discrimination**; they do not identify ground-truth faulty voxels.',
        '- Scene-group identity was inferred from source fields and has not been independently verified in full.',
        '',
        '## Next step',
        '',
        'Step 10: protocol checks, numerical summary, and careful RQ1 write-up. '
        'Consider a new independent dataset or other-model/budget replication before broad claims.',
        '',
    ]
    (out/'step09_research_summary.md').write_text('\n'.join(text),encoding='utf-8')


def run(root):
    table, prior, inputs, report08 = read_and_validate(root)
    ranks, per_alg, overall = compute_ranks_and_summaries(table, prior)
    pairs, by_alg, across, leader_vs, leader = paired_analysis(table, overall)
    out=root/DEST
    out.mkdir(parents=True,exist_ok=True)
    csvout(overall,out/'method_ranking_summary.csv')
    csvout(per_alg,out/'per_algorithm_method_summary.csv')
    csvout(ranks,out/'per_split_method_ranks.csv')
    pairs.to_csv(out/'paired_comparisons_by_split.csv.gz',index=False,
                 compression='gzip',float_format='%.16g')
    csvout(by_alg,out/'paired_comparisons_by_algorithm.csv')
    csvout(across,out/'paired_comparisons_overall.csv')
    csvout(leader_vs,out/'leader_vs_others.csv')
    csvout(overall[['method','tied_best_auc_blocks','tied_best_ap_blocks',
                    'top3_auc_blocks','top3_ap_blocks']],out/'winner_counts_by_method.csv')
    make_plots(out,ranks,per_alg,overall,leader_vs,leader,table)
    make_summary(out,leader,overall,leader_vs,report08)
    artifacts = [
        'method_ranking_summary.csv', 'per_algorithm_method_summary.csv',
        'per_split_method_ranks.csv', 'paired_comparisons_by_split.csv.gz',
        'paired_comparisons_by_algorithm.csv', 'paired_comparisons_overall.csv',
        'leader_vs_others.csv','winner_counts_by_method.csv',
        'step09_research_summary.md',
    ]
    figs=list((out/'figures').glob('*.png'))
    pdfs=list((out/'figures').glob('*.pdf'))
    require(len(figs)==8 and len(pdfs)==8, 'Expected 8 PNG/PDF figure pairs')
    report={
        'status':'PASS','step':'09_paired_descriptive_comparison',
        'model':'Pi0','budget':200,'algorithms':list(ALGORITHMS),
        'seeds':list(SEEDS),'methods':list(METHODS),
        'method_split_results':len(table),
        'paired_method_pairs':len(across),
        'paired_algorithm_split_rows':len(pairs),
        'algorithm_method_summaries':len(per_alg),
        'descriptive_leader':leader,
        'leading_equal_algorithm_mean_auc':float(overall.iloc[0].equal_algorithm_mean_roc_auc),
        'figures_png':len(figs),'figures_pdf':len(pdfs),
        'split_dependence_warning':'Ten 70/30 splits overlap within each algorithm. Descriptive analysis only; no naive p-values or CIs.',
        'scope_warning':'ROC-AUC/AP measure held-out failure discrimination, not ground-truth fault localization accuracy.',
        'scene_grouping_warning':report08.get('scene_identity_caution',''),
        'no_new_train_test_splits_or_fitting':True,
        'identical_test_episodes_compared_per_algorithm_seed':True,
        'input_sha256':{key:checksum(path) for key,path in inputs.items()},
        'output_sha256':{name:checksum(out/name) for name in artifacts},
    }
    (out/'step09_validation_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('\nRQ1 STEP 09 - PAIRED DESCRIPTIVE METHOD COMPARISON')
    print('Method/split results: {}/560 | Paired methods: {}/91 | Paired split rows: {}/3640'.format(
        len(table),len(across),len(pairs)))
    print('Algorithms: 4 | Splits per algorithm: 10 | Figures: {}/{} (PNG/PDF)'.format(len(figs),len(pdfs)))
    print('No independent-split significance tests or p-values reported.')
    print('Status: PASS')
    print('Outputs:',out)
    print('\nTOP 5 BY EQUAL-ALGORITHM MEAN ROC-AUC (DESCRIPTIVE):')
    print(overall[['auc_descriptive_rank','method','equal_algorithm_mean_roc_auc',
                   'equal_algorithm_mean_ap','mean_split_auc_rank']].head(5).to_string(index=False))
    print('\nLEADER VS SECOND PLACE:')
    row=leader_vs.set_index('competitor').loc[overall.iloc[1].method]
    print('{} minus {}: mean delta AUC={:+.4f}; algorithms positive/tied/negative={}/{}/{}'.format(
        leader,overall.iloc[1].method,row.mean_delta_auc_leader_minus_competitor,
        int(row.positive_auc_algorithms),int(row.tied_auc_algorithms),
        int(row.negative_auc_algorithms)))
    print('Next: Step 10 finalize cautious RQ1 results and reproducibility checks.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root',type=Path,
                        default=Path(__file__).resolve().parent.parent,
                        help='Repository folder containing output/RQ1/pi0_200/')
    args=parser.parse_args()
    try:
        run(args.repo_root.resolve())
        return 0
    except (OSError,ValueError,TypeError,KeyError,AssertionError,ImportError) as exc:
        print('BLOCKED:',str(exc),file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
