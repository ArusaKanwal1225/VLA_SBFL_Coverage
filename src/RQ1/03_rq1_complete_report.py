"""RQ1 script 3/3: final paired descriptive comparison and research report.

Run: python src/03_rq1_complete_report.py
Reads only frozen metrics from 02_rq1_test_and_evaluate.py.
Writes final CSV comparisons, PNG/PDF figures, Markdown summary, and hashes.
Repeated 70/30 splits overlap: no naive significance p-values are reported.
"""
import argparse
import itertools
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rq1_common import (ALGORITHMS,SEEDS,METHODS,REL_OUTPUT,require,
                        out_csv,out_json,fig_save,sha256,get_root)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo-root',default=None)
    args=ap.parse_args()
    root=get_root(__file__,args.repo_root)
    out=root/REL_OUTPUT
    reportfile=out/'02_testing_report.json'
    require(reportfile.is_file(),'Run scripts 01 and 02 before final report')
    info=json.loads(reportfile.read_text(encoding='utf-8-sig'))
    require(info['status']=='PASS' and info['method_split_metrics']==560,
            'Testing stage incomplete')
    path=out/'per_split_metrics.csv'
    require(path.is_file() and sha256(path)==info['metrics_sha256'],
            'The frozen evaluation metrics have been modified')
    df=pd.read_csv(path,encoding='utf-8-sig')
    require(len(df)==560 and set(df.method)==set(METHODS),
            'Expected 560 metric rows covering all 14 methods')
    expected={(a,s,m) for a in ALGORITHMS for s in SEEDS for m in METHODS}
    require(set(zip(df.algorithm,df.seed,df.method))==expected,
            'Missing or duplicate algorithm/seed/method results')
    require(df[['roc_auc','average_precision']].notna().all().all() and
            df.roc_auc.between(0,1).all() and df.average_precision.between(0,1).all(),
            'Invalid ROC-AUC/AP values')
    alg_method=(df.groupby(['algorithm','method'],as_index=False)
                .agg(mean_roc_auc=('roc_auc','mean'),sd_roc_auc=('roc_auc','std'),
                     mean_ap=('average_precision','mean'),sd_ap=('average_precision','std')))
    out_csv(alg_method,out/'final_by_algorithm_method.csv')
    leaderboard=(alg_method.groupby('method',as_index=False)
                 .agg(equal_algorithm_mean_roc_auc=('mean_roc_auc','mean'),
                      equal_algorithm_mean_ap=('mean_ap','mean'))
                 .sort_values(['equal_algorithm_mean_roc_auc','method'],ascending=[False,True])
                 .reset_index(drop=True))
    leaderboard.insert(0,'descriptive_auc_rank',np.arange(1,len(leaderboard)+1))
    out_csv(leaderboard,out/'final_method_leaderboard.csv')
    # Paired within the SAME algorithm and split; summarize by algorithm first.
    pivot_auc=df.pivot(index=['algorithm','seed'],columns='method',values='roc_auc')
    pivot_ap=df.pivot(index=['algorithm','seed'],columns='method',values='average_precision')
    paired=[]
    for one,two in itertools.combinations(sorted(METHODS),2):
        for (alg,seed), auc_row in pivot_auc.iterrows():
            ap_row=pivot_ap.loc[(alg,seed)]
            paired.append({'method_a':one,'method_b':two,'algorithm':alg,
                           'seed':int(seed),
                           'delta_auc_a_minus_b':float(auc_row[one]-auc_row[two]),
                           'delta_ap_a_minus_b':float(ap_row[one]-ap_row[two])})
    pair_df=pd.DataFrame(paired)
    require(len(pair_df)==3640,'Expected 91 method pairs x 4 algorithms x 10 seeds')
    out_csv(pair_df,out/'paired_by_split.csv.gz',compression='gzip')
    pair_alg=pair_df.groupby(['method_a','method_b','algorithm'],as_index=False).agg(
        mean_paired_delta_auc=('delta_auc_a_minus_b','mean'),
        mean_paired_delta_ap=('delta_ap_a_minus_b','mean'))
    out_csv(pair_alg,out/'paired_by_algorithm.csv')
    pair_overall=pair_alg.groupby(['method_a','method_b'],as_index=False).agg(
        equal_algorithm_mean_delta_auc=('mean_paired_delta_auc','mean'),
        equal_algorithm_mean_delta_ap=('mean_paired_delta_ap','mean'))
    out_csv(pair_overall,out/'paired_overall.csv')
    # Tie-aware winner/rank information for each split.
    ranks=df.copy()
    ranks['auc_rank']=ranks.groupby(['algorithm','seed']).roc_auc.rank(method='min',ascending=False)
    ranks['ap_rank']=ranks.groupby(['algorithm','seed']).average_precision.rank(method='min',ascending=False)
    out_csv(ranks,out/'split_method_ranks.csv')
    winners=(ranks.groupby('method',as_index=False)
             .agg(n_best_auc_splits=('auc_rank',lambda x:int((x==1).sum())),
                  mean_auc_rank=('auc_rank','mean'),
                  n_best_ap_splits=('ap_rank',lambda x:int((x==1).sum()))))
    out_csv(winners,out/'method_winner_counts.csv')
    leader=leaderboard.method.iloc[0]
    runner=leaderboard.method.iloc[1]
    leading=[]
    for opponent in leaderboard.method.iloc[1:]:
        for alg in ALGORITHMS:
            block=pivot_auc.loc[alg]
            delta=block[leader]-block[opponent]
            leading.append({'leader':leader,'opponent':opponent,'algorithm':alg,
                            'mean_delta_auc':float(delta.mean()),
                            'positive_seeds':int((delta>1e-10).sum()),
                            'ties':int((np.abs(delta)<=1e-10).sum()),
                            'negative_seeds':int((delta< -1e-10).sum())})
    lead=pd.DataFrame(leading)
    out_csv(lead,out/'leader_vs_other_methods.csv')
    # Optional verification against the already completed 10-step RQ1 study.
    legacy_path=(root/'output'/'RQ1'/'pi0_200'/'step08_roc_ap'/'per_split_metrics.csv')
    legacy_status='NOT_AVAILABLE'
    if legacy_path.is_file():
        old=pd.read_csv(legacy_path,encoding='utf-8-sig')
        required={'algorithm','seed','method','roc_auc','average_precision'}
        require(required.issubset(old.columns),
                'Existing Step08 metrics missing required columns')
        comparison=(df[list(required)].merge(old[list(required)],
                    on=['algorithm','seed','method'],how='outer',indicator=True,
                    suffixes=('_consolidated','_old'),validate='one_to_one'))
        require(len(comparison)==560 and comparison._merge.eq('both').all(),
                'Old and consolidated metric keys do not match')
        comparison['auc_abs_difference']=(comparison.roc_auc_consolidated-
                                          comparison.roc_auc_old).abs()
        comparison['ap_abs_difference']=(comparison.average_precision_consolidated-
                                         comparison.average_precision_old).abs()
        out_csv(comparison,out/'comparison_with_existing_step08.csv')
        max_difference=float(comparison[['auc_abs_difference','ap_abs_difference']].to_numpy().max())
        legacy_status='MATCH' if max_difference<1e-8 else 'DIFFER - INVESTIGATE'
        print(f'[LEGACY CROSS-CHECK] {legacy_status}; max AUC/AP difference={max_difference:.10g}')
    selected=lead[lead.opponent==runner]
    figdir=out/'figures'
    # 1. Final top-14 AUC ranking.
    fig,ax=plt.subplots(figsize=(10,6))
    ordered=leaderboard.iloc[::-1]
    ax.barh(ordered.method,ordered.equal_algorithm_mean_roc_auc)
    ax.set_xlim(0,1);ax.axvline(.5,linestyle='--',color='gray')
    ax.set_xlabel('Equal-algorithm mean held-out ROC-AUC')
    ax.set_title('Pi0 200: final descriptive SBFL ranking')
    fig_save(fig,figdir,'01_auc_leaderboard')
    # 2. Per-algorithm AUC heatmap.
    matrix=alg_method.pivot(index='method',columns='algorithm',values='mean_roc_auc')
    matrix=matrix.reindex(index=list(leaderboard.method),columns=list(ALGORITHMS))
    fig,ax=plt.subplots(figsize=(8,6))
    im=ax.imshow(matrix.to_numpy(),aspect='auto',vmin=0,vmax=1)
    ax.set_yticks(np.arange(len(matrix)),matrix.index)
    ax.set_xticks(range(4),[x.replace('_200scenes','') for x in ALGORITHMS],rotation=15)
    fig.colorbar(im,ax=ax,label='Mean ROC-AUC')
    ax.set_title('Pi0 200: algorithms by SBFL method')
    fig_save(fig,figdir,'02_auc_heatmap')
    # 3. Mean AUC differences, not p-values.
    fig,ax=plt.subplots(figsize=(9,4.5))
    ax.bar(selected.algorithm.str.replace('_200scenes','',regex=False),selected.mean_delta_auc)
    ax.axhline(0,linestyle='--',color='gray')
    ax.set_ylabel(f'Paired ROC-AUC: {leader} - {runner}')
    ax.set_title('Leader vs runner-up across algorithms')
    fig_save(fig,figdir,'03_leader_vs_runner')
    # 4. AP ranking.
    ordered_ap=leaderboard.sort_values('equal_algorithm_mean_ap').iloc[::-1]
    fig,ax=plt.subplots(figsize=(10,5))
    ax.barh(ordered_ap.method,ordered_ap.equal_algorithm_mean_ap)
    ax.set_xlim(0,1);ax.set_xlabel('Equal-algorithm mean Average Precision')
    ax.set_title('Pi0 200: Average Precision by method')
    fig_save(fig,figdir,'04_ap_ranking')
    # 5. Rank stability (descriptive).
    fig,ax=plt.subplots(figsize=(10,5))
    values=[ranks[ranks.method==m].auc_rank.to_numpy() for m in leaderboard.method]
    try:
        ax.boxplot(values, tick_labels=list(leaderboard.method),showfliers=False)
    except TypeError:
        ax.boxplot(values, labels=list(leaderboard.method),showfliers=False)
    ax.tick_params(axis='x',labelrotation=55)
    ax.set_ylabel('Per-split ROC-AUC rank (lower is better)')
    ax.set_title('Pi0 200: method-rank stability (overlapping splits)')
    fig_save(fig,figdir,'05_rank_stability')
    overall_delta=float(leaderboard.equal_algorithm_mean_roc_auc.iloc[0]
                        -leaderboard.equal_algorithm_mean_roc_auc.iloc[1])
    positive=int((selected.mean_delta_auc>1e-10).sum())
    report=f'''# RQ1 final descriptive report: Pi0, 200 scenes

## Study design
Four DeepHyperion-based algorithms, 10 fixed 70/30 scene-grouped splits per
algorithm, and 14 SBFL suspiciousness formulas applied to spatial TCP voxels.
Voxel size: 0.02 m; origin: (-1,-1,0). Episode risk is the maximum known-
training-voxel suspiciousness. ROC-AUC and Average Precision use held-out labels.

## Main result
**Descriptive first place:** {leader}, mean ROC-AUC {leaderboard.equal_algorithm_mean_roc_auc.iloc[0]:.6f},
mean AP {leaderboard.equal_algorithm_mean_ap.iloc[0]:.6f}.
**Second:** {runner}, mean ROC-AUC {leaderboard.equal_algorithm_mean_roc_auc.iloc[1]:.6f},
mean AP {leaderboard.equal_algorithm_mean_ap.iloc[1]:.6f}.
Mean paired ROC-AUC advantage: **{overall_delta:+.6f}**;
positive algorithm-level mean differences: **{positive}/4**.

## Important limitations
- The 10 split seeds reuse episodes, so these 40 split observations are not
  40 independent experiments. Differences and ranks are descriptive only;
  this report does not claim statistically significant superiority.
- The scene-group identity is based on inferred configuration fields and
  may omit simulator state not present in the extracted input.
- ROC-AUC/AP are metrics of held-out *failure discrimination*, not proof of
  the physical voxel that actually caused a failure.
- Conclusions here apply to Pi0 200-scene data, not all VLA models.

## Files
See final_method_leaderboard.csv, final_by_algorithm_method.csv,
paired_by_split.csv.gz, paired_by_algorithm.csv, paired_overall.csv,
leader_vs_other_methods.csv, and the saved figure PNG/PDF pairs.
'''
    (out/'final_rq1_report.md').write_text(report,encoding='utf-8')
    out_json({'status':'PASS','models':['Pi0'],'budget':200,
              'algorithm_count':4,'split_count':40,'methods':14,
              'method_split_metrics':560,'paired_method_pairs':91,
              'paired_split_rows':3640,'descriptive_leader':leader,
              'descriptive_runner_up':runner,'leader_mean_roc_auc':float(leaderboard.equal_algorithm_mean_roc_auc.iloc[0]),
              'runner_mean_roc_auc':float(leaderboard.equal_algorithm_mean_roc_auc.iloc[1]),
              'mean_auc_delta':overall_delta,'algorithms_positive':positive,
              'formal_significance_test':'NOT PERFORMED (overlapping splits)',
              'existing_step08_comparison':legacy_status,
              'metrics_sha256':sha256(path)},out/'03_final_report.json')
    print('\nRQ1 FINAL REPORT COMPLETE: PASS')
    print('Top 5:\n',leaderboard.head(5).to_string(index=False))
    print(f'Leader vs runner: {leader} - {runner} = {overall_delta:+.4f} mean AUC')
    print('Saved outputs and figures to:',out)

if __name__=='__main__':
    main()
