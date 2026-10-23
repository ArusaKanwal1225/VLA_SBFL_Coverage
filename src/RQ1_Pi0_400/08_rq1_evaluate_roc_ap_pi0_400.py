#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RQ1 Step 08: evaluate *frozen* Pi0 400-scene test scores.

Run from any working directory (Python 3.9+):
    python src/RQ1_Pi0_400/08_rq1_evaluate_roc_ap_pi0_400.py --repo-root .

Required input (made in Step 07, no training or score recomputation):
    output/RQ1/pi0_400/step07_heldout_scoring/test_episode_scores.csv.gz
    output/RQ1/pi0_400/step07_heldout_scoring/test_scoring_split_summary.csv
    output/RQ1/pi0_400/step07_heldout_scoring/test_method_score_diagnostics.csv
    output/RQ1/pi0_400/step07_heldout_scoring/step07_validation_report.json

Generated under output/RQ1/pi0_400/step08_roc_ap/:
    per_split_metrics.csv - 560 method x algorithm x seed rows
    split_class_balance.csv - 40 held-out split class counts
    per_algorithm_method_summary.csv - 56 method x algorithm summaries
    overall_method_summary.csv - 14 equal-algorithm-weight summaries
    roc_curve_points.csv.gz and pr_curve_points.csv.gz - raw curve points
    step08_validation_report.json - hashes, validation and protocol notes
    figures/ - 13 plots, each written as PNG and PDF

IMPORTANT: ROC-AUC and AP quantify held-out FAILURE DISCRIMINATION from
spatial suspiciousness, not ground-truth fault-localization accuracy.
The same 10 repeated splits overlap; treat variation as descriptive, NOT 40
independent replications. This step DOES NOT fit scores or select hyperparams.
ASCII-only source to avoid Windows cp1252 source encoding issues.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc, average_precision_score, precision_recall_curve,
    roc_auc_score, roc_curve,
)

ALGORITHMS = (
    'DH_400scenes', 'DH_HybridDT_400scenes',
    'DH_HybridRF_400scenes', 'DH_WeightedRF_400scenes',
)
SEEDS = (11, 23, 37, 41, 53, 67, 79, 83, 97, 101)
METHODS = (
    'Tarantula', 'Barinel', 'Kulczynski2', 'Wong2', 'Ochiai2',
    'Ochiai', 'DStar2', 'Jaccard', 'Wong3', 'GP13', 'OP2',
    'Wong1', 'Zoltar', 'Ample',
)
BASE = Path('output') / 'RQ1' / 'pi0_400'
STEP7 = BASE / 'step07_heldout_scoring'
STEP8 = BASE / 'step08_roc_ap'


def check(condition, description):
    if not condition:
        raise ValueError(description)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def input_file(path):
    check(path.is_file(), 'Missing Step07 input file: ' + str(path))
    return path


def save_csv(frame, path):
    frame.to_csv(path, index=False, encoding='utf-8', float_format='%.16g')


def save_figure(fig, directory, name):
    directory.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(directory / (name + '.' + ext), dpi=170, bbox_inches='tight')
    plt.close(fig)


def read_inputs(root):
    src = root / STEP7
    files = {
        'predictions': input_file(src / 'test_episode_scores.csv.gz'),
        'split_summary': input_file(src / 'test_scoring_split_summary.csv'),
        'method_diagnostics': input_file(src / 'test_method_score_diagnostics.csv'),
        'step07_report': input_file(src / 'step07_validation_report.json'),
    }
    report = json.loads(files['step07_report'].read_text(encoding='utf-8-sig'))
    check(report.get('status') == 'PASS', 'Step07 report must be PASS')
    check(report.get('model') == 'Pi0' and report.get('budget') == 400,
          'Step07 report must be for Pi0 with 400 scenes')
    check(report.get('labels_used_to_construct_risk_scores') is False,
          'Step07 risk scores must be independent of held-out labels')
    check(report.get('splits_scored') == 40 and
          report.get('method_split_combinations') == 560,
          'Step07 did not report 40 splits and 560 method/split evaluations')
    check(report.get('algorithms') == list(ALGORITHMS) and
          report.get('seeds') == list(SEEDS) and
          report.get('methods') == list(METHODS),
          'Step07 model protocol differs from Step08')
    check(report.get('prediction_file_sha256') == sha256(files['predictions']),
          'FROZEN Step07 predictions changed since Step07 report was created')

    data = pd.read_csv(files['predictions'], encoding='utf-8-sig',
                       dtype={'algorithm':'string','run_id':'string','episode_id':'string'},
                       compression='gzip')
    summary = pd.read_csv(files['split_summary'], encoding='utf-8-sig')
    diag = pd.read_csv(files['method_diagnostics'], encoding='utf-8-sig')
    risk_cols = ['score_' + m for m in METHODS]
    mandatory = {'algorithm','seed','run_id','episode_id','failure',
                 'episode_unique_voxels','known_voxels','unseen_voxels',
                 'known_voxel_fraction','no_known_voxels'} | set(risk_cols)
    check(mandatory.issubset(data.columns),
          'Step07 predictions missing columns: ' + str(sorted(mandatory-set(data.columns))))
    check(len(data) == int(report.get('test_episode_predictions', -1)),
          'Step07 predictions row count does not match report')
    check(not data[['algorithm','seed','episode_id']].duplicated().any(),
          'Duplicate held-out episode within algorithm/seed')
    check(set(data.algorithm.astype(str)) == set(ALGORITHMS) and
          set(data.seed.unique()) == set(SEEDS),
          'Unexpected algorithm or seed')
    check(data.failure.isin([0,1]).all(),
          'Failure labels must be numeric 0 or 1')
    check(data[['run_id','episode_id']].notna().all().all(),
          'Blank episode identity')
    check(data.episode_id.astype(str).str.len().gt(0).all(),
          'Empty episode identity')
    check(data.episode_unique_voxels.gt(0).all() and
          data.known_voxels.ge(0).all() and data.unseen_voxels.ge(0).all(),
          'Invalid voxel count in Step07 predictions')
    check((data.known_voxels + data.unseen_voxels ==
           data.episode_unique_voxels).all(),
          'Known plus unseen does not equal total unique voxels')
    frac = data.known_voxels / data.episode_unique_voxels
    check(np.allclose(frac, data.known_voxel_fraction, atol=1e-11),
          'Known fraction inconsistent with voxel counts')
    check((data.no_known_voxels == data.known_voxels.eq(0).astype(int)).all(),
          'No-known flag inconsistent with counts')
    check(np.isfinite(data[risk_cols].to_numpy(dtype='float64')).all(),
          'Non-finite held-out suspiciousness risk scores')

    check(len(summary) == 40 and
          {'algorithm','seed','test_episodes','test_failures','test_successes'}
          .issubset(summary.columns),
          'Invalid Step07 test split summary schema')
    check(not summary[['algorithm','seed']].duplicated().any(),
          'Duplicate Step07 split summary rows')
    check(len(diag) == 560 and
          {'algorithm','seed','method','test_episodes','distinct_predicted_scores'}
          .issubset(diag.columns),
          'Invalid Step07 method diagnostic schema')
    check(not diag[['algorithm','seed','method']].duplicated().any(),
          'Duplicate Step07 method diagnostics')
    check(set(diag.method) == set(METHODS), 'Step07 diagnostic method mismatch')
    check(len(set(data['episode_id'])) <= 16000, 'Unexpected episode identifier count')

    expected = {(a,s) for a in ALGORITHMS for s in SEEDS}
    check(set(zip(data.algorithm.astype(str),data.seed)) == expected,
          'Missing algorithm/seed block in held-out predictions')
    check(set(zip(summary.algorithm, summary.seed)) == expected,
          'Missing algorithm/seed block in split summaries')
    check(set(zip(diag.algorithm, diag.seed, diag.method)) ==
          {(a,s,m) for a,s in expected for m in METHODS},
          'Step07 method diagnostic coverage incomplete')
    check(int(report.get('per_method_predictions',-1)) == len(data)*len(METHODS),
          'Step07 risk value count mismatch')
    return files, report, data, summary, diag


def compute_metrics(data, summary, diag):
    split_reference = {(r.algorithm,int(r.seed)): r
                       for r in summary.itertuples(index=False)}
    diag_reference = {(r.algorithm,int(r.seed),r.method): r
                      for r in diag.itertuples(index=False)}
    metric_rows, class_rows, roc_rows, pr_rows = [], [], [], []
    representative = {}
    count = 0
    for alg in ALGORITHMS:
        by_alg = data[data.algorithm == alg]
        check(1150 <= by_alg.episode_id.nunique() <= 4000,
              alg + ': unexpected distinct held-out episode count')
        for seed in SEEDS:
            block = by_alg[by_alg.seed == seed].copy()
            key = (alg, seed)
            check(1150 <= len(block) <= 1250,
                  f'{key}: unexpected held-out episode count')
            y = block.failure.to_numpy(dtype=int)
            npos, nneg = int(y.sum()), int(len(y)-y.sum())
            check(npos > 0 and nneg > 0,
                  f'{key}: ROC-AUC undefined, test must contain both classes')
            ref = split_reference[key]
            check(len(block) == int(ref.test_episodes) and
                  npos == int(ref.test_failures) and
                  nneg == int(ref.test_successes),
                  f'{key}: held-out labels differ from frozen Step07 summary')
            prevalence = npos/len(y)
            class_rows.append({
                'algorithm':alg, 'seed':seed, 'test_episodes':len(y),
                'test_failures':npos,'test_successes':nneg,
                'test_failure_prevalence':prevalence,
                'random_ranking_roc_auc_reference':0.5,
                'no_skill_ap_reference':prevalence,
            })
            for method in METHODS:
                scores = block['score_'+method].to_numpy(dtype=float)
                d = diag_reference[(alg,seed,method)]
                check(int(d.test_episodes) == len(block) and
                      int(d.distinct_predicted_scores) == len(np.unique(scores)),
                      f'{key}/{method}: prediction count/ties disagree with Step07 diagnostics')
                # ROC-AUC uses the continuous raw risk scores, not thresholded labels.
                roc_auc_value = float(roc_auc_score(y, scores))
                ap = float(average_precision_score(y, scores))
                fpr, tpr, roc_thresholds = roc_curve(y, scores,
                                                       drop_intermediate=False)
                check(math.isclose(roc_auc_value, float(auc(fpr,tpr)),
                                   rel_tol=0, abs_tol=1e-10),
                      f'{key}/{method}: roc_auc numerical discrepancy')
                precision, recall, pr_thresholds = precision_recall_curve(y,scores)
                # sklearn average precision is the step-wise area, NOT trapezoid PR AUC.
                reconstructed_ap = float(-np.sum(np.diff(recall)*precision[:-1]))
                check(math.isclose(ap, reconstructed_ap,
                                   rel_tol=0, abs_tol=1e-10),
                      f'{key}/{method}: AP numerical discrepancy')
                check(np.all(np.diff(fpr) >= -1e-13) and
                      np.all(np.diff(recall) <= 1e-13),
                      f'{key}/{method}: curve points not correctly ordered')
                unique = len(np.unique(scores))
                is_constant = unique == 1
                if is_constant:
                    check(math.isclose(roc_auc_value,0.5,abs_tol=1e-10) and
                          math.isclose(ap, prevalence,abs_tol=1e-10),
                          f'{key}/{method}: constant scores should give chance-level metrics')
                metric_rows.append({
                    'algorithm':alg,'seed':seed,'method':method,
                    'test_episodes':len(block),'test_failures':npos,
                    'test_successes':nneg,
                    'test_failure_prevalence':prevalence,
                    'roc_auc':roc_auc_value,'average_precision':ap,
                    'ap_minus_prevalence':ap-prevalence,
                    'number_distinct_risk_scores':unique,
                    'constant_risk_score':bool(is_constant),
                    'fraction_test_episodes_at_max_score':float(d.fraction_episodes_at_max_score),
                })
                # Persist points so every generated ROC/PR plot is reproducible.
                for i, (x, t, threshold) in enumerate(zip(fpr,tpr,roc_thresholds)):
                    roc_rows.append({'algorithm':alg,'seed':seed,'method':method,
                                     'point_index':i,'false_positive_rate':float(x),
                                     'true_positive_rate':float(t),
                                     'threshold':float(threshold)})
                for i, (p, r) in enumerate(zip(precision, recall)):
                    pr_rows.append({'algorithm':alg,'seed':seed,'method':method,
                                    'point_index':i,'precision':float(p),
                                    'recall':float(r),
                                    'threshold':float(pr_thresholds[i]) if i<len(pr_thresholds)
                                                else np.nan})
                if seed == 11:
                    representative[(alg,method)] = {
                        'fpr':fpr,'tpr':tpr,'precision':precision,'recall':recall,
                        'auc':roc_auc_value,'ap':ap, 'prevalence':prevalence,
                    }
                count += 1
            print('[METRICS] {} seed={}: test={} failures={} prevalence={:.3f}'
                  .format(alg,seed,len(block),npos,prevalence))
    check(count == 560 and len(class_rows)==40,
          'Expected 560 metric rows and 40 class balance rows')
    return (pd.DataFrame(metric_rows),pd.DataFrame(class_rows),
            pd.DataFrame(roc_rows),pd.DataFrame(pr_rows),representative)


def aggregate(metrics):
    # Overlapping outer splits are descriptive repeated splits, not independent runs.
    per_alg = (metrics.groupby(['algorithm','method'],as_index=False)
               .agg(n_splits=('seed','nunique'),
                    mean_roc_auc=('roc_auc','mean'),
                    sd_roc_auc=('roc_auc','std'),
                    min_roc_auc=('roc_auc','min'),
                    max_roc_auc=('roc_auc','max'),
                    median_roc_auc=('roc_auc','median'),
                    mean_average_precision=('average_precision','mean'),
                    sd_average_precision=('average_precision','std'),
                    mean_test_prevalence=('test_failure_prevalence','mean'),
                    mean_ap_lift=('ap_minus_prevalence','mean'),
                    constant_risk_splits=('constant_risk_score','sum')))
    check(len(per_alg)==56 and per_alg.n_splits.eq(10).all(),
          'Expected 56 method/algorithm summaries with 10 splits each')
    # Equal weight to all 4 algorithms: avoids heavier weight due to class balance.
    overall = (per_alg.groupby('method',as_index=False)
               .agg(n_algorithms=('algorithm','nunique'),
                    equal_algorithm_mean_roc_auc=('mean_roc_auc','mean'),
                    between_algorithm_sd_mean_roc_auc=('mean_roc_auc','std'),
                    equal_algorithm_mean_average_precision=('mean_average_precision','mean'),
                    equal_algorithm_mean_ap_lift=('mean_ap_lift','mean'),
                    total_constant_risk_splits=('constant_risk_splits','sum')))
    overall=overall.sort_values(['equal_algorithm_mean_roc_auc','equal_algorithm_mean_average_precision'],
                                 ascending=[False,False]).reset_index(drop=True)
    overall.insert(0,'descriptive_auc_rank',np.arange(1,len(overall)+1))
    check(len(overall)==14 and overall.n_algorithms.eq(4).all(),
          'Expected 14 methods each evaluated on four algorithms')
    return per_alg,overall


def heatmap(per_alg, metric, title, name, figdir):
    pivot=per_alg.pivot(index='method',columns='algorithm',values=metric)
    pivot=pivot.reindex(index=list(METHODS),columns=list(ALGORITHMS))
    check(not pivot.isna().any().any(),'Incomplete heatmap: '+metric)
    fig,ax=plt.subplots(figsize=(9.2,6.7))
    mat=pivot.to_numpy(dtype=float)
    im=ax.imshow(mat,aspect='auto',vmin=0,vmax=1)
    ax.set_xticks(np.arange(len(ALGORITHMS)),
                  [a.replace('_400scenes','') for a in ALGORITHMS],rotation=20,ha='right')
    ax.set_yticks(np.arange(len(METHODS)),METHODS)
    ax.set_title(title)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j,i,f'{mat[i,j]:.3f}',ha='center',va='center',fontsize=8,
                    color='white' if mat[i,j] < 0.27 else 'black')
    fig.colorbar(im,ax=ax,shrink=.85,label=metric.replace('_',' '))
    save_figure(fig,figdir,name)


def plot_rank(overall,metric,label,name,figdir,baseline=None):
    table=overall.sort_values(metric,ascending=True)
    fig,ax=plt.subplots(figsize=(9.4,6.6))
    ax.barh(table.method,table[metric],alpha=.9)
    if baseline is not None:
        ax.axvline(baseline,color='gray',ls='--',lw=1.3,label=f'Reference={baseline:g}')
        ax.legend(loc='lower right',fontsize=9)
    ax.set_xlim(0,1)
    ax.set_xlabel(label)
    ax.set_title('Pi0 400: descriptive overall method ranking (4 algorithms equally weighted)')
    for i,value in enumerate(table[metric].to_numpy(dtype=float)):
        ax.text(min(value+.008,.985),i,f'{value:.3f}',fontsize=8,va='center')
    save_figure(fig,figdir,name)


def plot_seed11_curves(representative,alg,figdir,kind):
    fig,ax=plt.subplots(figsize=(9.3,6.0))
    for method in METHODS:
        cur=representative[(alg,method)]
        if kind == 'roc':
            x,y=cur['fpr'],cur['tpr']
            caption=f'{method} (AUC={cur["auc"]:.3f})'
        else:
            x,y=cur['recall'],cur['precision']
            caption=f'{method} (AP={cur["ap"]:.3f})'
        ax.plot(x,y,lw=1.1,alpha=.85,label=caption)
    if kind=='roc':
        ax.plot([0,1],[0,1],'k--',lw=1.2,alpha=.55,label='Random ranking (AUC=.5)')
        ax.set_xlabel('False positive rate');ax.set_ylabel('True positive rate')
        ax.set_title('Pi0 400 | '+alg+' | seed 11 | ROC')
        filename='05_roc_seed11_'+alg
    else:
        prevalence=representative[(alg,METHODS[0])]['prevalence']
        ax.axhline(prevalence,color='gray',ls='--',lw=1.3,
                   label=f'No-skill AP baseline = prevalence {prevalence:.3f}')
        ax.set_xlabel('Recall');ax.set_ylabel('Precision')
        ax.set_title('Pi0 400 | '+alg+' | seed 11 | Precision-Recall')
        filename='06_pr_seed11_'+alg
    ax.set_xlim(0,1);ax.set_ylim(0,1.02)
    ax.grid(alpha=.16)
    ax.legend(bbox_to_anchor=(1.01,1),loc='upper left',fontsize=7.5,ncol=1)
    save_figure(fig,figdir,filename)


def create_figures(metrics,per_alg,overall,classes,representative,figdir):
    heatmap(per_alg,'mean_roc_auc','Mean held-out ROC-AUC across 10 repeated splits',
            '01_mean_roc_auc_by_algorithm_method',figdir)
    heatmap(per_alg,'mean_average_precision','Mean held-out AP across 10 repeated splits',
            '02_mean_average_precision_by_algorithm_method',figdir)
    plot_rank(overall,'equal_algorithm_mean_roc_auc','Equal-algorithm mean ROC-AUC',
              '03_overall_auc_ranking',figdir,baseline=.5)
    plot_rank(overall,'equal_algorithm_mean_average_precision',
              'Equal-algorithm mean average precision',
              '04_overall_ap_ranking',figdir)
    for alg in ALGORITHMS:
        plot_seed11_curves(representative,alg,figdir,'roc')
        plot_seed11_curves(representative,alg,figdir,'pr')
    base=classes.groupby('algorithm',as_index=False).agg(
        mean_test_prevalence=('test_failure_prevalence','mean'),
        sd_test_prevalence=('test_failure_prevalence','std'))
    fig,ax=plt.subplots(figsize=(9,4.8))
    x=np.arange(len(base))
    ax.bar(x,base.mean_test_prevalence,yerr=base.sd_test_prevalence,capsize=5)
    ax.set_xticks(x,[a.replace('_400scenes','') for a in base.algorithm],rotation=15)
    ax.set_ylabel('Failure prevalence in held-out test splits')
    ax.set_ylim(0,1)
    ax.set_title('Pi0 400: test prevalence (no-skill AP reference)')
    save_figure(fig,figdir,'07_test_failure_prevalence_by_algorithm')
    check(len(list(figdir.glob('*.png'))) >= 13,'Expected at least 13 PNG figures')


def run(root):
    files,report,predictions,summaries,diagnostics=read_inputs(root)
    metrics,classes,roc,pr,rep=compute_metrics(predictions,summaries,diagnostics)
    per_alg,overall=aggregate(metrics)
    out=root/STEP8
    out.mkdir(parents=True,exist_ok=True)
    save_csv(metrics,out/'per_split_metrics.csv')
    save_csv(classes,out/'split_class_balance.csv')
    save_csv(per_alg,out/'per_algorithm_method_summary.csv')
    save_csv(overall,out/'overall_method_summary.csv')
    roc.to_csv(out/'roc_curve_points.csv.gz',index=False,
               compression='gzip',float_format='%.16g')
    pr.to_csv(out/'pr_curve_points.csv.gz',index=False,
              compression='gzip',float_format='%.16g')
    create_figures(metrics,per_alg,overall,classes,rep,out/'figures')
    report08={
        'status':'PASS','step':'08_roc_auc_average_precision',
        'model':'Pi0','budget':400,
        'algorithms':list(ALGORITHMS),'seeds':list(SEEDS),'methods':list(METHODS),
        'splits_evaluated':len(classes),'method_split_metrics':len(metrics),
        'heldout_episode_rows_evaluated':len(predictions),
        'per_method_prediction_scores':len(predictions)*len(METHODS),
        'constant_score_method_split_count':int(metrics.constant_risk_score.sum()),
        'roc_auc_reference':0.5,
        'average_precision_reference':'held-out failure prevalence per algorithm/seed',
        'average_precision_definition':'sklearn.metrics.average_precision_score, step-wise AP; not trapezoidal PR AUC',
        'direction':'higher SBFL risk predicts failure=1',
        'risk_rule_frozen_in_step07':report['risk_aggregation'],
        'computed_new_suspiciousness_maps':False,
        'computed_new_train_test_splits':False,
        'reused_identical_heldout_episodes_for_all_14_methods':True,
        'aggregation':'mean of per-algorithm ten-split means; equal weight to four algorithms',
        'statistical_caution':'Overlapping 70/30 splits are not independent replications. Step08 rankings are descriptive.',
        'scope_caution':'ROC-AUC/AP assess held-out failure discrimination, not known fault-location accuracy.',
        'scene_identity_caution':report.get('candidate_scene_grouping_warning',''),
        'input_files_sha256':{k:sha256(path) for k,path in files.items()},
        'output_files_sha256':{
            filename:sha256(out/filename) for filename in [
                'per_split_metrics.csv','split_class_balance.csv',
                'per_algorithm_method_summary.csv','overall_method_summary.csv',
                'roc_curve_points.csv.gz','pr_curve_points.csv.gz']},
        'figures_png_count':len(list((out/'figures').glob('*.png'))),
        'figures_pdf_count':len(list((out/'figures').glob('*.pdf'))),
    }
    (out/'step08_validation_report.json').write_text(json.dumps(report08,indent=2),encoding='utf-8')
    print('\nRQ1 STEP 08 - ROC-AUC AND AVERAGE PRECISION')
    print('Splits: {}/40 | Methods: 14 | Metric pairs: {}/560'.format(len(classes),len(metrics)))
    print('Held-out episode predictions: {} | Risk values: {}'.format(
        len(predictions),len(predictions)*len(METHODS)))
    print('Constant-risk method/split cases: {}'.format(report08['constant_score_method_split_count']))
    print('ROC and PR curve data: SAVED | Figures (PNG/PDF): {}/{}'.format(
        report08['figures_png_count'],report08['figures_pdf_count']))
    print('Test outcomes used ONLY for evaluation metrics: YES')
    print('New model training or score fitting: NO')
    print('Status: PASS')
    print('Output:',out)
    print('\nDESCRIPTIVE ROC-AUC LEADERS (not a significance conclusion):')
    print(overall[['descriptive_auc_rank','method','equal_algorithm_mean_roc_auc',
                   'equal_algorithm_mean_average_precision']].head(5).to_string(index=False))
    print('Next: Step09 paired method comparison using the SAME held-out episodes.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root',type=Path,
                        default=Path(__file__).resolve().parents[2],
                        help='Repository folder containing input/, src/, output/')
    args=parser.parse_args()
    try:
        run(args.repo_root.resolve())
        return 0
    except (OSError,ValueError,TypeError,KeyError,AssertionError) as exc:
        print('BLOCKED:',str(exc),file=sys.stderr)
        return 1


if __name__=='__main__':
    raise SystemExit(main())
