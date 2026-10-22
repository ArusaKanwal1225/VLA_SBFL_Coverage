"""RQ1 script 2/3: freeze training scores, score held-out trajectories,
compute ROC-AUC/AP, and save ROC/PR figures + CSV curve coordinates.

Run: python src/02_rq1_test_and_evaluate.py
Requires successful 01_rq1_train_suspiciousness.py and rq1_common.py.
No test outcomes are used in constructing risk scores.
"""
import argparse
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

from rq1_common import (ALGORITHMS, SEEDS, METHODS, REL_OUTPUT,
                        require, out_csv, out_json, fig_save, sha256, get_root)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo-root', default=None)
    args = ap.parse_args()
    root = get_root(__file__, args.repo_root)
    out = root / REL_OUTPUT
    report_path = out / '01_training_report.json'
    require(report_path.is_file(), 'Run script 01 first (training report missing)')
    training = json.loads(report_path.read_text(encoding='utf-8-sig'))
    require(training['status'] == 'PASS' and training['method_maps'] == 560,
            'Training stage incomplete')
    scorefile = out / 'training_voxel_scores.csv.gz'
    splitfile = out / 'split_manifest.csv.gz'
    visitfile = out / 'episode_voxel_visits.csv.gz'
    require(all(p.is_file() for p in (scorefile,splitfile,visitfile)),
            'One or more training artifacts are missing')
    require(sha256(scorefile) == training['scores_sha256'] and
            sha256(splitfile) == training['split_sha256'],
            'Frozen training scores/splits changed since training')
    visits_df = pd.read_csv(visitfile,encoding='utf-8-sig',compression='gzip',
                            dtype={'algorithm':str,'episode_id':str,'voxel':str})
    require({'algorithm','episode_id','voxel'}.issubset(visits_df.columns) and
            not visits_df.duplicated().any(), 'Invalid binary episode-voxel coverage')
    visits = {}
    for (alg,ep), block in visits_df.groupby(['algorithm','episode_id'],sort=False):
        visits[(alg,ep)] = frozenset(block.voxel)
    del visits_df
    split = pd.read_csv(splitfile,encoding='utf-8-sig',compression='gzip',
                        dtype={'algorithm':str,'episode_id':str,'run_id':str})
    split['seed'] = split.seed.astype(int)
    split['failure'] = split.failure.astype(int)
    score_df = pd.read_csv(scorefile,encoding='utf-8-sig',compression='gzip',
                           dtype={'algorithm':str,'voxel':str})
    require(set(METHODS).issubset(score_df.columns) and len(score_df) > 0,
            'Missing one or more SBFL score columns')
    require(score_df[['algorithm','seed','voxel']].duplicated().sum() == 0,
            'Duplicate voxel suspiciousness in training maps')
    test_rows, metrics, roc_rows, pr_rows, split_rows = [], [], [], [], []
    for alg in ALGORITHMS:
        a = split[split.algorithm == alg]
        scores_a = score_df[score_df.algorithm == alg]
        for seed in SEEDS:
            s = a[(a.seed == seed) & (a.split == 'test')].sort_values('episode_id')
            sub = scores_a[scores_a.seed == seed]
            require(len(sub) > 0 and len(s) > 0, f'Missing test scores for {alg} {seed}')
            block_values = sub[list(METHODS)].to_numpy(dtype='float64')
            minimum = block_values.min(axis=0)
            fallback = minimum - 1e-6*np.maximum(1.0,np.abs(minimum))
            vm = dict(zip(sub.voxel,block_values))
            predicted = []
            for item in s.itertuples(index=False):
                key = (alg,item.episode_id)
                require(key in visits, 'Missing test episode visits: '+str(key))
                vox = visits[key]
                known = [vm[v] for v in vox if v in vm]
                risk = np.max(np.vstack(known), axis=0) if known else fallback.copy()
                require(np.isfinite(risk).all(), 'Nonfinite test risks')
                n_known = len(known)
                record = dict(algorithm=alg,seed=seed,run_id=item.run_id,
                              episode_id=item.episode_id,failure=int(item.failure),
                              episode_unique_voxels=len(vox),known_voxels=n_known,
                              known_voxel_fraction=n_known/len(vox),
                              no_known_voxels=int(n_known == 0))
                for m,value in zip(METHODS,risk):
                    record['score_'+m] = float(value)
                predicted.append(record)
            frame = pd.DataFrame(predicted)
            y = frame.failure.to_numpy(dtype=int)
            require(set(y) == {0,1},f'Single-class test partition: {alg} seed {seed}')
            test_rows.extend(predicted)
            prevalence = float(y.mean())
            split_rows.append({'algorithm':alg,'seed':seed,'test_episodes':len(s),
                               'failures':int(y.sum()),'failure_prevalence':prevalence,
                               'mean_known_voxel_fraction':float(frame.known_voxel_fraction.mean()),
                               'no_known_voxel_episodes':int(frame.no_known_voxels.sum())})
            for method in METHODS:
                score = frame['score_'+method].to_numpy(dtype=float)
                auc = float(roc_auc_score(y,score))
                average_p = float(average_precision_score(y,score))
                metrics.append({'algorithm':alg,'seed':seed,'method':method,
                                'roc_auc':auc,'average_precision':average_p,
                                'test_episodes':len(y),'test_failures':int(y.sum()),
                                'test_prevalence':prevalence,
                                'distinct_risk_scores':len(np.unique(score))})
                fpr,tpr,_ = roc_curve(y,score)
                precision,recall,_ = precision_recall_curve(y,score)
                if seed == 11:
                    roc_rows.extend({'algorithm':alg,'seed':seed,'method':method,
                                     'fpr':float(x),'tpr':float(z)}
                                    for x,z in zip(fpr,tpr))
                    pr_rows.extend({'algorithm':alg,'seed':seed,'method':method,
                                    'recall':float(x),'precision':float(z)}
                                   for x,z in zip(recall,precision))
            print(f'[EVALUATED] {alg} seed={seed} held_out={len(s)}',flush=True)
    results = pd.DataFrame(metrics)
    pred = pd.DataFrame(test_rows)
    split_summary = pd.DataFrame(split_rows)
    require(len(results) == 560 and len(split_summary) == 40,
            'Expected 560 method evaluations and 40 test splits')
    require(not pred[['algorithm','seed','episode_id']].duplicated().any(),
            'Duplicate test episode in a split')
    require(len(pred)==int(split_summary.test_episodes.sum()) and
            len(pred)*len(METHODS)==len(pred)*14,
            'Internal test-score dimension mismatch')
    out_csv(pred, out/'test_episode_predictions.csv.gz',compression='gzip')
    out_csv(results,out/'per_split_metrics.csv')
    out_csv(split_summary,out/'test_split_summary.csv')
    out_csv(pd.DataFrame(roc_rows),out/'roc_curve_points_seed11.csv.gz',compression='gzip')
    out_csv(pd.DataFrame(pr_rows),out/'pr_curve_points_seed11.csv.gz',compression='gzip')
    figs = out / 'figures'
    # Diagnostic figure: same held-out episodes for every method.
    fig, ax = plt.subplots(figsize=(10,5))
    for alg in ALGORITHMS:
        s=split_summary[split_summary.algorithm == alg]
        ax.plot(s.seed,s.mean_known_voxel_fraction,marker='o',label=alg.replace('_200scenes',''))
    ax.set_xlabel('Split seed');ax.set_ylabel('Mean test fraction of known voxels')
    ax.set_ylim(0,1);ax.set_title('Pi0 200: held-out voxel familiarity')
    ax.legend(fontsize=8)
    fig_save(fig,figs,'04_test_known_voxel_fraction')
    for alg in ALGORITHMS:
        a = results[(results.algorithm==alg)&(results.seed==11)]
        f= pd.DataFrame(roc_rows)
        p= pd.DataFrame(pr_rows)
        fig, ax = plt.subplots(figsize=(6.8,5.1))
        for method in METHODS:
            pts = f[(f.algorithm==alg)&(f.method==method)]
            auc = float(a.loc[a.method==method,'roc_auc'].iloc[0])
            ax.plot(pts.fpr,pts.tpr,label=f'{method} ({auc:.3f})',linewidth=1.1)
        ax.plot([0,1],[0,1],linestyle='--',color='gray',label='Random')
        ax.set_xlabel('False positive rate');ax.set_ylabel('True positive rate')
        ax.set_title(f'{alg}: ROC, seed 11');ax.legend(fontsize=6,ncol=2)
        fig_save(fig,figs,'roc_seed11_'+alg)
        fig, ax = plt.subplots(figsize=(6.8,5.1))
        for method in METHODS:
            pts = p[(p.algorithm==alg)&(p.method==method)]
            apv = float(a.loc[a.method==method,'average_precision'].iloc[0])
            ax.plot(pts.recall,pts.precision,label=f'{method} ({apv:.3f})',linewidth=1.1)
        ax.axhline(float(split_summary[(split_summary.algorithm==alg)&(split_summary.seed==11)]
                         .failure_prevalence.iloc[0]),linestyle='--',color='gray')
        ax.set_xlabel('Recall');ax.set_ylabel('Precision');ax.set_title(f'{alg}: PR, seed 11')
        ax.legend(fontsize=6,ncol=2)
        fig_save(fig,figs,'pr_seed11_'+alg)
    out_json({'status':'PASS','splits':40,'method_split_metrics':560,
              'test_episode_predictions':len(pred),'risk_values':len(pred)*14,
              'methods':list(METHODS),'training_scores_hash':sha256(scorefile),
              'training_split_hash':sha256(splitfile),
              'test_outcomes_used_in_score_construction':False,
              'test_labels_used_only_for_evaluation':True,
              'metrics_sha256':sha256(out/'per_split_metrics.csv')},
             out/'02_testing_report.json')
    print('\nRQ1 HELD-OUT TESTING COMPLETE: 560/560 method metrics, status PASS')
    print('Predictions:',len(pred),'Score values:',len(pred)*14)
    print('Saved to:',out)

if __name__ == '__main__':
    main()
