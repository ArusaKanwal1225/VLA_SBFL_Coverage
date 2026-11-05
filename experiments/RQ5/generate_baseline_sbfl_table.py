#!/usr/bin/env python3
"""Combine complete RQ5 baselines with the top-five SBFL methods by Overall."""
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import pandas as pd
from experiments.RQ1.generate_tables import tie_aware_ranks, latex_escape

MODELS = ('EO1','GR00T','Pi0','SpatialVLA')
SBFL_MODEL = {'EO1':'EO1','GR00T':'Groot','Pi0':'Pi0','SpatialVLA':'SpatialVLA'}


def generate(baseline_dir, sbfl_dir, output):
    baseline_config = json.loads((baseline_dir/'analysis_config.json').read_text())
    sbfl_config = json.loads((sbfl_dir/'analysis_config.json').read_text())
    if not baseline_config.get('sample_sha256') or baseline_config['sample_sha256'] != sbfl_config.get('source_sample_sha256'):
        raise ValueError('Baseline and SBFL results do not have the same source sample manifest')
    base = pd.read_csv(baseline_dir/'baseline_performance.csv')
    model = pd.read_csv(sbfl_dir/'model_metrics.csv')
    required = {'vla','formula_id','technique','roc_auc'}
    if not required <= set(model.columns) or model.duplicated(['vla','formula_id']).any():
        raise ValueError('Invalid or duplicate SBFL model metrics')
    lookup = {}
    for name in MODELS:
        subset = model[model.vla == SBFL_MODEL[name]].set_index('formula_id')
        if len(subset) != 34: raise ValueError('Expected 34 SBFL results for '+name)
        lookup[name] = subset
    ids = set(lookup[MODELS[0]].index)
    if any(set(x.index) != ids for x in lookup.values()): raise ValueError('SBFL catalogues differ')
    by_id = {}
    for fid in ids:
        labels = {str(lookup[m].loc[fid,'technique']) for m in MODELS}
        if len(labels)!=1: raise ValueError('Inconsistent SBFL technique name: '+fid)
        vals = {m:float(lookup[m].loc[fid,'roc_auc']) for m in MODELS}
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in vals.values()): raise ValueError('Invalid SBFL ROC-AUC')
        by_id[fid] = {'id':fid,'method':labels.pop(),'family':'SBFL','task_aware':False, **vals,
                      'Overall':sum(vals.values())/len(MODELS)}
    sbfl_ranks = tie_aware_ranks({fid:row['Overall'] for fid,row in by_id.items()})
    selected = [by_id[fid] for fid in ids if sbfl_ranks[fid] <= 5]
    if len(selected) < 5: raise ValueError('Too few SBFL methods in top five')
    rows = []
    for r in base.to_dict('records'):
        if r['method'].lower() == 'harmonic mean': raise ValueError('Baseline input unexpectedly includes Harmonic Mean')
        vals={m:float(r[m]) for m in MODELS}
        if any(not math.isfinite(v) or not 0<=v<=1 for v in vals.values()): raise ValueError('Invalid baseline ROC-AUC')
        aware = r['task_aware'] if isinstance(r['task_aware'],bool) else str(r['task_aware']).lower()=='true'
        rows.append({'id':str(r['metric_id']),'method':str(r['method']),'family':str(r['family']), 'task_aware':aware,
                     **vals, 'Overall':sum(vals.values())/len(MODELS)})
    rows += selected
    ids_for_rank = {r['id']:r['Overall'] for r in rows}
    if len(ids_for_rank)!=len(rows): raise ValueError('Duplicate method IDs')
    ranks=tie_aware_ranks(ids_for_rank)
    for r in rows:r['Rank']=ranks[r['id']]
    rows.sort(key=lambda r:(-r['Overall'],r['id']))
    output.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(output/'baseline_top5_sbfl.csv',index=False,float_format='%.12g')
    cols=[*MODELS,'Overall']; first={c:max(r[c] for r in rows) for c in cols}
    second={c:next((x for x in sorted({r[c] for r in rows},reverse=True) if not math.isclose(x,first[c],rel_tol=1e-12,abs_tol=1e-12)),None) for c in cols}
    bs=chr(92)
    caption=('Move-near ROC-AUC of trajectory-level baselines and the top-five SBFL methods on the same selected sample. '
             '$^{'+chr(92)+'ddagger}$ indicates an SBFL technique. Overall is the equal-weight mean across the four VLAs. '
             'Bold values are best and underlined values second-best in each column. '
             '$^{'+bs+'dagger}$ indicates a task-aware metric using object/goal state.')
    lines=[bs+'begin{table*}[t]',bs+'centering',bs+'caption{'+caption+'}',bs+'label{tab:rq5-baselines-top5-sbfl}',
           '{'+bs+'scriptsize',bs+'setlength{'+bs+'tabcolsep}{4pt}',bs+'begin{tabular}{lcccccc}',bs+'toprule',
           'Method & '+' & '.join(cols)+' & Rank '+bs*2,bs+'midrule']
    for r in rows:
        label=latex_escape(r['method'])
        if r['family']=='SBFL': label += '$^{'+bs+'ddagger}$'
        if r['task_aware']: label += '$^{'+bs+'dagger}$'
        cells=[]
        for c in cols:
            v=r[c]; text=f'{v:.4f}'
            if math.isclose(v,first[c],rel_tol=1e-12,abs_tol=1e-12): text=bs+'textbf{'+text+'}'
            elif second[c] is not None and math.isclose(v,second[c],rel_tol=1e-12,abs_tol=1e-12): text=bs+'underline{'+text+'}'
            cells.append(text)
        lines.append(label+' & '+' & '.join(cells)+' & '+str(r['Rank'])+' '+bs*2)
    lines += [bs+'bottomrule',bs+'end{tabular}','}',bs+'end{table*}']
    (output/'baseline_top5_sbfl.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Top SBFL:', ', '.join(r['method'] for r in sorted(selected,key=lambda r:(-r['Overall'],r['id']))))
    print('Combined table:',output)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-dir',type=Path,required=True)
    p.add_argument('--sbfl-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args(); generate(a.baseline_dir,a.sbfl_dir,a.output_dir)

if __name__=='__main__':main()
