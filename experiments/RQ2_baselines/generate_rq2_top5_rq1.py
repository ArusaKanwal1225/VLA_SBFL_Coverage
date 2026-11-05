#!/usr/bin/env python3
"""Combine the complete RQ2 baselines with the top-five RQ1 SBFL techniques."""
import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import pandas as pd
from experiments.RQ1.generate_tables import tie_aware_ranks, latex_escape

VLA_ORDER = ('EO1', 'Groot', 'Pi0', 'SpatialVLA')
DISPLAY = {'EO1':'EO1','Groot':'GR00T','Pi0':'Pi0','SpatialVLA':'SpatialVLA'}
DEFAULT_RQ1 = ROOT/'experiments/RQ1/results/phase_a_primary/aggregate'
DEFAULT_RQ2 = ROOT/'experiments/RQ2_baselines/results/baseline_comparison/aggregate'
DEFAULT_OUTPUT = ROOT/'experiments/RQ2_baselines/results/baseline_comparison/tables'


def truth(value): return str(value).strip().lower() in ('true','1','yes')


def generate(rq1, rq2, output):
    overall = pd.read_csv(rq1/'overall_ranking.csv')
    groups = pd.read_csv(rq1/'subgroup_metrics.csv')
    by_vla = pd.read_csv(rq2/'performance_by_vla.csv')
    complete = pd.read_csv(rq2/'overall_performance.csv')
    if len(overall)!=34 or overall.formula_id.duplicated().any(): raise ValueError('Expected 34 unique RQ1 formulas')
    original = dict(zip(overall.formula_id, overall.macro_roc_auc.astype(float)))
    ranks = tie_aware_ranks(original)
    top = [fid for fid, rank in ranks.items() if rank <= 5]
    if len(top)<5: raise ValueError('Invalid RQ1 top-five ranks')
    available = [v for v in VLA_ORDER if v in set(by_vla.vla.astype(str))]
    if not available or set(by_vla.vla.astype(str)) != set(available): raise ValueError('Invalid RQ2 VLA set')
    all_complete = set(available) == set(VLA_ORDER)
    rows=[]
    for _,r in complete.iterrows():
        if not truth(r['complete_primary_comparison']) or str(r['method_id'])=='harmonic_mean': continue
        vals={}
        for vla in VLA_ORDER:
            cells=by_vla[(by_vla.method_id==r['method_id']) & (by_vla.vla==vla)]
            if vla in available and len(cells)!=1: raise ValueError('Missing baseline cell for '+str(r['method_id']))
            if len(cells)>1: raise ValueError('Duplicate baseline cell')
            vals[DISPLAY[vla]]=float(cells.iloc[0].macro_roc_auc) if len(cells) else float('nan')
        vals['Overall']=sum(vals[DISPLAY[v]] for v in VLA_ORDER)/4 if all_complete else float('nan')
        rows.append({'method_id':str(r['method_id']),'Method':str(r['method']),'Family':str(r['family']),
                     'task_aware':truth(r['task_aware']),'is_sbfl':False,**vals})
    for fid in top:
        vals={}
        for vla in VLA_ORDER:
            cells=groups[(groups.dimension=='vla') & (groups.group==vla) & (groups.formula_id==fid)]
            if len(cells)!=1: raise ValueError('Missing or duplicate RQ1 subgroup for '+fid+' '+vla)
            vals[DISPLAY[vla]]=float(cells.iloc[0].macro_roc_auc)
        mean=sum(vals[DISPLAY[v]] for v in VLA_ORDER)/4
        if not math.isclose(mean,original[fid],rel_tol=1e-10,abs_tol=1e-10): raise ValueError('RQ1 subgroup/overall mismatch: '+fid)
        vals['Overall']=mean
        method=overall.loc[overall.formula_id==fid,'technique'].iloc[0]
        rows.append({'method_id':fid,'Method':str(method),'Family':'VoxelSBFL','task_aware':False,'is_sbfl':True,**vals})
    # Validate the shared reference before combining the two analyses.
    for vla in available:
        reference=by_vla[(by_vla.method_id=='harmonic_mean') & (by_vla.vla==vla)]
        rq1=groups[(groups.dimension=='vla') & (groups.group==vla) & (groups.formula_id=='harmonic_mean')]
        if len(reference)!=1 or len(rq1)!=1 or not math.isclose(float(reference.iloc[0].macro_roc_auc),float(rq1.iloc[0].macro_roc_auc),abs_tol=1e-12):
            raise ValueError('RQ1/RQ2 reference mismatch for '+vla)
    numeric=[DISPLAY[v] for v in VLA_ORDER]+['Overall']
    sortable='Overall' if all_complete else DISPLAY[available[0]]
    rows.sort(key=lambda r:(-r[sortable],r['method_id']))
    if all_complete:
        combined_rank=tie_aware_ranks({r['method_id']:r['Overall'] for r in rows})
    else: combined_rank={}
    for r in rows:r['Rank']=combined_rank.get(r['method_id'],'')
    output.mkdir(parents=True,exist_ok=True)
    frame=pd.DataFrame(rows)
    frame.to_csv(output/'rq2_baseline_performance_top5_rq1.csv',index=False,float_format='%.12g')
    bs=chr(92)
    # Rank highlighting is meaningful only where at least one baseline is present.
    highlight_cols={DISPLAY[v] for v in available}
    if all_complete: highlight_cols.add('Overall')
    best={}; second={}
    for col in highlight_cols:
        vals=sorted({float(r[col]) for r in rows if not math.isnan(float(r[col]))},reverse=True)
        best[col]=vals[0]
        second[col]=next((v for v in vals[1:] if not math.isclose(v,vals[0],rel_tol=1e-12,abs_tol=1e-12)),None)
    note=('Overall and joint ranks are unavailable for baselines until all four VLAs are available; '
          'blank cells indicate unavailable results. Highlighting is limited to columns containing baselines.' if not all_complete else
          'Overall is the equal-weight mean over the four VLAs; Rank is the combined ranking.')
    caption=('RQ2 trajectory-level failure-discrimination ROC-AUC for baselines and the top-five RQ1 SBFL techniques. '
             '$^{'+bs+'star}$ identifies an SBFL technique and $^{'+bs+'dagger}$ a task-aware baseline. '
             'Bold values denote the best and underlined values the second-best in comparable columns. '+note)
    lines=[bs+'begin{table*}[t]',bs+'centering',bs+'caption{'+caption+'}',bs+'label{tab:rq2-baselines-top5-rq1}',
           '{'+bs+'scriptsize',bs+'setlength{'+bs+'tabcolsep}{4pt}',bs+'begin{tabular}{lcccccc}',bs+'toprule',
           'Method & EO1 & GR00T & Pi0 & SpatialVLA & Overall & Rank '+bs*2,bs+'midrule']
    for r in rows:
        name=latex_escape(r['Method'])
        if r['is_sbfl']: name += '$^{'+bs+'star}$'
        if r['task_aware']: name += '$^{'+bs+'dagger}$'
        cells=[]
        for col in numeric:
            val=float(r[col])
            if math.isnan(val): cells.append(''); continue
            txt=f'{val:.4f}'
            if col in highlight_cols:
                if math.isclose(val,best[col],rel_tol=1e-12,abs_tol=1e-12):txt=bs+'textbf{'+txt+'}'
                elif second[col] is not None and math.isclose(val,second[col],rel_tol=1e-12,abs_tol=1e-12):txt=bs+'underline{'+txt+'}'
            cells.append(txt)
        lines.append(name+' & '+' & '.join(cells)+' & '+str(r['Rank'])+' '+bs*2)
    lines += [bs+'bottomrule',bs+'end{tabular}','}',bs+'end{table*}']
    (output/'rq2_baseline_performance_top5_rq1.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Top RQ1 SBFL:', ', '.join(str(overall.loc[overall.formula_id==f,'technique'].iloc[0]) for f in sorted(top,key=lambda f:(-original[f],f))))
    print('RQ2 baseline models:', ', '.join(available))
    print('Output:',output)
    return output/'rq2_baseline_performance_top5_rq1.csv', output/'rq2_baseline_performance_top5_rq1.tex'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rq1-aggregate',type=Path,default=DEFAULT_RQ1)
    p.add_argument('--rq2-aggregate',type=Path,default=DEFAULT_RQ2)
    p.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT)
    a=p.parse_args(); generate(a.rq1_aggregate,a.rq2_aggregate,a.output_dir)

if __name__=='__main__':main()
