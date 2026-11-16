#!/usr/bin/env python3
"""Select final move-near episodes matching RQ1 class ratios and cell balance."""
import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from rq5_rq1_outcomes import read_rq1, DEFAULT_AUDIT

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'data/rq5'
MODELS = [('eo1','EO1'),('gr00t','Groot'),('pi0','Pi0'),('spatialvla-4b','SpatialVLA')]


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def round_ratio(n, numerator, denominator):
    # Exact integer round-half-up, independent of floating-point rounding.
    return (2*n*numerator+denominator)//(2*denominator)


def choose_conditionally(cells, pass_target, per_cell, rng):
    """Uniform sample of subsets with exactly k per cell and pass_target passes."""
    ordered=sorted(cells)
    choices=[]
    for cell in ordered:
        quota=per_cell[cell] if isinstance(per_cell, dict) else per_cell
        passes=sorted((r for r in cells[cell] if r['failure']=='0'),key=lambda r:int(r['eval_id']))
        fails=sorted((r for r in cells[cell] if r['failure']=='1'),key=lambda r:int(r['eval_id']))
        opts=[(k, math.comb(len(passes),k)*math.comb(len(fails),quota-k))
              for k in range(max(0,quota-len(fails)),min(quota,len(passes))+1)]
        choices.append((passes,fails,opts,quota))
    # suffix[i][k] counts all valid episode subsets using cells i..end with k passes.
    suffix=[{} for _ in range(len(ordered)+1)]; suffix[-1]={0:1}
    for i in range(len(ordered)-1,-1,-1):
        acc=defaultdict(int)
        for k,weight in choices[i][2]:
            for remaining,nways in suffix[i+1].items():
                if k+remaining<=pass_target: acc[k+remaining]+=weight*nways
        suffix[i]=dict(acc)
    if pass_target not in suffix[0]: raise ValueError('No sample meets both cell and pass/fail constraints')
    remaining=pass_target; selected=[]
    for i,(passes,fails,opts,quota) in enumerate(choices):
        weights=[(k,weight*suffix[i+1].get(remaining-k,0)) for k,weight in opts]
        draw=rng.randrange(sum(w for _,w in weights))
        for k,w in weights:
            if draw<w: break
            draw-=w
        selected.extend(rng.sample(passes,k)); selected.extend(rng.sample(fails,quota-k)); remaining-=k
    if remaining: raise RuntimeError('Incorrect conditional sample')
    return selected


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rq1-audit',type=Path,default=DEFAULT_AUDIT)
    p.add_argument('--target',type=Path,default=BASE/'move_near_504')
    p.add_argument('--scene-metadata',type=Path,default=ROOT/'data/rq5/scenes_504/scene_metadata_504.json')
    p.add_argument('--output',type=Path,default=ROOT/'reproduced/rq5-selection')
    p.add_argument('--seed',type=int,default=20261028)
    p.add_argument('--per-cell',type=int,default=4)
    p.add_argument('--total',type=int,help='Scenes per model; cells differ by at most one')
    p.add_argument('--min-per-class',type=int,default=0)
    p.add_argument('--fixed-pass',type=int,help='Fix pass count; choose per-model N closest to the RQ1 ratio')
    args=p.parse_args()
    if args.per_cell<1 or args.per_cell>7: p.error('--per-cell must be between 1 and 7')
    if args.fixed_pass is not None and (args.fixed_pass<1 or args.total is not None): p.error('--fixed-pass must be positive and cannot be combined with --total')
    n = args.total if args.total is not None else 72*args.per_cell
    if not 72 <= n <= 504 or args.min_per_class < 0 or 2*args.min_per_class > n: p.error('Invalid total or minimum per class')
    low, extra = divmod(n, 72)
    # One shared reproducible set of 4/5-cell quotas for every model.
    # Cell (x,y) with all available pi0 passes may need the larger quota.
    allcells=[(x,y) for x in range(9) for y in range(8)]
    order_rng=random.Random(args.seed)
    order_rng.shuffle(allcells)
    if args.output.exists() and any(args.output.iterdir()): raise FileExistsError('Output directory is not empty: '+str(args.output))
    rq1=read_rq1(args.rq1_audit); metadata=json.loads(args.scene_metadata.read_text())
    scene_ids={str(i) for i in range(504)}
    if set(metadata)!=scene_ids: raise ValueError('Scene metadata must contain 504 IDs')
    manifest=[]; report=[]; run_configs={}
    for model,vla in MODELS:
        src=Counter()
        for (v,condition,budget,run),counts in rq1.items():
            if v==vla: src.update(counts)
        root=args.target/model
        with (root/'episodes.csv').open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
        if len(rows)!=504 or {r['eval_id'] for r in rows}!=scene_ids or any(r['model']!=model or r['failure'] not in ('0','1') for r in rows):
            raise ValueError('Expected 504 valid final episodes for '+model)
        if len({r['episode_id'] for r in rows})!=504: raise ValueError('Duplicate episode IDs for '+model)
        config=json.loads((root/'run_config.json').read_text())
        if config.get('model')!=model: raise ValueError('Invalid model config')
        run_configs[model]=[config.get(k) for k in ('task','seed','tasks_sha256','prompts_sha256')]
        cells=defaultdict(list)
        for row in rows:
            logpath=root/'allMetrics'/row['eval_id']/'log.json'; log=json.loads(logpath.read_text())
            val=log[max(log,key=int)]['success']
            if isinstance(val,str): val={'true':True,'false':False}.get(val.strip().lower())
            if not isinstance(val,bool) or int(not val)!=int(row['failure']): raise ValueError('Log/CSV mismatch: '+str(logpath))
            info=metadata[row['eval_id']]; cells[(info['cell_x'],info['cell_y'])].append(row)
        if len(cells)!=72 or any(len(v)!=7 for v in cells.values()): raise ValueError('Expected 72 cells with 7 scenes per cell')
        available=Counter(r['failure'] for r in rows)
        if args.fixed_pass is not None:
            if args.fixed_pass < args.min_per_class or args.fixed_pass > available['0']:
                raise ValueError('Insufficient pass episodes for '+model)
            possible = range(max(72,args.fixed_pass+args.min_per_class),
                             min(504,args.fixed_pass+available['1'])+1)
            n = min(possible, key=lambda total: (abs(Fraction(args.fixed_pass,total)-Fraction(src[0],sum(src.values()))),total))
            low, extra = divmod(n,72)
        raw_target=args.fixed_pass if args.fixed_pass is not None else round_ratio(n,src[0],sum(src.values()))
        lower=max(args.min_per_class,n-available['1'])
        upper=min(n-args.min_per_class,available['0'])
        if lower>upper:
            raise ValueError('Insufficient pass/fail episodes for '+model)
        pass_target=max(lower,min(upper,raw_target))
        seed=int.from_bytes(hashlib.sha256(f'{args.seed}:{model}'.encode()).digest()[:8],'big')
        # The quota order is model-specific and seeded; when every available
        # pass must be used, cells containing >low passes need the extra slot.
        mandatory={cell for cell in allcells if sum(r['failure']=='0' for r in cells[cell]) > low and pass_target == available['0']}
        if len(mandatory)>extra: raise ValueError('No cell quotas can include all required passes')
        extra_cells=set(mandatory)
        extra_cells.update(cell for cell in allcells if cell not in extra_cells and len(extra_cells)<extra)
        quotas={cell:low+(cell in extra_cells) for cell in allcells}
        selected=choose_conditionally(cells,pass_target,quotas,random.Random(seed))
        selected.sort(key=lambda r:int(r['eval_id']))
        for row in selected: row['target_model']=model
        manifest.extend(selected)
        report.append({'model':model,'rq1_episodes':sum(src.values()),'rq1_pass':src[0],'rq1_fail':src[1],
                       'rq1_pass_pct':100*src[0]/sum(src.values()),'selected_n':n,'selected_pass':pass_target,
                       'selected_fail':n-pass_target,'selected_pass_pct':100*pass_target/n,'pass_pct_difference_pp':100*pass_target/n-100*src[0]/sum(src.values()),
                       'cells':72,'cell_min':low,'cell_max':low+(extra>0),'cells_with_extra':extra})
    if any(any(x is None for x in v) for v in run_configs.values()) or len({tuple(v) for v in run_configs.values()})!=1:
        raise ValueError('Run inputs, prompts, task or seed differ between models')
    args.output.mkdir(parents=True,exist_ok=True)
    for name,rows in [('sample_manifest.csv',manifest),('selection_report.csv',report)]:
        with (args.output/name).open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    info={'seed':args.seed,'sample_per_model':{r['model']:r['selected_n'] for r in report},'cells':72,
          'cell_quotas_by_model':{r['model']:{'min':r['cell_min'],'max':r['cell_max'],'cells_with_extra':r['cells_with_extra']} for r in report},
          'min_per_class':args.min_per_class,
          'method':'uniform over subsets meeting per-cell and per-model class constraints; model-specific RNG',
          'class_target':('fixed pass count; N minimizes absolute difference from RQ1 ratio subject to class minimum and available counts' if args.fixed_pass is not None else 'round half up of N times RQ1 pooled pass proportion, clamped to minimum per class and available counts'),
          'rq1_audit_sha256':digest(args.rq1_audit),'scene_metadata_sha256':digest(args.scene_metadata),
          'source_configurations':run_configs,
          'note':'Retrospective outcome-stratified selection; differs from previously reported 100/100 sample. Cell quotas can differ by model.'}
    (args.output/'selection_metadata.json').write_text(json.dumps(info,indent=2)+'\n')
    print(json.dumps(report,indent=2)); print('Output:',args.output.resolve())

if __name__=='__main__': main()
