#!/usr/bin/env python3
"""Compare RQ1 class distributions with RQ5 pilot sample-size scenarios.

The pilot is the sole source of q; the 504-scene final data are not read.
"""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = Path('/home/ubuntu/Deephyperion_VLA/results')
DEFAULT_AUDIT = RESULTS / 'rq1_current_recalculated/audit/episode_manifest.csv.gz'
DEFAULT_PILOT = RESULTS / 'move_near_100'
DEFAULT_OUT = RESULTS / 'rq5_sample_size_rq1_comparison'
MODELS = [('eo1', 'EO1'), ('gr00t', 'Groot'), ('pi0', 'Pi0'), ('spatialvla-4b', 'SpatialVLA')]
DISPLAY = {'EO1': 'EO1', 'Groot': 'GR00T', 'Pi0': 'Pi0', 'SpatialVLA': 'SpatialVLA'}
CONDITIONS = ('DH', 'DH_HybridDT', 'DH_HybridRF', 'DH_WeightedRF')


def positive(value):
    x = int(value)
    if x < 1: raise argparse.ArgumentTypeError('must be a positive integer')
    return x


def hash_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


def read_rq1(path):
    import gzip
    opener = gzip.open if path.suffix == '.gz' else open
    groups = defaultdict(Counter); ids = set()
    with opener(path, 'rt', encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            vla, condition, budget, run = row['vla'], row['condition'], int(row['budget']), row['run_id']
            if vla not in DISPLAY or condition not in CONDITIONS or budget not in (200, 400):
                raise ValueError('Unexpected RQ1 configuration')
            key = (vla, condition, budget, run, row['episode_id'])
            if key in ids: raise ValueError('Duplicate RQ1 episode')
            ids.add(key)
            label = int(row['failure'])
            if label not in (0, 1): raise ValueError('Non-binary RQ1 failure')
            groups[(vla, condition, budget, run)][label] += 1
    expected = {(vla, c, b, f'run_{r}') for vla in DISPLAY for c in CONDITIONS for b in (200, 400) for r in range(1, 11)}
    if set(groups) != expected: raise ValueError('RQ1 must contain all 320 configuration/run blocks')
    if any(sum(count.values()) != budget for (vla, condition, budget, run), count in groups.items()):
        raise ValueError('RQ1 run episode counts do not match budget')
    return groups


def get_success(path):
    log = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(log, dict) or not log: raise ValueError('Invalid log: ' + str(path))
    value = log[max(log, key=int)].get('success')
    if isinstance(value, str): value = {'true': True, 'false': False}.get(value.strip().lower())
    if not isinstance(value, bool): raise ValueError('Invalid final success: ' + str(path))
    return value


def read_pilot(root):
    result = {}; common = None
    for model, vla in MODELS:
        path = root / model; config = json.loads((path / 'run_config.json').read_text())
        signature = tuple(config.get(k) for k in ('task', 'seed', 'tasks_sha256', 'prompts_sha256'))
        if config.get('model') != model or any(x is None for x in signature): raise ValueError('Invalid pilot config: ' + model)
        if common is None: common = signature
        if signature != common: raise ValueError('Pilot inputs differ across models')
        logs = list((path/'allMetrics').glob('*/log.json'))
        if {x.parent.name for x in logs} != {str(i) for i in range(100)} or len(logs) != 100:
            raise ValueError(model + ' pilot must contain exactly 100 completed episodes')
        passed = sum(get_success(path/'allMetrics'/str(i)/'log.json') for i in range(100))
        result[vla] = {'passes': passed, 'failures': 100-passed,
                       'q': Fraction(min(passed, 100-passed),100)}
    return result, common


def ceil_fraction(x): return -(-x.numerator // x.denominator)


def design(q, k, target, floor):
    if q == 0: return None
    raw = ceil_fraction(Fraction(target, 1) / q)
    minimum = max(floor, raw)
    return {'target': target, 'floor': floor, 'raw_minimum': raw,
            'N': k * ((minimum + k - 1) // k)}


def binomial_probability_both(n, p, threshold):
    """P(threshold <= X <= n-threshold) under independent Bernoulli trials."""
    if 2*threshold > n: return 0.0
    if p == 0: return float(threshold == 0)
    if p == 1: return float(threshold == 0)
    return math.fsum(math.exp(math.lgamma(n+1)-math.lgamma(i+1)-math.lgamma(n-i+1)
                              + i*math.log(p)+(n-i)*math.log1p(-p))
                     for i in range(threshold,n-threshold+1))


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rq1-audit', type=Path, default=DEFAULT_AUDIT)
    p.add_argument('--pilot', type=Path, default=DEFAULT_PILOT)
    p.add_argument('--output', type=Path, default=DEFAULT_OUT)
    p.add_argument('--regions', type=positive, default=72)
    p.add_argument('--minority-target', type=positive, default=100)
    p.add_argument('--floor', type=positive, default=500)
    p.add_argument('--smaller-minority-target', type=positive, default=90)
    p.add_argument('--smaller-floor', type=positive, default=400)
    args=p.parse_args()
    groups=read_rq1(args.rq1_audit); pilot, signature=read_pilot(args.pilot)
    q=min(x['q'] for x in pilot.values()); primary=design(q,args.regions,args.minority_target,args.floor)
    alternative=design(q,args.regions,args.smaller_minority_target,args.smaller_floor)
    if primary is None: raise ValueError('Pilot has zero minority outcomes; cannot calculate finite N')
    if args.output.exists() and any(args.output.iterdir()): raise FileExistsError('Output directory is not empty: '+str(args.output))
    args.output.mkdir(parents=True, exist_ok=True)
    rows=[]; per_run=[]
    for _,vla in MODELS:
        for budget in (200,400):
            selected=[(key,c) for key,c in groups.items() if key[0]==vla and key[2]==budget]
            passed=sum(c[0] for _,c in selected); failed=sum(c[1] for _,c in selected); n=passed+failed
            rows.append({'model':DISPLAY[vla], 'budget_per_run':budget,'runs':len(selected),
                         'episodes':n,'pass':passed,'fail':failed,'pass_proportion':passed/n,
                         'fail_proportion':failed/n,'pooled_minority_proportion':min(passed,failed)/n,
                         'mean_run_minority_count':sum(min(c[0],c[1]) for _,c in selected)/len(selected)})
            for (v,c,b,run), count in sorted(selected):
                per_run.append({'model':DISPLAY[v], 'condition':c, 'budget':b, 'run_id':run,
                                'pass':count[0], 'fail':count[1], 'minority_count':min(count[0],count[1])})
    write_csv(args.output/'rq1_distribution_by_model_budget.csv',rows)
    write_csv(args.output/'rq1_distribution_by_run.csv',per_run)
    pilot_rows=[{'model':DISPLAY[v], 'episodes':100,'pass':pilot[v]['passes'],'fail':pilot[v]['failures'],
                 'minority_proportion':float(pilot[v]['q']),
                 'required_for_100_expected':ceil_fraction(Fraction(100,1)/pilot[v]['q']) if pilot[v]['q'] else ''}
                for _,v in MODELS]
    write_csv(args.output/'rq5_pilot_distribution.csv',pilot_rows)
    scenarios=[]
    for label,d in [('published_rule',primary),('illustrative_reduced_target',alternative)]:
        n=d['N']; scenarios.append({'scenario':label,'regions':args.regions,'target_expected_minority':d['target'],
            'minimum_floor':d['floor'],'unrounded_minimum':d['raw_minimum'],'selected_N':n,
            'scenes_per_region':n//args.regions,'pilot_q':float(q),'expected_minority':n*float(q),
            'probability_at_least_100_each_iid_fixed_q':binomial_probability_both(n,float(q),100)})
    write_csv(args.output/'rq5_sample_size_scenarios.csv',scenarios)
    bs=chr(92)
    table=[bs+'begin{table}[t]',bs+'centering',bs+'caption{RQ1 pick-up outcome distribution by model and run budget.}',
           bs+'label{tab:rq1-outcome-distribution}',bs+'begin{tabular}{lrrrrr}',bs+'hline',
           'VLA & Budget & Runs & Pass & Fail & Pass (\\%) '+bs*2,bs+'hline']
    for r in rows:
        table.append(f"{r['model']} & {r['budget_per_run']} & {r['runs']} & {r['pass']} & {r['fail']} & {r['pass_proportion']*100:.2f} "+bs*2)
    table += [bs+'hline',bs+'end{tabular}',bs+'end{table}']
    (args.output/'rq1_outcome_distribution.tex').write_text('\n'.join(table)+'\n')
    lines=[bs+'begin{table}[t]',bs+'centering',bs+'caption{RQ5 pilot-based sample-size scenarios.}',bs+'label{tab:rq5-sample-scenarios}',
           bs+'begin{tabular}{lrrrr}',bs+'hline','Scenario & $K$ & Target & $N$ & Expected minority '+bs*2,bs+'hline']
    for r in scenarios:
        label='Current rule' if r['scenario']=='published_rule' else 'Illustrative alternative'
        lines.append(f"{label} & {r['regions']} & {r['target_expected_minority']} & {r['selected_N']} & {r['expected_minority']:.2f} "+bs*2)
    lines += [bs+'hline',bs+'end{tabular}',bs+'end{table}']
    (args.output/'rq5_sample_size_scenarios.tex').write_text('\n'.join(lines)+'\n')
    text=f'''# RQ1 outcome distribution and RQ5 sample-size scenarios

RQ1 uses 200 and 400 episodes per held-out run (40 runs per model at each budget).
See `rq1_distribution_by_model_budget.csv` and `rq1_distribution_by_run.csv`.
The budgets are not independent replications: RQ1 found seven exact overlaps,
and its main protocol weights budgets equally. Pooled counts are descriptive.

The independent RQ5 pilot's minimum minority proportion is {float(q):.4f}.
With K={args.regions}, the current rule (target {args.minority_target} expected
minority cases, minimum floor {args.floor}) gives N={primary['N']}, or
{primary['N']//args.regions} scenes per region. A smaller N cannot satisfy this
same rule and equal allocation across all {args.regions} regions.

Illustrative sensitivity only: if the study changes its requirement to {args.smaller_minority_target}
expected minority cases and the floor to {args.smaller_floor}, the same formula
gives N={alternative['N']}, with {alternative['N']//args.regions} scenes per region and
{alternative['N']*float(q):.2f} expected minority cases at the pilot estimate.
This is a changed protocol, not evidence that the original 100-per-class target
is met. RQ1's 400 budget is a precedent for a run size, not a power analysis.

Approximate binomial P(at least 100 pass and 100 fail), treating the pilot
proportion as known and executions as independent, is
{binomial_probability_both(primary['N'],float(q),100):.4f} for N={primary['N']} and
{binomial_probability_both(alternative['N'],float(q),100):.4f} for N={alternative['N']}.
These are model-based illustrations, not guarantees; the pilot estimate is
uncertain and balanced/paired designs can violate independence.

For the reported RQ5 ROC-AUC, a balanced 100-pass/100-fail evaluation sample
requires at least 100 observed examples of each class. If a reduced N produces
fewer than 100, that evaluation cannot use 100 per class. Choosing a subset
or sample size after seeing final outcomes is exploratory and must be disclosed;
fix any decision rule before confirmatory analysis. Since 504 final scenes have
already been generated, reducing the number is not needed for their generation.
'''
    (args.output/'interpretation.md').write_text(text,encoding='utf-8')
    paper = f"""# Draft language for the paper

## Description consistent with the current protocol

RQ1 evaluated held-out runs of 200 and 400 pickup executions per VLA and
training condition. The RQ5 pilot comprised 100 move-near executions per VLA.
The smallest observed class proportion was {float(q):.2f}. With {args.regions}
destination regions, a minimum overall budget of {args.floor}, and a target of
at least {args.minority_target} expected executions in the minority class, the
predefined sample-size formula yields N={primary['N']} executions per VLA,
with {primary['N']//args.regions} scenes per destination region. The RQ5
ROC-AUC analysis subsequently selected 100 passes and 100 failures per model
from the final executions using a fixed random seed.

## Alternative design language, only if the study protocol is changed

If the target were changed to {args.smaller_minority_target} expected minority
outcomes and the minimum budget to {args.smaller_floor}, the same rounding rule
would yield N={alternative['N']}, or {alternative['N']//args.regions} executions
per destination region. At the pilot estimate this corresponds to
{alternative['N']*float(q):.2f} expected minority outcomes. This alternative
does not satisfy the original {args.minority_target}-expected-per-class rule;
it must not be described as the original prespecified design. Because final
outcomes have already been observed, a newly selected reduced subset is a
post-hoc analysis unless independently confirmed.
"""
    (args.output/'paper_text_suggestions.md').write_text(paper,encoding='utf-8')
    metadata={'rq1_audit':str(args.rq1_audit.resolve()),'rq1_audit_sha256':hash_file(args.rq1_audit),
              'pilot_root':str(args.pilot.resolve()),'pilot_signature':signature,
              'pilot_manifest_sha256':{m:hash_file(args.pilot/m/'run_config.json') for m,_ in MODELS},
              'arguments':vars(args)|{'rq1_audit':str(args.rq1_audit),'pilot':str(args.pilot),'output':str(args.output)},
              'note':'Pilot-only N calculation; no final RQ5 outcomes used.'}
    (args.output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    print(text); print('Results:',args.output.resolve())

if __name__ == '__main__': main()
