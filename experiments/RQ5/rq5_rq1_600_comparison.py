#!/usr/bin/env python3
"""RQ1 pooled 200+400 distributions and RQ5 pilot/final sample audit."""
import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from rq5_sample_size_comparison import (DEFAULT_AUDIT, DEFAULT_PILOT, MODELS, DISPLAY,
    read_rq1, read_pilot, design, binomial_probability_both, hash_file)

BASE = Path('/home/ubuntu/Deephyperion_VLA/results')
META = Path(__file__).resolve().parents[2] / 'output/Pick_and_place_504/scene_metadata_504.json'


def write(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rq1-audit', type=Path, default=DEFAULT_AUDIT)
    p.add_argument('--pilot', type=Path, default=DEFAULT_PILOT)
    p.add_argument('--final', type=Path, default=BASE/'move_near_504')
    p.add_argument('--sample', type=Path, default=BASE/'rq5_transfer_final/sample_manifest.csv')
    p.add_argument('--scene-metadata', type=Path, default=META)
    p.add_argument('--output', type=Path, default=BASE/'rq5_rq1_600_analysis')
    a=p.parse_args(); groups=read_rq1(a.rq1_audit); pilot,_=read_pilot(a.pilot)
    with a.sample.open(encoding='utf-8-sig', newline='') as f: sample=list(csv.DictReader(f))
    meta=json.loads(a.scene_metadata.read_text())
    if a.output.exists() and any(a.output.iterdir()): raise FileExistsError('Output directory is not empty: '+str(a.output))
    rows=[]; coverage=[]; chosen_sets={}
    for model,vla in MODELS:
        counts=Counter()
        for (v,condition,budget,run), outcome in groups.items():
            if v==vla: counts.update(outcome)
        episodes_path=a.final/model/'episodes.csv'
        with episodes_path.open(encoding='utf-8-sig',newline='') as f: final=list(csv.DictReader(f))
        if len(final)!=504 or {r['eval_id'] for r in final}!={str(i) for i in range(504)}: raise ValueError('Incomplete final executions: '+model)
        final_counts=Counter(r['failure'] for r in final)
        chosen=[r for r in sample if r['model']==model]
        if len(chosen)!=200 or Counter(r['failure'] for r in chosen)!={'0':100,'1':100}: raise ValueError('Invalid RQ5 sample: '+model)
        lookup={r['episode_id']:r for r in final}
        if any(r['episode_id'] not in lookup or lookup[r['episode_id']]['failure']!=r['failure'] or lookup[r['episode_id']]['eval_id']!=r['eval_id'] for r in chosen): raise ValueError('Sample and final CSV disagree: '+model)
        chosen_sets[model]={r['eval_id'] for r in chosen}
        cellcounts=Counter((meta[r['eval_id']]['cell_x'],meta[r['eval_id']]['cell_y']) for r in chosen)
        objcounts=Counter(r['object_name'] for r in chosen)
        rows.append({'model':DISPLAY[vla],'rq1_pooled_episodes':sum(counts.values()),'rq1_pass':counts[0], 'rq1_fail':counts[1],
                     'rq1_pass_proportion':counts[0]/sum(counts.values()), 'rq1_fail_proportion':counts[1]/sum(counts.values()),
                     'pilot_pass':pilot[vla]['passes'],'pilot_fail':pilot[vla]['failures'],
                     'pilot_minority_proportion':float(pilot[vla]['q']),'final_episodes':len(final),
                     'final_pass':final_counts['0'],'final_fail':final_counts['1']})
        coverage.append({'model':DISPLAY[vla], 'sample_episodes':len(chosen),'sample_pass':100,'sample_fail':100,
                         'destination_cells_covered':len(cellcounts),'destination_cells_missing':72-len(cellcounts),
                         'min_per_covered_cell':min(cellcounts.values()),'max_per_covered_cell':max(cellcounts.values()),
                         'object_counts_json':json.dumps(objcounts,sort_keys=True)})
    q=min(pilot[v]['q'] for _,v in MODELS); r=design(q,72,100,500)
    count_rq1_min=min(min(x['rq1_pass_proportion'], x['rq1_fail_proportion']) for x in rows)
    kfree=72*math.ceil(math.ceil(100/count_rq1_min)/72)
    result={'rq1_pooled_class_ratios_are_descriptive':True,'rq1_sum_counts':{'pass':sum(r['rq1_pass'] for r in rows),'fail':sum(r['rq1_fail'] for r in rows)},
            'pilot_q':float(q),'rq1_min_q':count_rq1_min,'regions':72,'floor':500,'minimum_expected_minority':100,
            'n_under_pilot_rule':r['N'],'n_under_rq1_q_with_floor':72*math.ceil(max(500,math.ceil(100/count_rq1_min))/72),
            'n_under_rq1_q_without_floor_illustrative':kfree,
            'p_at_least_100_each_assuming_fixed_q_and_iid':binomial_probability_both(r['N'],float(q),100),
            'number_of_scene_ids_shared_by_all_four_200_samples':len(set.intersection(*chosen_sets.values())),
            'source_sha256':{'rq1_audit':hash_file(a.rq1_audit),'sample_manifest':hash_file(a.sample),'scene_metadata':hash_file(a.scene_metadata)},
            'caveat':'200 and 400 source sets are not guaranteed independent; post-outcome sample design is descriptive.'}
    a.output.mkdir(parents=True,exist_ok=True); write(a.output/'rq1_600_and_rq5_counts.csv',rows); write(a.output/'rq5_sample_coverage.csv',coverage)
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2)); print('Output:',a.output)

if __name__=='__main__': main()
