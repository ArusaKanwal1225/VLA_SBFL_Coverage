"""RQ1 script 1/3: load Pi0 200 episodes, use grouped 70/30 splits,
voxelize trajectories, and calculate 14 TRAINING-ONLY SBFL maps.

From repository root:
  python src/01_rq1_train_suspiciousness.py

This reuses the validated old split manifest when present. To reproduce
from extracted input alone, first review scene metadata then use:
  python src/01_rq1_train_suspiciousness.py --allow-new-splits

Requires numpy, pandas, matplotlib; shared rq1_common.py in src/.
Original input and old RQ1 outputs are never changed.
"""
import argparse
import csv
import gzip
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rq1_common import (ALGORITHMS, SEEDS, METHODS, GROUP_KEY, ORIGIN, VOXEL_SIZE,
                        REL_OUTPUT, FORMULAS, require, sbfl_scores, test_formulas,
                        out_csv, out_json, fig_save, sha256, get_root,
                        verify_40_splits)


def load_episodes_and_visits(root):
    episode_tables = []
    visits = {}
    stats = []
    origin = np.array(ORIGIN, dtype='float64')
    for alg in ALGORITHMS:
        visits[alg] = {}
        for run_number in range(1, 11):
            run = 'run_' + str(run_number)
            folder = root / 'input' / 'Pi0' / alg / run
            epfile, trfile = folder / 'episodes.csv', folder / 'trajectories.csv'
            require(epfile.is_file() and trfile.is_file(), 'Missing input CSV: ' + str(folder))
            ep = pd.read_csv(epfile, dtype={'episode_id':str, 'run_id':str,
                                           'algorithm':str}, encoding='utf-8-sig')
            needed = {'episode_id','run_id','algorithm','failure','num_steps'}
            require(needed.issubset(ep.columns), 'Missing episode fields: ' + str(epfile))
            require(len(ep) == 200 and ep.episode_id.nunique() == 200,
                    f'Expected 200 unique episodes: {folder}')
            require((ep.run_id == run).all() and (ep.algorithm == alg).all()
                    and ep.failure.isin([0,1]).all(), 'Episode labels or identifiers invalid: ' + str(folder))
            traj = pd.read_csv(trfile, encoding='utf-8-sig',
                               dtype={'episode_id':str}, usecols=['episode_id','step','x','y','z'])
            require(len(traj) > 0 and traj[['x','y','z']].notna().all().all(),
                    'Empty trajectory or missing XYZ: ' + str(trfile))
            xyz = traj[['x','y','z']].to_numpy(dtype='float64')
            require(np.isfinite(xyz).all(), 'Nonfinite coordinates: ' + str(trfile))
            idx = np.floor((xyz - origin) / VOXEL_SIZE).astype('int64')
            traj['voxel'] = [','.join(map(str, v)) for v in idx]
            expected = dict(zip(ep.episode_id, ep.num_steps.astype(int)))
            require(set(traj.episode_id) == set(expected),
                    'Episode IDs differ between labels and trajectories: ' + str(folder))
            for eid, block in traj.groupby('episode_id', sort=False):
                steps = block.step.to_numpy(dtype=int)
                require(len(steps) == expected[eid] and
                        np.array_equal(steps, np.arange(len(steps))),
                        'Trajectory steps differ from episode manifest: ' + eid)
                vox = frozenset(block.voxel)
                require(len(vox) > 0 and eid not in visits[alg],
                        'Empty or duplicate coverage episode: ' + eid)
                visits[alg][eid] = vox
                stats.append({'algorithm':alg,'episode_id':eid,
                              'unique_voxels':len(vox),'samples':len(steps)})
            episode_tables.append(ep[['algorithm','run_id','episode_id','failure']])
        require(len(visits[alg]) == 2000, f'Expected 2000 episodes for {alg}')
        print(f'[INPUT] {alg}: episodes=2000, coverage objects=2000', flush=True)
    episodes = pd.concat(episode_tables, ignore_index=True)
    require(len(episodes) == 8000 and episodes.episode_id.nunique() == 8000,
            'All input episodes must have globally unique IDs')
    return episodes, visits, pd.DataFrame(stats)


def make_new_splits(root, episodes):
    meta_path = root / 'input' / 'Pi0' / 'scene_metadata_candidates_200.csv'
    require(meta_path.is_file(),
            'Cannot generate scene-grouped splits: missing ' + str(meta_path))
    metadata = pd.read_csv(meta_path, dtype=str, keep_default_na=False,
                           encoding='utf-8-sig')
    required = {'algorithm','run_id','episode_id','failure',GROUP_KEY}
    require(required.issubset(metadata.columns) and len(metadata) == 8000,
            'Scene metadata incomplete: ' + str(meta_path))
    for col in required:
        require((metadata[col] != '').all(), 'Blank scene metadata field: ' + col)
    baseline = episodes.copy()
    baseline['failure'] = baseline['failure'].astype(str)
    joined = baseline.merge(metadata[['algorithm','run_id','episode_id','failure',GROUP_KEY]],
                            on=['algorithm','run_id','episode_id','failure'],how='inner',
                            validate='one_to_one')
    require(len(joined) == 8000, 'Metadata episode identity/outcomes do not match input')
    rng_rows = []
    for alg in ALGORITHMS:
        sub = joined[joined.algorithm == alg].copy()
        group_infos = []
        for key, block in sub.groupby(GROUP_KEY, sort=True):
            n_fail = int((block.failure == '1').sum())
            n_pass = int((block.failure == '0').sum())
            category = 'mixed' if n_fail and n_pass else ('fail' if n_fail else 'pass')
            primary_run = min(block.run_id, key=lambda x: int(x.rsplit('_',1)[1]))
            group_infos.append((key, primary_run, category))
        strata = defaultdict(list)
        for key, run, category in group_infos:
            strata[(run, category)].append(key)
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            test_groups = set()
            for stratum in sorted(strata):
                keys = strata[stratum]
                count = int(round(0.3 * len(keys)))
                for pick in rng.choice(len(keys), size=count, replace=False):
                    test_groups.add(keys[int(pick)])
            block = sub[['algorithm','run_id','episode_id','failure',GROUP_KEY]].copy()
            block['seed'] = seed
            block['split'] = np.where(block[GROUP_KEY].isin(test_groups),'test','train')
            rng_rows.append(block)
    return pd.concat(rng_rows, ignore_index=True)


def select_splits(root, episodes, allow_new):
    frozen = root / 'output' / 'RQ1' / 'pi0_200' / 'step03_grouped_splits' / 'split_manifest.csv'
    if frozen.is_file():
        print('[SPLITS] Reusing validated frozen Step03 splits: ' + str(frozen))
        df = pd.read_csv(frozen, encoding='utf-8-sig',
                         dtype={'episode_id':str, 'run_id':str, 'algorithm':str})
        origin = str(frozen)
    else:
        require(allow_new,
                'Frozen split manifest missing. To regenerate after manually reviewing '
                'scene identity, run with --allow-new-splits.')
        df = make_new_splits(root, episodes)
        origin = 'new splits generated from candidate scene grouping key'
    df['seed'] = df['seed'].astype(int)
    df['failure'] = df['failure'].astype(int)
    verify_40_splits(df, episodes)
    return df, origin


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo-root', default=None)
    ap.add_argument('--allow-new-splits', action='store_true',
                    help='Explicit approval to generate grouped splits if previous manifest is absent')
    args = ap.parse_args()
    root = get_root(__file__, args.repo_root)
    out = root / REL_OUTPUT
    out.mkdir(parents=True, exist_ok=True)
    test_formulas()
    episodes, visits, ep_stats = load_episodes_and_visits(root)
    split, source = select_splits(root, episodes, args.allow_new_splits)
    out_csv(split, out / 'split_manifest.csv.gz', compression='gzip')
    out_csv(ep_stats, out / 'per_episode_voxel_statistics.csv')
    with gzip.open(out / 'episode_voxel_visits.csv.gz','wt',encoding='utf-8',newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['algorithm','episode_id','voxel'])
        for alg in ALGORITHMS:
            for eid in sorted(visits[alg]):
                for voxel in sorted(visits[alg][eid]):
                    writer.writerow((alg,eid,voxel))

    fields = ['algorithm','seed','voxel','ef','ep','nf','np','support'] + list(METHODS)
    run_rows, example_rows = [], []
    tmp = out / 'training_voxel_scores.csv.gz.incomplete'
    with gzip.open(tmp,'wt',encoding='utf-8',newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for alg in ALGORITHMS:
            alg_ids = visits[alg]
            a = split[split.algorithm == alg]
            for seed in SEEDS:
                s = a[a.seed == seed]
                train = s[s.split == 'train']
                test = s[s.split == 'test']
                require(len(train) + len(test) == 2000,'Split length invalid')
                total_fail = int(train.failure.sum()); total_pass = int(len(train)-total_fail)
                require(total_fail > 0 and total_pass > 0, 'Single-class training data')
                ef, ep = Counter(), Counter()
                for r in train.itertuples(index=False):
                    target = ef if r.failure == 1 else ep
                    target.update(alg_ids[r.episode_id])
                voxels = sorted(set(ef) | set(ep))
                efv = np.array([ef.get(v, 0) for v in voxels], dtype=int)
                epv = np.array([ep.get(v, 0) for v in voxels], dtype=int)
                nfv, npv = total_fail-efv, total_pass-epv
                scores = sbfl_scores(efv, epv, nfv, npv)
                support = efv + epv
                for j, voxel in enumerate(voxels):
                    rec = dict(algorithm=alg,seed=seed,voxel=voxel,
                               ef=int(efv[j]),ep=int(epv[j]),nf=int(nfv[j]),
                               np=int(npv[j]),support=int(support[j]))
                    for method in METHODS:
                        rec[method] = float(scores[method][j])
                    writer.writerow(rec)
                    if seed == 11 and j < 50:
                        example_rows.append(rec)
                test_vox = set().union(*(alg_ids[e] for e in test.episode_id))
                train_vox = set(voxels)
                run_rows.append({'algorithm':alg,'seed':seed,
                                 'train_episodes':len(train),'test_episodes':len(test),
                                 'train_failures':total_fail,'train_passes':total_pass,
                                 'train_voxels':len(voxels),'test_voxels':len(test_vox),
                                 'known_test_voxel_fraction':len(train_vox & test_vox)/len(test_vox),
                                 'sparse_fraction_support_le2':float(np.mean(support <= 2))})
                print(f'[TRAINED] {alg} seed={seed} training_voxels={len(voxels)}',flush=True)
    tmp.replace(out / 'training_voxel_scores.csv.gz')
    sm = pd.DataFrame(run_rows)
    require(len(sm) == 40, 'Expected 40 trained suspiciousness maps with 14 techniques each')
    out_csv(sm, out / 'training_split_summary.csv')
    out_csv(pd.DataFrame(example_rows), out / 'example_seed11_voxels.csv')
    out_csv(pd.DataFrame([{'method':k,'formula':v} for k,v in FORMULAS.items()]),
            out / 'formula_catalog.csv')
    figs = out / 'figures'
    for col, title, name in [
        ('train_voxels','Distinct training-covered voxels','01_training_voxels_by_seed'),
        ('sparse_fraction_support_le2','Fraction of training voxels with support <= 2',
         '02_sparse_voxel_fraction'),
        ('known_test_voxel_fraction','Fraction of test voxels known in training',
         '03_known_test_voxel_fraction')]:
        fig, ax = plt.subplots(figsize=(9,4.6))
        for alg in ALGORITHMS:
            s = sm[sm.algorithm == alg]
            ax.plot(s.seed,s[col],marker='o',label=alg.replace('_200scenes',''))
        ax.set_xlabel('Split seed');ax.set_ylabel(title);ax.set_title('Pi0 200: ' + title)
        ax.legend(fontsize=8)
        fig_save(fig,figs,name)
    report = {'status':'PASS','model':'Pi0','budget':200,'algorithms':list(ALGORITHMS),
              'seeds':list(SEEDS),'methods':list(METHODS),'runs':40,'episodes':8000,
              'trained_splits':40,'method_maps':560,'voxel_size_m':VOXEL_SIZE,
              'origin_m':ORIGIN,'scene_key':GROUP_KEY,'split_source':source,
              'train_only_scores':True,
              'scene_identity_caveat':'Candidate scene key; cannot rule out additional simulator context',
              'split_sha256':sha256(out/'split_manifest.csv.gz'),
              'scores_sha256':sha256(out/'training_voxel_scores.csv.gz')}
    out_json(report,out/'01_training_report.json')
    print('\nRQ1 TRAINING COMPLETE: 40/40 splits; 560/560 maps; status PASS')
    print('Saved to:',out)

if __name__ == '__main__':
    main()
