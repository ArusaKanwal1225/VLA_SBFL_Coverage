#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RQ1 Step 06 - Calculate 14 SBFL suspiciousness maps for Pi0, 400 scenes.

Run from the SBFL-only repository:
    python src/06_rq1_calculate_suspiciousness_pi0_400.py

Input (frozen Step 05):
    output/RQ1/pi0_400/step05_training_contingency/
        train_voxel_contingency_counts.csv.gz
        training_contingency_split_summary.csv
        step05_validation_report.json

Output:
    output/RQ1/pi0_400/step06_suspiciousness/
        voxel_suspiciousness_scores.csv.gz
        per_method_split_diagnostics.csv
        top_voxels_seed11.csv
        formula_definitions.csv
        step06_validation_report.json
        figures/*.png and *.pdf

No train/test split is changed. No held-out test outcome is used here.
Some SBFL formulas yield negative or unbounded scores; do NOT treat scores
as failure probabilities or compare raw magnitudes between methods.

Policy: zero denominators produce 0.0 for otherwise undefined 0/0 cases.
DStar2 positive/zero ratios have mathematically infinite rank; these are
encoded as (max finite DStar2 score + 1) WITHIN a split to preserve rank,
and counted in diagnostics. This remains a convention, not a probability.

References (formula conventions):
    https://www.mdpi.com/2227-7390/10/21/3924
    https://www.mdpi.com/2076-3417/13/22/12344
    https://citeseerx.ist.psu.edu/document?doi=9f24531686c349d39d30ac8c9da36991d72924f8&repid=rep1&type=pdf
"""
from __future__ import annotations

import argparse
import csv
import gzip
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
    'DH_400scenes',
    'DH_HybridDT_400scenes',
    'DH_HybridRF_400scenes',
    'DH_WeightedRF_400scenes',
)
SEEDS = (11, 23, 37, 41, 53, 67, 79, 83, 97, 101)
METHODS = (
    'Tarantula', 'Barinel', 'Kulczynski2', 'Wong2', 'Ochiai2',
    'Ochiai', 'DStar2', 'Jaccard', 'Wong3', 'GP13', 'OP2',
    'Wong1', 'Zoltar', 'Ample',
)
STEM = Path('output/RQ1/pi0_400')
STEP5 = 'step05_training_contingency'
STEP6 = 'step06_suspiciousness'
FORMULAS = {
    'Tarantula': '(ef/F) / ((ef/F) + (ep/P))',
    'Barinel': 'ef / (ef + ep)',
    'Kulczynski2': '0.5 * (ef/F + ef/(ef+ep))',
    'Wong2': 'ef - ep',
    'Ochiai2': '(ef*np) / sqrt((ef+ep)*(nf+np)*(ef+nf)*(ep+np))',
    'Ochiai': 'ef / sqrt(F*(ef+ep))',
    'DStar2': 'ef**2 / (ep+nf)',
    'Jaccard': 'ef / (ef+ep+nf)',
    'Wong3': 'ef - h(ep), with standard piecewise Wong3 h',
    'GP13': 'ef * (1 + 1/(ef+2*ep))',
    'OP2': 'ef - ep/(P+1)',
    'Wong1': 'ef',
    'Zoltar': 'ef / (ef+ep+nf + 10000*ep*nf/ef)',
    'Ample': 'abs(ef/F - ep/P)',
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def safe_div(n, d):
    """0.0 when denominator equals 0; standard IEEE division otherwise."""
    n, d = np.broadcast_arrays(np.asarray(n, dtype=float), np.asarray(d, dtype=float))
    return np.divide(n, d, out=np.zeros_like(n, dtype=float), where=d != 0)


def sbfl_scores(ef, ep, nf, npass):
    """Vectorized, documented 14 SBFL formulas, returns scores and flags."""
    ef = np.asarray(ef, dtype=np.float64)
    ep = np.asarray(ep, dtype=np.float64)
    nf = np.asarray(nf, dtype=np.float64)
    npass = np.asarray(npass, dtype=np.float64)
    F, P = ef + nf, ep + npass
    supp = ef + ep
    nonvisit = nf + npass
    if (supp <= 0).any() or (F <= 0).any() or (P <= 0).any():
        raise ValueError('Invalid contingency spectrum (zero support or empty outcome class)')

    rf = safe_div(ef, F)
    rp = safe_div(ep, P)
    r_covered_fail = safe_div(ef, supp)
    metrics = {}
    metrics['Tarantula'] = safe_div(rf, rf + rp)
    metrics['Barinel'] = r_covered_fail
    metrics['Kulczynski2'] = 0.5 * (rf + r_covered_fail)
    metrics['Wong2'] = ef - ep
    with np.errstate(over='raise', invalid='raise'):
        ochiai2_denom = np.sqrt(supp * nonvisit * F * P)
        metrics['Ochiai2'] = safe_div(ef * npass, ochiai2_denom)
        metrics['Ochiai'] = safe_div(ef, np.sqrt(F * supp))

    # An infinite DStar2 value means ef>0 and (ep+nf)=0. A finite
    # sentinel above the maximum finite value preserves the within-split
    # ranking and is compatible with downstream ROC computation.
    dsden = ep + nf
    dstar_inf = (dsden == 0) & (ef > 0)
    dstar = safe_div(ef ** 2, dsden)
    if dstar_inf.any():
        finite = dstar[~dstar_inf]
        sentinel = float(np.max(finite)) + 1.0 if len(finite) else 1.0
        dstar[dstar_inf] = sentinel
    metrics['DStar2'] = dstar
    metrics['Jaccard'] = safe_div(ef, ef + ep + nf)

    h = np.where(ep <= 2, ep, np.where(ep <= 10, 2 + 0.1*(ep-2),
                                        2.8 + 0.001*(ep-10)))
    metrics['Wong3'] = ef - h
    metrics['GP13'] = ef * (1.0 + safe_div(1, ef + 2*ep))
    metrics['OP2'] = ef - safe_div(ep, P + 1)
    metrics['Wong1'] = ef.copy()
    zden = ef + ep + nf + safe_div(10000.0*ep*nf, ef)
    metrics['Zoltar'] = safe_div(ef, zden)
    metrics['Ample'] = np.abs(rf - rp)

    for name, arr in metrics.items():
        if not np.isfinite(arr).all():
            raise ValueError(f'{name}: nonfinite suspiciousness')
    flags = {'dstar_zero_denominator_positive': dstar_inf,
             'ochiai2_zero_denominator': ochiai2_denom == 0}
    return metrics, flags


def unit_tests():
    # Perfect failed-only voxel: F=2, P=3, all failures cover, no passes.
    s, f = sbfl_scores([2], [0], [0], [3])
    for name in ['Tarantula','Barinel','Ochiai','Ochiai2','Jaccard','Ample','Zoltar']:
        assert np.isclose(s[name][0], 1), (name, s[name][0])
    assert np.isclose(s['Kulczynski2'][0], 1)
    assert np.isclose(s['DStar2'][0], 1) and f['dstar_zero_denominator_positive'][0]
    assert np.isclose(s['GP13'][0], 3)
    assert np.isclose(s['OP2'][0], 2)
    assert np.isclose(s['Wong2'][0], 2)
    # Passing-only voxel: F=2, P=3, no failures cover, all passes cover.
    s, f = sbfl_scores([0], [3], [2], [0])
    for name in ['Tarantula','Barinel','Kulczynski2','Ochiai','Ochiai2',
                 'Jaccard','GP13','Zoltar']:
        assert np.isclose(s[name][0], 0), (name, s[name][0])
    assert np.isclose(s['Ample'][0], 1)
    assert np.isclose(s['Wong2'][0], -3)
    assert np.isclose(s['OP2'][0], -0.75)
    # Voxel universally visited: Ochiai2 = 0/0, documented fallback 0.
    s, f = sbfl_scores([2], [3], [0], [0])
    assert np.isclose(s['Ochiai2'][0], 0) and f['ochiai2_zero_denominator'][0]
    assert set(s) == set(METHODS)


def save_figure(fig, figdir: Path, name: str):
    figdir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(figdir / (name + '.' + ext), dpi=200, bbox_inches='tight')
    plt.close(fig)


def validate_prerequisites(root):
    base = root / STEM / STEP5
    counts = base / 'train_voxel_contingency_counts.csv.gz'
    summary = base / 'training_contingency_split_summary.csv'
    report = base / 'step05_validation_report.json'
    for path in (counts, summary, report):
        if not path.is_file():
            raise ValueError(f'Missing Step 05 prerequisite: {path}')
    meta = json.loads(report.read_text(encoding='utf-8'))
    if meta.get('status') != 'PASS' or meta.get('total_spectra') != 40:
        raise ValueError('Step 05 must have 40 validated PASS spectra')
    if meta.get('test_outcomes_used_to_fit_scores') is not False:
        raise ValueError('Step 05 test-outcome leakage flag is not false')
    df = pd.read_csv(summary)
    required = {'algorithm','seed','training_failures','training_passes',
                'training_covered_voxels','training_episodes','consistency'}
    if not required.issubset(df.columns) or len(df) != 40:
        raise ValueError('Incomplete Step 05 train split summary')
    if df[['algorithm','seed']].duplicated().any() or not df.consistency.eq('PASS').all():
        raise ValueError('Invalid Step 05 train split summary')
    reference = {(str(r.algorithm),int(r.seed)): r for r in df.itertuples(index=False)}
    if set(reference) != {(a,seed) for a in ALGORITHMS for seed in SEEDS}:
        raise ValueError('Step 05 algorithm/seed combinations differ from protocol')
    return counts, summary, report, reference


def plots(diagnostics: pd.DataFrame, seed11: pd.DataFrame, figdir: Path):
    # 1. Fraction tied at the maximum: diagnostic, NOT predictive ranking.
    d = diagnostics[diagnostics.seed == 11]
    ordered = list(METHODS)
    tick_algs = [a.replace('_400scenes', '') for a in ALGORITHMS]
    for field, filename, title in [
        ('max_tie_fraction', '01_fraction_tied_at_max_seed11', 'Fraction of training voxels tied at maximum score'),
        ('fraction_sparse_among_max', '02_sparse_maximum_score_voxels_seed11',
         'Fraction of maximum-scoring voxels with support <=2'),
    ]:
        values = np.array([[float(d[(d.algorithm==a)&(d.method==m)][field].iloc[0])
                            for a in ALGORITHMS] for m in ordered])
        fig, ax = plt.subplots(figsize=(8.0, 6.3))
        img = ax.imshow(values, aspect='auto', vmin=0, vmax=1)
        ax.set_yticks(range(len(ordered)), ordered)
        ax.set_xticks(range(len(ALGORITHMS)), tick_algs, rotation=12)
        ax.set_title('Pi0 400, seed 11: ' + title)
        fig.colorbar(img, ax=ax, label='Fraction (0-1)')
        save_figure(fig, figdir, filename)

    # 3. Ochiai distribution, comparable bounded [0,1] scores.
    fig, ax = plt.subplots(figsize=(9, 4.8))
    bins = np.linspace(0, 1, 31)
    for alg in ALGORITHMS:
        values = seed11.loc[seed11.algorithm == alg, 'Ochiai'].to_numpy(dtype=float)
        ax.hist(values, bins=bins, histtype='step', density=True,
                linewidth=1.6, label=alg.replace('_400scenes', ''))
    ax.set_xlabel('Ochiai suspiciousness (training voxels only)')
    ax.set_ylabel('Density')
    ax.set_title('Pi0 400, seed 11: Ochiai score distributions')
    ax.legend(fontsize=8)
    save_figure(fig, figdir, '03_ochiai_score_distributions_seed11')

    # 4. XY projection of Ochiai scores; Z layers collapse into 2D.
    fig, axs = plt.subplots(2, 2, figsize=(10.2, 9.0), sharex=True, sharey=True)
    pts = None
    for ax, alg in zip(axs.flat, ALGORITHMS):
        data = seed11[seed11.algorithm == alg]
        projected = data.groupby(['voxel_i','voxel_j'], as_index=False)['Ochiai'].max()
        pts = ax.scatter(projected.voxel_i, projected.voxel_j,
                         c=projected.Ochiai, s=4, cmap='viridis', vmin=0, vmax=1,
                         edgecolors='none')
        ax.set_title(alg.replace('_400scenes',''))
        ax.set_xlabel('Voxel index i (X)')
        ax.set_ylabel('Voxel index j (Y)')
    fig.colorbar(pts, ax=axs.ravel().tolist(), shrink=0.78, label='Maximum Ochiai over Z')
    fig.suptitle('Pi0 400, seed 11: 2D visualization of 3D training suspiciousness')
    for ext in ('png', 'pdf'):
        fig.savefig(figdir / f'04_ochiai_xy_projection_seed11.{ext}',
                    dpi=200, bbox_inches='tight')
    plt.close(fig)


def run(root: Path):
    unit_tests()
    counts_path, summary_path, step5_report, reference = validate_prerequisites(root)
    out = root / STEM / STEP6
    out.mkdir(parents=True, exist_ok=True)
    figdir = out / 'figures'
    diagrows, top_rows, seed11s = [], [], []
    completed = []
    dstar_special, ochiai2_special, voxel_total = 0, 0, 0
    outcounts = out / 'voxel_suspiciousness_scores.csv.gz'
    cols = ['algorithm','seed','voxel_i','voxel_j','voxel_k',
            'e_f','e_p','n_f','n_p','support'] + list(METHODS)

    with gzip.open(counts_path, 'rt', newline='', encoding='utf-8-sig') as inp, \
         gzip.open(outcounts, 'wt', newline='', encoding='utf-8', compresslevel=6) as fout:
        reader = csv.DictReader(inp)
        req = {'algorithm','seed','voxel_i','voxel_j','voxel_k','e_f','e_p','n_f','n_p',
               'support','training_total_failures','training_total_passes'}
        if not req.issubset(reader.fieldnames or ()):
            raise ValueError(f'Missing Step05 count columns: {req - set(reader.fieldnames or ())}')
        writer = csv.writer(fout)
        writer.writerow(cols)
        for (alg, seedstr), rows in itertools.groupby(reader, key=lambda r:(r['algorithm'],r['seed'])):
            if not seedstr.isdecimal():
                raise ValueError('Invalid seed in Step05 counts')
            seed = int(seedstr)
            key = (alg, seed)
            expected_key = [(a, s) for a in ALGORITHMS for s in SEEDS][len(completed)] if len(completed) < 40 else None
            if key != expected_key:
                raise ValueError(f'Unexpected count block order: got {key}, expected {expected_key}')
            block = pd.DataFrame(list(rows))
            n = len(block)
            ref = reference[key]
            if n != int(ref.training_covered_voxels):
                raise ValueError(f'{key}: count rows {n} != {ref.training_covered_voxels}')
            for field in ('voxel_i','voxel_j','voxel_k','e_f','e_p','n_f','n_p',
                          'support','training_total_failures','training_total_passes'):
                block[field] = pd.to_numeric(block[field], errors='raise').astype('int64')
            if block[['voxel_i','voxel_j','voxel_k']].duplicated().any():
                raise ValueError(f'{key}: duplicated voxel indices')
            ef = block.e_f.to_numpy()
            ep = block.e_p.to_numpy()
            nf = block.n_f.to_numpy()
            npass = block.n_p.to_numpy()
            supp = block.support.to_numpy()
            F, P = int(ref.training_failures), int(ref.training_passes)
            if (np.minimum.reduce([ef, ep, nf, npass]) < 0).any() or (supp != ef+ep).any():
                raise ValueError(f'{key}: invalid per-voxel counts')
            if not ((ef+nf == F).all() and (ep+npass == P).all()):
                raise ValueError(f'{key}: inconsistent train totals')
            if not ((block.training_total_failures == F).all()
                    and (block.training_total_passes == P).all()
                    and F+P == int(ref.training_episodes)):
                raise ValueError(f'{key}: Step05 training totals differ')

            scores, flags = sbfl_scores(ef, ep, nf, npass)
            dstar_special += int(flags['dstar_zero_denominator_positive'].sum())
            ochiai2_special += int(flags['ochiai2_zero_denominator'].sum())
            voxel_total += n
            numeric = [block[col].to_numpy() for col in
                       ('voxel_i','voxel_j','voxel_k','e_f','e_p','n_f','n_p','support')]
            for i in range(n):
                writer.writerow([alg,seed] + [int(v[i]) for v in numeric]
                                + [format(float(scores[m][i]), '.15g') for m in METHODS])

            if seed == 11:
                sample = block[['voxel_i','voxel_j','voxel_k','support']].copy()
                sample.insert(0, 'algorithm', alg)
                for m in ('Ochiai',):
                    sample[m] = scores[m]
                seed11s.append(sample)
            for method in METHODS:
                v = scores[method]
                best = float(v.max())
                atbest = np.isclose(v, best, rtol=0, atol=1e-12)
                tied_n = int(atbest.sum())
                sparse_best = int(((supp <= 2) & atbest).sum())
                diagrows.append({
                    'algorithm':alg, 'seed':seed, 'method':method,
                    'training_voxels':n,
                    'score_min':float(v.min()), 'score_median':float(np.median(v)),
                    'score_max':best, 'score_mean':float(v.mean()),
                    'zero_scores':int((v==0).sum()),
                    'negative_scores':int((v<0).sum()),
                    'max_tied_voxels':tied_n,
                    'max_tie_fraction':tied_n/n,
                    'sparse_max_voxels_support_le_2':sparse_best,
                    'fraction_sparse_among_max':sparse_best/tied_n,
                    'median_support_among_max':float(np.median(supp[atbest])),
                    'dstar_positive_over_zero_count':int(flags['dstar_zero_denominator_positive'].sum()) if method=='DStar2' else 0,
                    'ochiai2_zero_over_zero_count':int(flags['ochiai2_zero_denominator'].sum()) if method=='Ochiai2' else 0,
                })
                if seed == 11:
                    # Descending score, then descending support, then coordinates for deterministic ties.
                    ix = np.lexsort((block.voxel_k.to_numpy(),
                                     block.voxel_j.to_numpy(),
                                     block.voxel_i.to_numpy(), -supp, -v))[:10]
                    for rank, i in enumerate(ix, 1):
                        top_rows.append({
                            'algorithm':alg,'seed':seed,'method':method,'rank':rank,
                            'voxel_i':int(block.voxel_i.iloc[i]),
                            'voxel_j':int(block.voxel_j.iloc[i]),
                            'voxel_k':int(block.voxel_k.iloc[i]),
                            'e_f':int(ef[i]),'e_p':int(ep[i]),
                            'n_f':int(nf[i]),'n_p':int(npass[i]),
                            'support':int(supp[i]), 'score':float(v[i]),
                        })
            completed.append(key)
            print(f'[SCORES] {alg} seed={seed}: voxels={n}, methods=14; '
                  f'DStar2 zero-den={int(flags["dstar_zero_denominator_positive"].sum())}, '
                  f'Ochiai2 0/0={int(flags["ochiai2_zero_denominator"].sum())}')

    if len(completed) != 40:
        raise ValueError(f'Expected 40 count spectra, got {len(completed)}')
    diagnostics = pd.DataFrame(diagrows)
    if len(diagnostics) != 40*14:
        raise ValueError('Did not calculate all 560 method/split diagnostics')
    diagnostics.to_csv(out/'per_method_split_diagnostics.csv', index=False)
    pd.DataFrame(top_rows).to_csv(out/'top_voxels_seed11.csv', index=False)
    pd.DataFrame([{'method':m, 'formula':FORMULAS[m],
                  'zero_denominator_policy':
                  '0.0 for undefined 0/0; DStar2 positive/0 encoded max finite+1 (within split)' }
                 for m in METHODS]).to_csv(out/'formula_definitions.csv', index=False)
    plots(diagnostics, pd.concat(seed11s, ignore_index=True), figdir)
    meta = {
        'status':'PASS', 'model':'Pi0', 'scene_budget':400,
        'algorithms':list(ALGORITHMS), 'seeds':list(SEEDS),
        'total_training_spectra':len(completed), 'total_method_spectra':len(diagnostics),
        'total_voxel_rows_across_splits':voxel_total,
        'methods':list(METHODS),
        'input_training_counts_sha256':sha256(counts_path),
        'input_training_summary_sha256':sha256(summary_path),
        'input_step05_report_sha256':sha256(step5_report),
        'scores_file':'voxel_suspiciousness_scores.csv.gz',
        'scores_format':'one row per (algorithm, seed, training-covered voxel); 14 wide suspiciousness columns',
        'test_outcomes_used_to_fit_scores':False,
        'dstar2_positive_over_zero_encoded_as_max_finite_plus_one':dstar_special,
        'ochiai2_zero_over_zero_encoded_as_zero':ochiai2_special,
        'unseen_test_voxel_policy':'Not scored here. Step07: ignore unseen voxels in per-episode max; if none known, use below minimum known score per method/split.',
        'comparison_warning':'Raw values are NOT comparable across methods, and maxima/ties are NOT held-out predictive performance.',
        'scene_grouping_limitation':'Current grouped splits rely on candidate scene identity rather than complete independent simulator confirmation.',
    }
    (out/'step06_validation_report.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print('\nRQ1 STEP 06 - 14 SBFL SUSPICIOUSNESS MAPS')
    print(f'Training spectra: {len(completed)}/40 | Method maps: {len(diagnostics)}/560 | Status: PASS')
    print(f'Voxel rows (across 40 splits): {voxel_total}')
    print(f'DStar2 positive/0 instances: {dstar_special}; Ochiai2 0/0 instances: {ochiai2_special}')
    print('Test labels used for suspiciousness: NO')
    print('Files saved to:', out)
    print('Next: Step 07 score the SAME held-out test episodes using max known voxel suspiciousness.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path,
                        default=Path(__file__).resolve().parent.parent,
                        help='Root of the SBFL-only repo containing input/src/output')
    args = parser.parse_args()
    try:
        run(args.repo_root.resolve())
    except (OSError, ValueError, KeyError, TypeError, AssertionError, IndexError) as err:
        print('BLOCKED:', err, file=sys.stderr)
        return 1
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
