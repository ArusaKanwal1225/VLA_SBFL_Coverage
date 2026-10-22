"""Shared RQ1 definitions. Python 3.9+; ASCII-only source for Windows.

The experiment is limited to Pi0 and 200-scene runs. The grouped scene key
is a candidate key inferred from object_name, orientation_id, x, y.
Metrics are held-out failure discrimination, not ground-truth fault location.
"""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ALGORITHMS = (
    'DH_200scenes', 'DH_HybridDT_200scenes',
    'DH_HybridRF_200scenes', 'DH_WeightedRF_200scenes',
)
SEEDS = (11, 23, 37, 41, 53, 67, 79, 83, 97, 101)
METHODS = (
    'Tarantula', 'Barinel', 'Kulczynski2', 'Wong2', 'Ochiai2',
    'Ochiai', 'DStar2', 'Jaccard', 'Wong3', 'GP13', 'OP2',
    'Wong1', 'Zoltar', 'Ample',
)
ORIGIN = (-1.0, -1.0, 0.0)
VOXEL_SIZE = 0.02
GROUP_KEY = 'scene_group_exact_candidate'
REL_OUTPUT = Path('output') / 'RQ1' / 'pi0_200' / 'consolidated'

FORMULAS = {
    'Tarantula': '(ef/F) / ((ef/F) + (ep/P))',
    'Barinel': 'ef / (ef+ep)',
    'Kulczynski2': '0.5 * (ef/F + ef/(ef+ep))',
    'Wong2': 'ef-ep',
    'Ochiai2': '(ef*np) / sqrt((ef+ep)*(nf+np)*F*P)',
    'Ochiai': 'ef / sqrt(F*(ef+ep))',
    'DStar2': 'ef**2 / (ep+nf), rank-preserving finite sentinel for +infinity',
    'Jaccard': 'ef / (ef+ep+nf)',
    'Wong3': 'ef - h(ep), piecewise h',
    'GP13': 'ef * (1+1/(ef+2*ep))',
    'OP2': 'ef - ep/(P+1)',
    'Wong1': 'ef',
    'Zoltar': 'ef / (ef+ep+nf+10000*ep*nf/ef)',
    'Ample': 'abs(ef/F-ep/P)',
}


def require(condition, description):
    if not condition:
        raise ValueError(description)


def safe_div(numerator, denominator):
    n, d = np.broadcast_arrays(np.asarray(numerator, dtype=float),
                               np.asarray(denominator, dtype=float))
    return np.divide(n, d, out=np.zeros_like(n, dtype=float), where=d != 0)


def sbfl_scores(ef, ep, nf, npass):
    ef, ep, nf, npass = [np.asarray(x, dtype=np.float64)
                          for x in (ef, ep, nf, npass)]
    F, P = ef + nf, ep + npass
    support = ef + ep
    require(np.all(support > 0) and np.all(F > 0) and np.all(P > 0),
            'Each voxel needs positive support and both training classes')
    rf, rp = safe_div(ef, F), safe_div(ep, P)
    fail_frac = safe_div(ef, support)
    h = np.where(ep <= 2, ep,
                 np.where(ep <= 10, 2 + .1 * (ep - 2), 2.8 + .001 * (ep - 10)))
    dden = ep + nf
    dstar = safe_div(ef**2, dden)
    inf_mask = (dden == 0) & (ef > 0)
    if inf_mask.any():
        finite = dstar[~inf_mask]
        dstar[inf_mask] = (float(np.max(finite)) + 1.0) if len(finite) else 1.0
    od = np.sqrt(support * (nf+npass) * F * P)
    result = {
        'Tarantula': safe_div(rf, rf+rp),
        'Barinel': fail_frac,
        'Kulczynski2': 0.5 * (rf+fail_frac),
        'Wong2': ef-ep,
        'Ochiai2': safe_div(ef*npass, od),
        'Ochiai': safe_div(ef, np.sqrt(F * support)),
        'DStar2': dstar,
        'Jaccard': safe_div(ef, ef+ep+nf),
        'Wong3': ef-h,
        'GP13': ef*(1 + safe_div(1, ef+2*ep)),
        'OP2': ef-safe_div(ep, P+1),
        'Wong1': ef.copy(),
        'Zoltar': safe_div(ef, ef+ep+nf+safe_div(10000*ep*nf, ef)),
        'Ample': np.abs(rf-rp),
    }
    require(set(result) == set(METHODS), 'Formula names do not match protocol')
    for name, scores in result.items():
        require(np.isfinite(scores).all(), 'Nonfinite scores for ' + name)
    return result


def test_formulas():
    high = sbfl_scores(np.array([2]),np.array([0]),np.array([0]),np.array([3]))
    require(np.isclose(high['Zoltar'][0], 1.0), 'Zoltar unit test failed')
    require(np.isclose(high['Ochiai'][0], 1.0), 'Ochiai unit test failed')
    low = sbfl_scores(np.array([0]),np.array([3]),np.array([2]),np.array([0]))
    require(np.isclose(low['Zoltar'][0], 0.0), 'Zoltar zero-failure unit test failed')


def out_csv(frame, path, **kwargs):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding='utf-8', **kwargs)


def out_json(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str),
                    encoding='utf-8')


def fig_save(fig, directory, name):
    directory.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(directory / (name + '.' + ext), dpi=180, bbox_inches='tight')
    plt.close(fig)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def verify_40_splits(manifest, episodes):
    required = {'algorithm','seed','episode_id','failure','run_id','split',GROUP_KEY}
    require(required.issubset(manifest.columns), 'Split manifest columns missing')
    require(len(manifest) == 80000 and
            not manifest[['algorithm','seed','episode_id']].duplicated().any(),
            'Expected 80,000 unique episode/split records')
    require(set(manifest.algorithm) == set(ALGORITHMS), 'Wrong algorithms in split manifest')
    require(set(manifest.seed.astype(int)) == set(SEEDS), 'Wrong seeds in split manifest')
    require(set(manifest.split) == {'train','test'}, 'Expected train and test labels')
    require(set(manifest.failure.astype(str)) <= {'0','1'}, 'Bad failure label')
    ep_index = episodes.set_index(['algorithm','episode_id'])[['failure','run_id']]
    for alg in ALGORITHMS:
        expected = episodes.loc[episodes.algorithm == alg, 'episode_id']
        require(len(expected) == 2000 and expected.nunique() == 2000,
                'Input episode count wrong for ' + alg)
        for seed in SEEDS:
            s = manifest[(manifest.algorithm == alg) & (manifest.seed == seed)]
            require(len(s) == 2000 and set(s.episode_id) == set(expected),
                    f'Episode mismatch for {alg}, seed {seed}')
            train = s[s.split == 'train']; test = s[s.split == 'test']
            require(580 <= len(test) <= 620 and train.failure.nunique() == 2
                    and test.failure.nunique() == 2, f'Bad split class/count: {alg}, {seed}')
            require(not set(train[GROUP_KEY]) & set(test[GROUP_KEY]),
                    f'Scene group leakage for {alg}, {seed}')
            matching = ep_index.loc[pd.MultiIndex.from_arrays([s.algorithm,s.episode_id])]
            require(np.array_equal(matching.failure.astype(int).to_numpy(),
                                   s.failure.astype(int).to_numpy()) and
                    np.array_equal(matching.run_id.to_numpy(),s.run_id.to_numpy()),
                    f'Split labels/run IDs mismatch: {alg}, {seed}')


def get_root(script, user_root):
    return Path(user_root).expanduser().resolve() if user_root else Path(script).resolve().parent.parent
