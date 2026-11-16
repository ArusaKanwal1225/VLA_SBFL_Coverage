"""Read the frozen RQ1 outcome audit for the final RQ5 selection."""
import csv
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AUDIT = ROOT / 'data/reference/rq1/audit/episode_manifest.csv.gz'
DISPLAY = {'EO1': 'EO1', 'Groot': 'GR00T', 'Pi0': 'Pi0', 'SpatialVLA': 'SpatialVLA'}
CONDITIONS = ('DH', 'DH_HybridDT', 'DH_HybridRF', 'DH_WeightedRF')

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
