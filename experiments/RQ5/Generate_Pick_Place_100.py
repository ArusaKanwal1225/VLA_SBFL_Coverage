"""Generate 100 balanced pilot scenes; does not run a VLA model.
Requires matplotlib. Initial mesh validity and pick/place success need simulator checks.
"""
import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

NAMES = ['redbull can', 'blue plastic bottle', 'eggplant', 'carrot', 'apple', 'spoon', 'sponge']
MODELS = ['opened_redbull_can', 'blue_plastic_bottle', 'eggplant',
          'bridge_carrot_generated_modified', 'apple', 'bridge_spoon_generated_modified', 'sponge']
ORIENTATIONS = {1: [0.707, 0.707, 0, 0], 2: [0.707, 0, 0, 0.707],
                3: [0.707, 0, 0.707, 0], 4: [0.98, 0, 0, 0.20], 5: [0.20, 0, 0, 0.98]}
X_RANGE, Y_RANGE, Z_RANGE = (-0.5, -0.05), (0, 0.4), (0, 0.5)
VOX_X, VOX_Y, VOX_Z = (-0.65, 0.05), (-0.15, 0.65), (0.80, 1.15)
VOXEL_SIZE = 0.05


def schedule(rng):
    # 105 entries minus five: each orientation 20, each object 14-15.
    omitted = set(zip(rng.sample(range(7), 5), range(1, 6)))
    items = [(o, q) for o in range(7) for q in range(1, 6)
             for _ in range(3 - ((o, q) in omitted))]
    rng.shuffle(items)
    return items


def pair_targets(source, target, rng):
    edges = [[j for j, t in enumerate(target) if s[0] != t[0]] for s in source]
    for options in edges:
        rng.shuffle(options)
    owner = {}
    def assign(i, seen):
        for j in edges[i]:
            if j not in seen:
                seen.add(j)
                if j not in owner or assign(owner[j], seen):
                    owner[j] = i
                    return True
        return False
    for i in range(100):
        if not assign(i, set()):
            raise RuntimeError('Cannot pair distinct objects.')
    return [target[j] for j, i in sorted(owner.items(), key=lambda item: item[1])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=20261026)
    parser.add_argument('--output-dir', type=Path, default=Path('output/Pick_and_place_100'))
    parser.add_argument('--min-separation', type=float, default=0.20)
    parser.add_argument('--preserve-raw-quaternions', action='store_true')
    args = parser.parse_args()
    if not 0 < args.min_separation < math.hypot(0.45, 0.4):
        parser.error('Separation must be positive and below the workspace diagonal.')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out, rng = args.output_dir, random.Random(args.seed)
    (out/'figures').mkdir(parents=True, exist_ok=True)
    source = schedule(rng)
    target = pair_targets(source, schedule(rng), rng)
    scenes, references = {}, []
    for i, ((s, sq), (t, tq)) in enumerate(zip(source, target)):
        for _ in range(10000):
            a, b = [[round(rng.uniform(*X_RANGE), 5), round(rng.uniform(*Y_RANGE), 5)] for _ in range(2)]
            if math.dist(a, b) >= args.min_separation:
                break
        else:
            raise RuntimeError('Position sampling failed; reduce --min-separation.')
        options = {}
        for obj, ori, xy in [(s, sq, a), (t, tq, b)]:
            q = ORIENTATIONS[ori]
            norm = 1 if args.preserve_raw_quaternions else math.sqrt(sum(v*v for v in q))
            options[MODELS[obj]] = dict(init_xy=xy, init_rot_quat=[v/norm for v in q])
        scenes[str(i)] = dict(model_ids=[MODELS[s], MODELS[t]], obj_init_options=options,
                              source_obj_id=0, target_obj_id=1)
        references.append(f'move {NAMES[s]} near {NAMES[t]}')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    for ax, role, design in zip(axes, ['source', 'target'], [source, target]):
        counts = Counter(design)
        assert sorted(Counter(o for o, q in design).values()) == [14]*5 + [15]*2
        assert sorted(Counter(q for o, q in design).values()) == [20]*5
        assert len(counts) == 35 and set(counts.values()) == {2, 3}
        ax.imshow([[counts[o,q] for q in range(1,6)] for o in range(7)], vmin=0, vmax=3, cmap='Blues', aspect='auto')
        for o in range(7):
            for q in range(1,6):
                ax.text(q-1, o, str(counts[o,q]), ha='center', va='center')
        ax.set(xticks=range(5), xticklabels=range(1,6), yticks=range(7), yticklabels=NAMES,
               xlabel='Orientation ID', title=f'{role.capitalize()} object: 100 scenes')
    fig.savefig(out/'figures/object_orientation_balance.png', dpi=200)
    plt.close(fig)
    assert len(scenes) == len(references) == 100
    assert all(s[0] != t[0] for s, t in zip(source, target))
    for name, data in [('t-move_near_100', scenes), ('move_near_prompts', references)]:
        (out/f'{name}.json').write_text(json.dumps(data, indent=2)+'\n', encoding='utf-8')
    print(f'100 balanced scene specifications saved to {out.resolve()}. No VLA executions performed.')


if __name__ == '__main__':
    main()
