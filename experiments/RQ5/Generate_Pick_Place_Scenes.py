#!/usr/bin/env python3
"""Generate 504 balanced move-near scenes in 72 target-position cells.

This generates scene specifications only, not simulator-validated executions.
"""
import argparse
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path

from Generate_Pick_Place_100 import NAMES, MODELS, ORIENTATIONS, X_RANGE, Y_RANGE, VOXEL_SIZE

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PILOT = ROOT / 'output/Pick_and_place_100/t-move_near_100.json'
DEFAULT_OUTPUT = ROOT / 'output/Pick_and_place_504'
GRID_X, GRID_Y = 9, 8
SCENE_COUNT = GRID_X * GRID_Y * len(MODELS)
SCALE = 100000  # retain the pilot's five decimal places, with exact cell boundaries


def scene_key(scene):
    names = scene['model_ids']
    source, target = names[scene['source_obj_id']], names[scene['target_obj_id']]
    options = scene['obj_init_options']
    # Ignore rotations here, making overlap checks conservative: same objects
    # and initial XY positions count as the same scene even if poses differ.
    return (source, target, tuple(options[source]['init_xy']), tuple(options[target]['init_xy']))


def normalized_quaternion(orientation_id):
    q = ORIENTATIONS[orientation_id]
    norm = math.sqrt(sum(v * v for v in q))
    return [v / norm for v in q]


def object_orientations(rng):
    """Each object gets 14/15 of each pose; globally poses occur 100/101 times."""
    result = {}
    for obj in range(len(MODELS)):
        extras = (2 * obj % 5, (2 * obj + 1) % 5)
        values = [ori + 1 for ori in range(5) for _ in range(14 + (ori in extras))]
        rng.shuffle(values)
        result[obj] = values
    return result


def sample_xy(rng, xmin, xmax, ymin, ymax):
    # Integer sampling avoids coordinates rounding across a cell boundary.
    return [rng.randrange(xmin, xmax) / SCALE, rng.randrange(ymin, ymax) / SCALE]


def generate(seed, separation, pilot_scenes):
    rng = random.Random(seed)
    source_rotations = object_orientations(rng)
    target_rotations = object_orientations(rng)
    source_seen, target_seen = Counter(), Counter()
    cells = [(x, y) for x in range(GRID_X) for y in range(GRID_Y)]
    rng.shuffle(cells)
    # Every directed source-target pair occurs exactly 12 times. Each cell has
    # all seven source and target objects exactly once (cyclic derangements).
    offsets = [offset for offset in range(1, 7) for _ in range(12)]
    rng.shuffle(offsets)
    pilot_keys = {scene_key(scene) for scene in pilot_scenes.values()}
    used_keys = set()
    scenes = []
    x_min, x_max = (round(x * SCALE) for x in X_RANGE)
    y_min, y_max = (round(y * SCALE) for y in Y_RANGE)
    cell_size = round(VOXEL_SIZE * SCALE)
    for (ix, iy), offset in zip(cells, offsets):
        sources = list(range(len(MODELS)))
        rng.shuffle(sources)
        for source in sources:
            target = (source + offset) % len(MODELS)
            source_ori = source_rotations[source][source_seen[source]]
            target_ori = target_rotations[target][target_seen[target]]
            for _ in range(10000):
                source_xy = sample_xy(rng, x_min, x_max, y_min, y_max)
                target_xy = sample_xy(rng, x_min + ix * cell_size, x_min + (ix + 1) * cell_size,
                                      y_min + iy * cell_size, y_min + (iy + 1) * cell_size)
                if math.dist(source_xy, target_xy) < separation:
                    continue
                source_name, target_name = MODELS[source], MODELS[target]
                key = (source_name, target_name, tuple(source_xy), tuple(target_xy))
                if key not in pilot_keys and key not in used_keys:
                    break
            else:
                raise RuntimeError(f'Could not sample a valid non-pilot scene in cell {(ix, iy)}')
            used_keys.add(key)
            source_seen[source] += 1
            target_seen[target] += 1
            scene = {'model_ids': [source_name, target_name],
                     'obj_init_options': {
                         source_name: {'init_xy': source_xy, 'init_rot_quat': normalized_quaternion(source_ori)},
                         target_name: {'init_xy': target_xy, 'init_rot_quat': normalized_quaternion(target_ori)}},
                     'source_obj_id': 0, 'target_obj_id': 1}
            scenes.append((scene, f'move {NAMES[source]} near {NAMES[target]}',
                           {'cell_x': ix, 'cell_y': iy, 'source_orientation_id': source_ori,
                            'target_orientation_id': target_ori}))
    rng.shuffle(scenes)
    return scenes


def audit(scenes, pilot_scenes, separation):
    if len(scenes) != SCENE_COUNT:
        raise ValueError('Incorrect scene count')
    cell_counts, source_counts, target_counts = Counter(), Counter(), Counter()
    source_poses, target_poses, pairs = Counter(), Counter(), Counter()
    keys = set()
    pilot_keys = {scene_key(scene) for scene in pilot_scenes.values()}
    for scene, prompt, metadata in scenes:
        key = scene_key(scene)
        if key in keys or key in pilot_keys:
            raise ValueError('Duplicate or pilot scene')
        keys.add(key)
        source, target, source_xy, target_xy = key
        if source == target or math.dist(source_xy, target_xy) < separation:
            raise ValueError('Invalid pair or separation')
        if not all(lo <= value < hi for value, (lo, hi) in zip(source_xy, (X_RANGE, Y_RANGE))):
            raise ValueError('Source outside workspace')
        if not all(lo <= value < hi for value, (lo, hi) in zip(target_xy, (X_RANGE, Y_RANGE))):
            raise ValueError('Target outside workspace')
        ix, iy = metadata['cell_x'], metadata['cell_y']
        # Check boundaries in integer space, not binary floating-point division.
        x, y = (round(value * SCALE) for value in target_xy)
        if not (round(X_RANGE[0] * SCALE) + ix * 5000 <= x < round(X_RANGE[0] * SCALE) + (ix + 1) * 5000
                and iy * 5000 <= y < (iy + 1) * 5000):
            raise ValueError('Target outside assigned cell')
        cell_counts[(ix, iy)] += 1
        source_counts[source] += 1; target_counts[target] += 1
        source_poses[(source, metadata['source_orientation_id'])] += 1
        target_poses[(target, metadata['target_orientation_id'])] += 1
        pairs[(source, target)] += 1
    if len(cell_counts) != 72 or set(cell_counts.values()) != {7}:
        raise ValueError('Cells are not balanced')
    if set(source_counts.values()) != {72} or set(target_counts.values()) != {72}:
        raise ValueError('Object counts are not balanced')
    if len(pairs) != 42 or set(pairs.values()) != {12}:
        raise ValueError('Object pairs are not balanced')
    if set(source_poses.values()) != {14, 15} or set(target_poses.values()) != {14, 15}:
        raise ValueError('Object orientations are not balanced')
    return {'scenes': len(scenes), 'target_cells': len(cell_counts), 'scenes_per_cell': 7,
            'source_counts': dict(source_counts), 'target_counts': dict(target_counts),
            'source_orientation_counts': {str(i): sum(n for (obj, ori), n in source_poses.items() if ori == i) for i in ORIENTATIONS},
            'target_orientation_counts': {str(i): sum(n for (obj, ori), n in target_poses.items() if ori == i) for i in ORIENTATIONS},
            'ordered_object_pairs': len(pairs), 'scenes_per_ordered_pair': 12,
            'pilot_scene_overlap': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=20261028)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--pilot', type=Path, default=DEFAULT_PILOT)
    parser.add_argument('--min-separation', type=float, default=0.20)
    args = parser.parse_args()
    if not math.isfinite(args.min_separation) or not 0 < args.min_separation < math.hypot(0.45, 0.4):
        parser.error('min-separation must be positive and below the workspace diagonal')
    try:
        raw_pilot = args.pilot.read_bytes()
        pilot = json.loads(raw_pilot)
        if not isinstance(pilot, dict) or set(pilot) != {str(i) for i in range(100)}:
            raise ValueError('Pilot must contain scene IDs 0 through 99')
        scenes = generate(args.seed, args.min_separation, pilot)
        report = audit(scenes, pilot, args.min_separation)
        report.update({'seed': args.seed, 'pilot_sha256': hashlib.sha256(raw_pilot).hexdigest(),
                       'min_separation': args.min_separation, 'voxel_size_xy': VOXEL_SIZE,
                       'x_range': X_RANGE, 'y_range': Y_RANGE, 'grid_shape': [GRID_X, GRID_Y],
                       'note': 'Initial XY positions only; reachability, collisions and success require simulator validation.'})
        out = args.output_dir
        names = ['t-move_near_504.json', 'move_near_prompts_504.json', 'scene_metadata_504.json', 'generation_report.json']
        existing = [str(out / name) for name in names if (out / name).exists()]
        if existing:
            raise FileExistsError('Output files already exist: ' + ', '.join(existing))
        out.mkdir(parents=True, exist_ok=True)
        payloads = [{str(i): scene for i, (scene, _, _) in enumerate(scenes)},
                    [prompt for _, prompt, _ in scenes],
                    {str(i): meta for i, (_, _, meta) in enumerate(scenes)}, report]
        for name, data in zip(names, payloads):
            (out / name).write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    except (OSError, ValueError, KeyError, IndexError, TypeError, RuntimeError) as exc:
        parser.exit(1, f'Error: {exc}\n')
    print(json.dumps(report, indent=2))
    print(f'Generated {SCENE_COUNT} scenes in {out.resolve()}')


if __name__ == '__main__':
    main()
