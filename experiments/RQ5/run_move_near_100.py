#!/usr/bin/env python3
"""Evaluate the 100 fixed move-near scenes with one VLA per process."""
import argparse
import csv
import math
import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
COVERAGE_ROOT = HERE.parents[1]
DEEPHYPERION_ROOT = Path(os.environ.get('DEEPHYPERION_ROOT', '/home/ubuntu/Deephyperion_VLA')).resolve()
DEFAULT_DATA = COVERAGE_ROOT / 'output' / 'Pick_and_place_100'
MODELS = ('pi0', 'gr00t', 'eo1', 'spatialvla-4b')
METHODS = ('position_instability', 'velocity_instability', 'acceleration_instability')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--tasks', type=Path, default=DEFAULT_DATA / 't-move_near_100.json')
    parser.add_argument('--prompts', type=Path, default=DEFAULT_DATA / 'move_near_prompts.json')
    parser.add_argument('--output', type=Path, default=DEEPHYPERION_ROOT / 'results' / 'move_near_100')
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--resume', action='store_true', help='Skip completed cases from the same configuration')
    parser.add_argument('--video', action='store_true', help='Save an MP4 for each episode')
    return parser.parse_args()


def load_inputs(args):
    tasks = json.loads(args.tasks.read_text(encoding='utf-8'))
    prompts = json.loads(args.prompts.read_text(encoding='utf-8'))
    if not isinstance(tasks, dict) or set(tasks) != {str(i) for i in range(100)}:
        raise ValueError('The task JSON must contain exactly IDs 0 through 99')
    if not isinstance(prompts, list) or len(prompts) != 100 or not all(isinstance(p, str) and p.strip() for p in prompts):
        raise ValueError('The prompts JSON must contain exactly 100 nonempty strings')
    for idx in range(100):
        case = tasks[str(idx)]
        if not isinstance(case, dict) or not {'model_ids', 'obj_init_options', 'source_obj_id', 'target_obj_id'} <= case.keys():
            raise ValueError('Invalid task configuration at ID %s' % idx)
    return tasks, prompts


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def native(value):
    import numpy as np
    if isinstance(value, np.ndarray):
        return native(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(v) for v in value]
    return value


def write_json(path, value):
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', encoding='utf-8') as handle:
        json.dump(native(value), handle, indent=2, allow_nan=False)
    os.replace(temp, path)


class PromptPolicy:
    """Keep the JSON prompt in reset and in every inference step."""
    def __init__(self, policy):
        self.policy = policy
        self.prompt = None

    def reset(self, prompt):
        self.prompt = prompt
        return self.policy.reset(prompt)

    def step(self, image, *args, **kwargs):
        if args:
            args = (self.prompt,) + args[1:]
        return self.policy.step(image, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.policy, name)


def write_csv(path, fields, rows):
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def export_csvs(result_dir, model, tasks, prompts):
    # Reuse the same trajectory coverage, feature bins and proxy as DeepHyperion.
    from DH_Hybrid_DT import (ORIENTATIONS, OBJECTS, NUM_VOXELS, NUM_CONES,
                              trajectory_coverage, feature_bin, compute_proxy)
    from types import SimpleNamespace
    dh_fields = ['eval_id', 'x', 'y', 'orientation_id', 'object_name', 'feat_x', 'feat_y',
                 'cell_x', 'cell_y', 'failure', 'proxy', 'new_coverage', 'surrogate_pf',
                 'surrogate_pfail', 'surrogate_ready', 'filled_cells', 'mapped_misbehaviours',
                 'spatial_coverage', 'directional_coverage', 'target_object_name', 'target_x',
                 'target_y', 'prompt']
    episode_fields = ['model', 'algorithm', 'budget', 'run_id', 'eval_id', 'episode_id',
                      'failure', 'object_name', 'num_steps']
    object_names = {value: key for key, value in OBJECTS.items()}
    dh_rows, episode_rows = [], []
    covered_voxels, covered_cones = set(), set()
    for i in range(100):
        folder = result_dir / 'allMetrics' / str(i)
        log_path, tcp_path = folder / 'log.json', folder / 'tcp_poses.json'
        if not log_path.is_file():
            continue
        stats = json.loads(log_path.read_text(encoding='utf-8'))
        poses = json.loads(tcp_path.read_text(encoding='utf-8'))
        if not isinstance(stats, dict) or not isinstance(poses, list):
            raise ValueError('Invalid episode files for %s' % i)
        last = stats[max(stats, key=lambda key: int(key))] if stats else {}
        success = last.get('success')
        if isinstance(success, str):
            success = {'true': True, 'false': False}.get(success.strip().lower())
        if not isinstance(success, bool):
            raise ValueError('Missing or invalid final success in %s: %r' % (log_path, last.get('success')))
        failure = int(not success)
        task = tasks[str(i)]
        source = task['model_ids'][task['source_obj_id']]
        target = task['model_ids'][task['target_obj_id']]
        source_opt, target_opt = task['obj_init_options'][source], task['obj_init_options'][target]
        x, y = source_opt['init_xy']; target_x, target_y = target_opt['init_xy']
        quat = source_opt.get('init_rot_quat')
        orientation_id = ''
        if quat is not None:
            distances = [(min(sum((a-b)**2 for a,b in zip(quat,q)),
                              sum((a+b)**2 for a,b in zip(quat,q))), idx)
                         for idx, q in ORIENTATIONS.items()]
            distance, match = min(distances)
            if math.sqrt(distance) < 0.005:
                orientation_id = match
        trajectory = [pose[:3] for pose in poses]
        voxels, cones = trajectory_coverage(trajectory)
        new_coverage = len(voxels - covered_voxels) + len(cones - covered_cones)
        covered_voxels.update(voxels); covered_cones.update(cones)
        object_name = object_names.get(source, source)
        episode_id = '%s_fixed_move_near_100_run_1_%d' % (model, i)
        dh_rows.append({'eval_id': i, 'x': x, 'y': y, 'orientation_id': orientation_id,
                        'object_name': object_name, 'feat_x': x, 'feat_y': y,
                        'cell_x': feature_bin('obj_x', x), 'cell_y': feature_bin('obj_y', y),
                        'failure': failure, 'proxy': compute_proxy(SimpleNamespace(x=x,y=y), trajectory),
                        'new_coverage': new_coverage,
                        'spatial_coverage': len(covered_voxels) / NUM_VOXELS,
                        'directional_coverage': len(covered_cones) / NUM_CONES,
                        'target_object_name': object_names.get(target, target),
                        'target_x': target_x, 'target_y': target_y, 'prompt': prompts[i]})
        episode_rows.append({'model': model, 'algorithm': 'fixed_move_near_100', 'budget': 100,
                             'run_id': 'run_1', 'eval_id': i, 'episode_id': episode_id,
                             'failure': failure, 'object_name': object_name, 'num_steps': len(poses)})
    write_csv(result_dir / 'dh_evaluations.csv', dh_fields, dh_rows)
    write_csv(result_dir / 'episodes.csv', episode_fields, episode_rows)


def main():
    args = parse_args()
    if not 0 <= args.seed <= 4294967295:
        raise ValueError('seed must be a uint32')
    tasks, prompts = load_inputs(args)
    result_dir = args.output.resolve() / args.model
    config = {'model': args.model, 'task': 'google_robot_move_near_customizable', 'seed': args.seed,
              'tasks_sha256': digest(args.tasks), 'prompts_sha256': digest(args.prompts), 'video': args.video}
    manifest = result_dir / 'run_config.json'
    if result_dir.exists():
        if not args.resume:
            raise FileExistsError('Output exists: %s. Use --resume or choose another --output.' % result_dir)
        if not manifest.is_file() or json.loads(manifest.read_text(encoding='utf-8')) != config:
            raise ValueError('Cannot resume: run configuration does not match %s' % manifest)
    else:
        result_dir.mkdir(parents=True)
        write_json(manifest, config)
    # The existing interface uses checkpoint paths relative to experiments.
    os.chdir(DEEPHYPERION_ROOT / 'experiments')
    import torch
    from model_interface_allMetrics import VLAInterface
    from run_fuzzer_allMetrics_integrated import SimRunner
    from tqdm import tqdm
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)
    export_csvs(result_dir, args.model, tasks, prompts)
    pending = [i for i in range(100) if not (result_dir / 'allMetrics' / str(i) / 'log.json').is_file()]
    if not pending:
        print('All 100 episodes are already complete.', flush=True)
        return
    vla = VLAInterface(task=config['task'], model_name=args.model, instability_methods=list(METHODS))
    vla.model = PromptPolicy(vla.model)
    runner = SimRunner(vla, args.model, str(args.tasks.resolve()), None, seed=args.seed, resume=True)
    # tqdm estimates the remaining time from episodes completed in this run,
    # while initial displays episodes already completed before --resume.
    with tqdm(total=100, initial=100 - len(pending), desc=args.model,
              unit='episode', dynamic_ncols=True) as progress:
        def after_episode(idx, options, prompt, episode_stats, tcp_poses):
            export_csvs(result_dir, args.model, tasks, prompts)
            progress.update(1)
        runner.run_exec(task_type_override='move', prompts=prompts, result_dir_override=result_dir,
                        episode_callback=after_episode, write_video=args.video)
    export_csvs(result_dir, args.model, tasks, prompts)
    print('Completed 100 episodes: %s' % result_dir, flush=True)


if __name__ == '__main__':
    main()
