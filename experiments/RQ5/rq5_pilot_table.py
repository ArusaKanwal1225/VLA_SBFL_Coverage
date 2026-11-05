#!/usr/bin/env python3
"""Produce the RQ5 pilot outcome and sample-size table from episode logs."""
import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path

MODELS = [('pi0', r'$\pi_0$'), ('gr00t', 'GR00T'), ('eo1', 'EO1'),
          ('spatialvla-4b', 'SpatialVLA')]
DEFAULT_INPUT = Path('/home/ubuntu/Deephyperion_VLA/results/move_near_100')
DEFAULT_OUTPUT = DEFAULT_INPUT / 'rq5_pilot_sample_size.tex'
PILOT_SIZE = 100


def positive(value):
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return n


def get_success(path):
    with path.open(encoding='utf-8') as handle:
        log = json.load(handle)
    if not isinstance(log, dict) or not log:
        raise ValueError(f'{path}: empty or invalid episode log')
    try:
        last = log[max(log, key=int)]
        value = last['success']
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError(f'{path}: cannot read final success') from exc
    if isinstance(value, str):
        value = {'true': True, 'false': False}.get(value.strip().lower())
    if not isinstance(value, bool):
        raise ValueError(f'{path}: final success must be true or false')
    return value


def load_pilots(root):
    results, reference = [], None
    for model, label in MODELS:
        directory = root / model
        config_path = directory / 'run_config.json'
        config = json.loads(config_path.read_text(encoding='utf-8'))
        if config.get('model') != model:
            raise ValueError(f'{config_path}: model does not match {model}')
        signature = tuple(config.get(k) for k in ('task', 'seed', 'tasks_sha256', 'prompts_sha256'))
        if any(value is None for value in signature):
            raise ValueError(f'{config_path}: incomplete run configuration')
        if reference is None:
            reference = signature
        elif signature != reference:
            raise ValueError(f'{config_path}: pilot scenes, prompts, task or seed differ across models')
        log_paths = list((directory / 'allMetrics').glob('*/log.json'))
        ids = {p.parent.name for p in log_paths}
        expected = {str(i) for i in range(PILOT_SIZE)}
        if ids != expected or len(log_paths) != PILOT_SIZE:
            raise ValueError(f'{model}: expected episode IDs 0-99, found {len(ids)} logs')
        passes = sum(get_success(directory / 'allMetrics' / str(i) / 'log.json') for i in range(PILOT_SIZE))
        fails = PILOT_SIZE - passes
        results.append({'model': model, 'label': label, 'passes': passes, 'fails': fails,
                        'q': Fraction(min(passes, fails), PILOT_SIZE)})
    return results


def ceil_fraction(value):
    return -(-value.numerator // value.denominator)


def sample_size(results, regions):
    q = min(row['q'] for row in results)
    if not q:
        return q, None, None
    minimum = max(500, ceil_fraction(Fraction(100, 1) / q))
    size = regions * ((minimum + regions - 1) // regions) if regions is not None else None
    return q, minimum, size


def latex(results, q, minimum, size, regions):
    if minimum is None:
        n_text = r'--'
        note = (r'At least one pilot has no examples of one outcome; '
                r'the stated sample-size formula cannot give a finite $N$.')
    elif regions is None:
        n_text = rf'$K\lceil {minimum}/K\rceil$'
        note = (rf'The conservative minority proportion is $\hat{{q}}={float(q):.2f}$. '
                rf'The number of destination regions $K$ must be specified to compute $N$.')
    else:
        n_text = f'{size:,}'.replace(',', r'{,}')
        note = (rf'$\hat{{q}}={float(q):.2f}$, $K={regions}$; '
                rf'the same $N={n_text}$ is used for every VLA.')
    lines = [r'\begin{table}[t]', r'\centering',
             r'\caption{Pilot outcomes and target sample size per VLA.}',
             r'\label{tab:rq5-pilot-sample-size}',
             r'\begin{tabular}{lrrrrrr}', r'\hline',
             r'VLA & Pilot & Pass & Fail & $\hat{q}_m$ & Minimum $N_m$ & Selected $N$ \\', r'\hline']
    for row in results:
        model_minimum = ceil_fraction(Fraction(100, 1) / row['q']) if row['q'] else None
        minimum_text = f'{model_minimum:,}'.replace(',', r'{,}') if model_minimum is not None else '--'
        lines.append(rf"{row['label']} & {PILOT_SIZE} & {row['passes']} & {row['fails']} & {float(row['q']):.2f} & {minimum_text} & {n_text} \\")
    lines += [r'\hline', r'\end{tabular}', r'\par\smallskip',
              r'\footnotesize Minimum $N_m$ is $\lceil100/\hat{q}_m\rceil$, before the 500-execution floor and rounding. ' + note, r'\end{table}']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT, help='Pilot results root')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT, help='Output .tex file')
    parser.add_argument('--regions', type=positive, help='Number K of destination regions')
    args = parser.parse_args()
    try:
        results = load_pilots(args.input)
        q, minimum, size = sample_size(results, args.regions)
        output = latex(results, q, minimum, size, args.regions)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding='utf-8')
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        parser.exit(1, f'Error: {exc}\n')
    print(output)
    print(f'Table written to {args.output}', file=sys.stderr)
    if q == 0:
        print('No finite N: at least one pilot has zero passes or zero failures.', file=sys.stderr)
    elif args.regions is None:
        print('N is symbolic because --regions was not provided.', file=sys.stderr)
    else:
        print('All models share N; consider omitting the table as suggested in the draft.', file=sys.stderr)


if __name__ == '__main__':
    main()
