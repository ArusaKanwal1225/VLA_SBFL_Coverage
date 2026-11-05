#!/usr/bin/env python3
"""Generate an RQ1-style RQ5 transfer ROC-AUC table from model_metrics.csv."""
import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import pandas as pd
from experiments.RQ1.generate_tables import tie_aware_ranks, format_auc_rank, latex_escape

VLA_ORDER = ('EO1', 'Groot', 'Pi0', 'SpatialVLA')
DISPLAY = {'EO1': 'EO1', 'Groot': 'GR00T', 'Pi0': 'Pi0', 'SpatialVLA': 'SpatialVLA'}


def generate(input_path, output_dir):
    frame = pd.read_csv(input_path, dtype={'formula_id': str, 'technique': str, 'vla': str})
    required = {'vla', 'formula_id', 'technique', 'roc_auc'}
    if not required <= set(frame.columns) or frame.empty:
        raise ValueError('model_metrics.csv lacks required data')
    if frame.duplicated(['vla', 'formula_id']).any():
        raise ValueError('duplicate model/formula rows')
    models = [m for m in VLA_ORDER if m in set(frame.vla)]
    if not models or set(models) != set(frame.vla):
        raise ValueError('missing or unrecognised model names')
    if frame[['vla', 'formula_id', 'technique']].isna().any().any():
        raise ValueError('missing model, formula ID, or technique name')
    frame['roc_auc'] = pd.to_numeric(frame.roc_auc, errors='coerce')
    if not frame.roc_auc.map(math.isfinite).all() or not frame.roc_auc.between(0, 1).all():
        raise ValueError('invalid ROC-AUC values')
    groups = {model: frame[frame.vla == model].set_index('formula_id') for model in models}
    formula_ids = set(groups[models[0]].index)
    if len(formula_ids) != 34 or any(set(group.index) != formula_ids for group in groups.values()):
        raise ValueError('each evaluated model must contain the same 34 techniques')
    names = groups[models[0]]['technique'].to_dict()
    if any(group['technique'].to_dict() != names for group in groups.values()):
        raise ValueError('technique names differ between models')
    values = {model: groups[model]['roc_auc'].to_dict() for model in models}
    values['overall'] = {formula: sum(values[m][formula] for m in models) / len(models) for formula in formula_ids}
    ranks = {column: tie_aware_ranks(scores) for column, scores in values.items()}
    ordered = sorted(formula_ids, key=lambda f: (-values['overall'][f], f))
    rows = []
    for formula in ordered:
        row = {'formula_id': formula, 'technique': names[formula]}
        for column in [*models, 'overall']:
            name = DISPLAY.get(column, column)
            row[name + '_roc_auc'] = values[column][formula]
            row[name + '_rank'] = ranks[column][formula]
        rows.append(row)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / 'roc_auc_table.csv', index=False, float_format='%.12g')
    bs = chr(92)
    is_partial = len(models) != 4
    scope = ('the evaluated models only; this is a partial result' if is_partial else 'the four VLA models')
    caption = ('Mean move-near ROC-AUC of 34 SBFL techniques. Overall is the equal-weight mean across '
               + scope + '. Parentheses show tie-aware ranks; bold denotes rank 1 and underlining denotes ranks 2--5.')
    lines = [bs+'begin{table*}[t]', bs+'centering', bs+'caption{'+caption+'}',
             bs+'label{tab:rq5-transfer-roc-auc}', '{'+bs+'scriptsize',
             bs+'setlength{'+bs+'tabcolsep}{4pt}', bs+'renewcommand{'+bs+'arraystretch}{0.96}',
             bs+'begin{tabular}{l'+'c'*(len(models)+1)+'}', bs+'toprule',
             'Technique & '+' & '.join(DISPLAY[m] for m in models)+' & Overall '+bs*2, bs+'midrule']
    for formula in ordered:
        cells = [format_auc_rank(values[c][formula], ranks[c][formula], 5) for c in [*models, 'overall']]
        lines.append(latex_escape(names[formula])+' & '+' & '.join(cells)+' '+bs*2)
    lines += [bs+'bottomrule', bs+'end{tabular}', '}', bs+'end{table*}']
    (output_dir / 'roc_auc_table.tex').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('Generated table for:', ', '.join(DISPLAY[m] for m in models),
          '(partial overall)' if is_partial else '(all models)')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path, help='model_metrics.csv from an RQ5 evaluation')
    parser.add_argument('--output-dir', type=Path, help='Defaults to the input directory')
    args = parser.parse_args()
    generate(args.input, args.output_dir or args.input.parent)


if __name__ == '__main__':
    main()
