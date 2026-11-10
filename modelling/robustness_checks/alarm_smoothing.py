"""Evaluate consecutive-hour alarm confirmation using final saved scores.

Models, eligible origins and calibration-selected cutoffs stay fixed. The
check changes alarm construction only; no fitting or cutoff search is run.
"""

import argparse
from pathlib import Path

from modelling.robustness_checks.alarm_rules import (
    assess, consecutive_flags, export_checks, load_cells,
)
from modelling.utils.artifacts import FINAL_RUNS


def main():
    """Export original and consecutive-hour alarm-rule comparisons."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, action='append')
    parser.add_argument('--out-dir', type=Path,
                        default=Path('reports/robustness/alarm_smoothing'))
    parser.add_argument('--horizons', type=int, nargs='+', default=[6, 24],
                        choices=[1, 6, 24])
    parser.add_argument('--years', type=int, nargs='+')
    parser.add_argument('--hours', type=int, default=2)
    args = parser.parse_args()
    if args.hours < 2:
        parser.error('--hours must be at least two; baseline already uses one')
    cells, sources = load_cells(
        args.run_dir or FINAL_RUNS, args.horizons, args.years
    )
    rows, records = [], []
    for (model, config, h, year), cell in sorted(cells.items()):
        for variant, count in [('baseline', 1),
                               (f'consecutive_{args.hours}h', args.hours)]:
            identity = dict(model=model, config=config, horizon_hours=h,
                            variant=variant)
            results = {}
            for part in ('inner_val', 'test'):
                pred = cell['parts'][part]
                flags = consecutive_flags(pred, cell['cutoff'], count)
                values, events = assess(
                    pred, cell['episodes'], flags, pred.y_score.to_numpy()
                )
                results[part] = values
                if part == 'test':
                    events = events.assign(test_year=year,
                                           source_cutoff=cell['cutoff'])
                    records.append((identity, pred, events,
                                    pred.y_score.to_numpy()))
            row = dict(**identity, test_year=year,
                       source_cutoff=cell['cutoff'], **results['test'])
            row.update({f'calibration_{key}': value
                        for key, value in results['inner_val'].items()})
            row['calibration_within_original_budget'] = (
                results['inner_val']['false_alarms_per_coin_month'] <= 1.
            )
            rows.append(row)
    export_checks(args.out_dir, rows, records, sources, [
        f'Require {args.hours} consecutive eligible same-coin calendar hours '
        'above the fixed saved cutoff. Gaps and partition starts break runs.',
        'Use saved practice scores for calibration and saved final-model scores '
        'for test years. No new training or cutoff selection is performed.',
        'Apply the standard horizon-length cooldown after the confirmation '
        'rule. The denominator remains the original eligible population.',
        'Average precision uses unchanged model scores and therefore does not '
        'measure a change caused by smoothing.',
    ])
    print(f'Saved {len(rows)} yearly alarm-smoothing rows to {args.out_dir}')


if __name__ == '__main__':
    main()
