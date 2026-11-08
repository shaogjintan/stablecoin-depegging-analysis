"""Combine final models' raw alarm flags with fixed calibration cutoffs.

An OR flag occurs when either component score crosses its own saved cutoff.
One shared cooldown is then applied. No pair or threshold is selected using
held-out outcomes; all available pairs are reported. A component's calibration
budget does not imply that the combined rule satisfies that budget.
"""

import argparse
from itertools import combinations
from pathlib import Path

import pandas as pd

from modelling.robustness_checks.alarm_rules import (
    align_pair, assess, export_checks, load_cells,
)
from modelling.utils.artifacts import FINAL_RUNS


def main():
    """Export each model pair's component and OR-rule performance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, action='append')
    parser.add_argument('--out-dir', type=Path,
                        default=Path('reports/robustness/or_combination'))
    parser.add_argument('--horizons', type=int, nargs='+', default=[6, 24],
                        choices=[1, 6, 24])
    parser.add_argument('--years', type=int, nargs='+')
    args = parser.parse_args()
    cells, sources = load_cells(
        args.run_dir or FINAL_RUNS, args.horizons, args.years
    )
    scope = sorted({key[1:] for key in cells})
    rows, records = [], []
    for config, h, year in scope:
        models = sorted(key[0] for key in cells if key[1:] == (config, h, year))
        for model_a, model_b in combinations(models, 2):
            first = cells[(model_a, config, h, year)]
            second = cells[(model_b, config, h, year)]
            pd.testing.assert_frame_equal(first['episodes'], second['episodes'])
            evaluated = {}
            for part in ('inner_val', 'test'):
                a, b = align_pair(first['parts'][part], second['parts'][part])
                flags_a = a.y_score.ge(first['cutoff']).to_numpy()
                flags_b = b.y_score.ge(second['cutoff']).to_numpy()
                evaluated[part] = {}
                for variant, flags, risk in [
                    ('component_a', flags_a, a.y_score.to_numpy()),
                    ('component_b', flags_b, b.y_score.to_numpy()),
                    ('or', flags_a | flags_b, None),
                ]:
                    values, events = assess(a, first['episodes'], flags, risk)
                    evaluated[part][variant] = values
                    if part == 'test':
                        identity = dict(model=f'{model_a}+{model_b}',
                                        config=config, horizon_hours=h,
                                        variant=variant)
                        events = events.assign(
                            test_year=year, cutoff_a=first['cutoff'],
                            cutoff_b=second['cutoff'],
                        )
                        records.append((identity, a, events, risk))
            for variant in ('component_a', 'component_b', 'or'):
                row = dict(model=f'{model_a}+{model_b}', config=config,
                           horizon_hours=h, variant=variant, test_year=year,
                           model_a=model_a, model_b=model_b,
                           cutoff_a=first['cutoff'], cutoff_b=second['cutoff'],
                           **evaluated['test'][variant])
                calibration = evaluated['inner_val'][variant]
                row.update({f'calibration_{key}': value
                            for key, value in calibration.items()})
                row['calibration_within_original_budget'] = (
                    calibration['false_alarms_per_coin_month'] <= 1.
                )
                rows.append(row)
    if not rows:
        raise ValueError('OR comparison requires at least two models per cell')
    export_checks(args.out_dir, rows, records, sources, [
        'Report every available model pair within the same tier, horizon and '
        'test year. Components must have identical origins and labels.',
        'An hour is flagged if either model reaches its own fixed saved '
        'calibration cutoff. Apply one standard cooldown to the union of flags.',
        'No optimisation of thresholds, pairs or alarm budgets uses test '
        'outcomes. This is not the historical test-optimised OR analysis.',
        'The OR rule may exceed the original calibration budget. Inspect '
        'calibration and test false-alarm rates alongside recall and lead time. '
        'Cooldown interactions mean OR recall need not dominate both components.',
        'OR combines binary decisions, not probabilities; its average precision '
        'is undefined. Component AP uses each original model score.',
    ])
    print(f'Saved {len(rows)} yearly OR-comparison rows to {args.out_dir}')


if __name__ == '__main__':
    main()
