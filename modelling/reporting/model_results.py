"""Combine saved final h1/h6/h24 results without fitting models."""
import argparse
import json
from pathlib import Path

import pandas as pd

from data_processing.splits import canonical_input_hashes
from modelling.utils.io import sha256
from modelling.reporting.schema import HEADLINE_METRICS, write_comparison

RUNS = [
    Path('artifacts/runs') / group / name
    for group, name in [
        ('primary', 'REFIT-LOWEST-NEW'),
        ('primary', 'LR-REFIT-LOWEST-NEW'),
        ('supplementary', 'REFIT-LOWEST-H1'),
        ('supplementary', 'LR-REFIT-LOWEST-H1'),
    ]
]


def export_results(out, runs=None):
    """Combine compatible completed refits, rejecting duplicate result cells."""
    runs = RUNS if runs is None else runs
    summaries, yearly, sources = [], [], []
    expected_hashes = None
    for run in runs:
        info = json.loads((run / 'run.json').read_text())
        if (info.get('status') != 'complete'
                or not info.get('refit_after_calibration')
                or info.get('policy') != 'exhaustive_lowest'):
            raise ValueError(f'Not a completed final refit: {run}')
        hashes = canonical_input_hashes(info['input_sha256'])
        if expected_hashes is not None and hashes != expected_hashes:
            raise ValueError(f'Different dataset inputs: {run}')
        expected_hashes = hashes
        files = [run / name for name in ['summary.csv', 'by_year.csv']]
        for path, frames in zip(files, [summaries, yearly]):
            frame = pd.read_csv(path, float_precision='round_trip')
            if (not set(frame.horizon_hours).issubset(info['horizons'])
                    or not set(frame.model).issubset(info['models'])
                    or not set(frame.config).issubset(info['configs'])
                    or not frame.policy.eq(info['policy']).all()):
                raise ValueError(f'Results disagree with manifest: {path}')
            frames.append(frame)
        sources.append(dict(
            path=str(run), run_id=info['run_id'],
            sha256={p.name: sha256(p) for p in [run / 'run.json', *files]},
        ))
    out = Path(out)
    outputs = {}
    for name, frames, extra in [
        ('model_comparison_summary.csv', summaries, []),
        ('model_comparison_by_year.csv', yearly, ['test_year']),
    ]:
        frame = pd.concat(frames, ignore_index=True)
        keys = ['model', 'config', 'horizon_hours', 'policy', *extra]
        if frame.duplicated(keys).any():
            raise ValueError(f'Duplicate result cells: {name}')
        outputs[name] = frame.sort_values(keys).reset_index(drop=True)
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        keys = ['model', 'config', 'horizon_hours', 'policy']
        if 'test_year' in frame:
            keys.append('test_year')
        write_comparison(frame, out / name, keys)
    (out / 'model_comparison_manifest.json').write_text(json.dumps(dict(
        sources=sources, headline_metrics=HEADLINE_METRICS, primary_horizons=[6, 24], supplementary_horizons=[1],
        note='Saved results combined without retraining or changing metrics.',
    ), indent=2) + '\n')
    return outputs


def main():
    """Combine saved model summaries and yearly results into aligned reports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path,
                        default=Path('reports/model_results'))
    parser.add_argument('--run-dir', type=Path, action='append')
    args = parser.parse_args()
    outputs = export_results(args.out_dir, args.run_dir)
    print({name: len(frame) for name, frame in outputs.items()})


if __name__ == '__main__':
    main()
