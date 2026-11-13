"""Regenerate saved reports or reproduce the full protocol from local sources.

Run with --full to rebuild features and data, train all three models for all
study horizons and years, verify scores, and generate robustness and reports.
The default mode retains saved-evidence report generation without training.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def full_plan(output, from_processed=False):
    """Build ordered module commands sharing one reproduction workspace.

    Args:
        output: Absolute or repository-relative output directory.
        from_processed: Retrain using the retained processed dataset.

    Returns:
        Ordered pairs of module names and command-line arguments.
    """
    output = Path(output).resolve()
    reports = output / 'reports'
    runs = output / 'artifacts' / 'runs'
    run = runs / 'primary' / 'ALL-MODELS'
    selected = ['--run-dir', str(run)]
    results = reports / 'model_results'
    stages = [
        ('data_processing.build_market_features', []),
        ('data_processing.build_onchain_features', []),
        ('data_processing.build_dataset', ['--rebuild']),
        ('modelling.training.train_models', ['--out-dir', str(run),
            '--models', 'logreg', 'random_forest', 'xgboost']),
        ('modelling.validation.verify_model_results', [*selected,
            '--check-model-scores', '--out', str(reports / 'verification.json')]),
        ('modelling.robustness_checks.alternative_definitions', [*selected,
            '--out-dir', str(reports / 'robustness' / 'alt_definitions')]),
        ('modelling.robustness_checks.alarm_smoothing', [*selected,
            '--out-dir', str(reports / 'robustness' / 'alarm_smoothing')]),
        ('modelling.robustness_checks.or_combination', [*selected,
            '--out-dir', str(reports / 'robustness' / 'or_combination')]),
        ('modelling.reporting.model_results', [*selected,
            '--out-dir', str(results)]),
        ('modelling.reporting.calibration_audit', [*selected, '--include-h1',
            '--out-dir', str(reports / 'validation_checks')]),
        ('modelling.reporting.by_coin', ['--artifacts', str(runs),
            '--summary', str(results / 'model_comparison_summary.csv'),
            '--out-dir', str(results)]),
        ('modelling.reporting.event_table', [
            '--out-dir', str(reports / 'depeg_events'),
            '--episode-out-dir', str(reports / 'depeg_episodes')]),
        ('modelling.reporting.detection_by_type', [*selected,
            '--event-dir', str(reports / 'depeg_events'),
            '--out-dir', str(results)]),
        ('modelling.reporting.coverage_table', [
            '--out-dir', str(reports / 'data_coverage')]),
        ('modelling.reporting.benchmarks', [*selected,
            '--out-dir', str(results)]),
        ('modelling.reporting.model_interpretation', [*selected,
            '--recompute-shap', '--out-dir', str(reports / 'model_interpretation')]),
    ]
    return stages[3:] if from_processed else stages


def saved_plan():
    """Return the existing frozen-evidence report workflow without training."""
    from modelling.utils.artifacts import published_baseline

    baseline = published_baseline()
    if baseline:
        run = (ROOT / baseline['run_dir']).resolve()
        original = str(ROOT / 'artifacts/runs/primary/ALL-MODELS')
        original_runs = str(ROOT / 'artifacts/runs')
        return [
            (module, [str(run) if arg == original else
                      str(run.parent.parent) if arg == original_runs else arg
                      for arg in args])
            for module, args in full_plan(ROOT)[4:]
        ]
    return [(module, []) for module in [
        'data_processing.build_dataset',
        'modelling.reporting.model_results',
        'modelling.reporting.calibration_audit',
        'modelling.reporting.by_coin',
        'modelling.reporting.event_table',
        'modelling.reporting.detection_by_type',
        'modelling.reporting.coverage_table',
        'modelling.reporting.model_interpretation',
    ]]


def run_full(output, plan, env, from_processed=False):
    """Execute a fresh local rebuild and record successful stages or failure.

    Args:
        output: Fresh output workspace; existing paths are rejected.
        plan: Ordered module commands from full_plan().
        env: Child-process environment scoped to the new dataset root.
        from_processed: Copy the retained processed inputs without rebuilding.
    """
    from modelling.utils.io import sha256

    hourly = ROOT / 'data/raw/hourly_stablecoin.parquet'
    archive = ROOT / 'data/raw/dune_onchain.zip'
    sources = [hourly, archive]
    if from_processed:
        sources.extend(ROOT / 'data/processed' / name for name in (
            'master.parquet', 'depeg_episodes.parquet',
            'dataset_metadata.json',
        ))
    for path in sources:
        if not path.is_file():
            raise FileNotFoundError(f'Required retained source is missing: {path}')
    output.mkdir(parents=True, exist_ok=False)
    interim = output / 'data' / 'interim_reproduction'
    interim.mkdir(parents=True)
    env = dict(env, STABLECOIN_INTERIM_DIR=str(interim.resolve()))
    raw = output / "data" / "raw"
    raw.mkdir(parents=True)
    shutil.copy2(hourly, raw / hourly.name)
    shutil.copy2(archive, raw / archive.name)
    if from_processed:
        processed = output / 'data/processed'
        processed.mkdir(parents=True)
        for path in sources[2:]:
            shutil.copy2(path, processed / path.name)
    state = dict(status='running', sources=[
        dict(path=str(path), sha256=sha256(path)) for path in sources
    ], input_mode='retained-processed' if from_processed else 'source-rebuild',
        dataset_root=env['STABLECOIN_DATA_DIR'], completed=[],
        commands=[dict(module=m, args=a) for m, a in plan])
    manifest = output / 'reproduction.json'
    try:
        for module, args in plan:
            state['current'] = module
            manifest.write_text(json.dumps(state, indent=2) + '\n')
            print(f'Running {module}', flush=True)
            subprocess.run([sys.executable, '-W', 'ignore::DeprecationWarning',
                            '-m', module, *args], cwd=ROOT, env=env, check=True)
            state['completed'].append(module)
        state['status'] = 'complete'
    except Exception as exc:
        state['status'] = 'failed'
        state['error'] = str(exc)
        raise
    finally:
        manifest.write_text(json.dumps(state, indent=2) + '\n')


def main():
    """Select saved-report generation or full local-source reproduction."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full', action='store_true',
                        help='rebuild data, retrain all models, and report')
    parser.add_argument('--from-processed', action='store_true',
                        help='with --full, retrain on frozen processed inputs')
    parser.add_argument('--out-dir', type=Path,
                        default=Path('output/reproduction'))
    parser.add_argument('--dry-run', action='store_true',
                        help='print the ordered workflow without writing files')
    args = parser.parse_args()
    if args.from_processed and not args.full:
        parser.error('--from-processed requires --full')
    output = args.out_dir.resolve()
    env = dict(os.environ, MPLCONFIGDIR=str(
        Path(tempfile.gettempdir()) / 'stablecoin-matplotlib'
    ))
    if not args.full:
        from modelling.utils.artifacts import published_baseline
        from modelling.utils.io import sha256

        baseline = published_baseline()
        if baseline:
            run = ROOT / baseline['run_dir']
            info = json.loads((run / 'run.json').read_text())
            if (info['run_id'] != baseline['run_id']
                    or sha256(run / 'run.json') != baseline['manifest_sha256']):
                raise ValueError('Published baseline run identity has changed')
            env['STABLECOIN_DATA_DIR'] = str(ROOT / baseline['data_dir'])
    plan = full_plan(output, args.from_processed) if args.full else saved_plan()
    if args.full:
        env['STABLECOIN_DATA_DIR'] = str(output / 'data')
        env['STABLECOIN_INTERIM_DIR'] = str(
            output / 'data' / 'interim_reproduction')
    if args.dry_run:
        print(json.dumps(dict(
            mode='full' if args.full else 'saved-reports',
            input_mode=('retained-processed' if args.from_processed
                        else 'source-rebuild') if args.full else 'saved-evidence',
            dataset_root=env.get('STABLECOIN_DATA_DIR', 'data'),
            interim_root=env.get('STABLECOIN_INTERIM_DIR', 'data/interim'),
            commands=[dict(module=m, args=a) for m, a in plan],
        ), indent=2))
    elif args.full:
        run_full(output, plan, env, args.from_processed)
    else:
        for module, arguments in plan:
            subprocess.run([sys.executable, '-W', 'ignore::DeprecationWarning',
                            '-m', module, *arguments], cwd=ROOT, env=env,
                           check=True)


if __name__ == '__main__':
    main()
