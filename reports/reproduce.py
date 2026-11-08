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


def full_plan(output):
    """Build ordered module commands sharing one reproduction workspace.

    Args:
        output: Absolute or repository-relative output directory.

    Returns:
        Ordered pairs of module names and command-line arguments.
    """
    output = Path(output).resolve()
    reports = output / 'reports'
    runs = output / 'artifacts' / 'runs'
    run = runs / 'primary' / 'ALL-MODELS'
    selected = ['--run-dir', str(run)]
    results = reports / 'model_results'
    return [
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


def saved_plan():
    """Return the existing frozen-evidence report workflow without training."""
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


def run_full(output, plan, env):
    """Execute a fresh local rebuild and record successful stages or failure.

    Args:
        output: Fresh output workspace; existing paths are rejected.
        plan: Ordered module commands from full_plan().
        env: Child-process environment scoped to the new dataset root.
    """
    from modelling.utils.io import sha256

    hourly = ROOT / 'data/raw/hourly_stablecoin.parquet'
    archive = ROOT / 'data/raw/dune_onchain.zip'
    for path in (hourly, archive):
        if not path.is_file():
            raise FileNotFoundError(f'Required retained source is missing: {path}')
    output.mkdir(parents=True, exist_ok=False)
    interim = output / 'data' / 'interim'
    interim.mkdir(parents=True)
    raw = output / "data" / "raw"
    raw.mkdir(parents=True)
    shutil.copy2(hourly, raw / hourly.name)
    shutil.copy2(archive, raw / archive.name)
    state = dict(status='running', sources=[
        dict(path=str(path), sha256=sha256(path)) for path in (hourly, archive)
    ], dataset_root=env['STABLECOIN_DATA_DIR'], completed=[],
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
    parser.add_argument('--out-dir', type=Path,
                        default=Path('artifacts/reproduction'))
    parser.add_argument('--dry-run', action='store_true',
                        help='print the ordered workflow without writing files')
    args = parser.parse_args()
    output = args.out_dir.resolve()
    env = dict(os.environ, MPLCONFIGDIR=str(
        Path(tempfile.gettempdir()) / 'stablecoin-matplotlib'
    ))
    plan = full_plan(output) if args.full else saved_plan()
    if args.full:
        env['STABLECOIN_DATA_DIR'] = str(output / 'data')
    if args.dry_run:
        print(json.dumps(dict(
            mode='full' if args.full else 'saved-reports',
            dataset_root=env.get('STABLECOIN_DATA_DIR', 'data'),
            commands=[dict(module=m, args=a) for m, a in plan],
        ), indent=2))
    elif args.full:
        run_full(output, plan, env)
    else:
        for module, arguments in plan:
            subprocess.run([sys.executable, '-W', 'ignore::DeprecationWarning',
                            '-m', module, *arguments], cwd=ROOT, env=env,
                           check=True)


if __name__ == '__main__':
    main()
