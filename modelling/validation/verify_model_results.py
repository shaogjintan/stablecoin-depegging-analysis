"""Audit saved model metrics, alarm outcomes, calibration, and split boundaries.

Does not retrain or rewrite run evidence. JSON results go outside frozen runs.
"""
import argparse
import json
from pathlib import Path
from modelling.utils.artifacts import evidence_dir, FINAL_RUNS
import joblib
import numpy as np
import pandas as pd
from modelling.utils.io import sha256
from data_processing.splits import resolve_recorded_input
from data_processing.config import PURGE_HOURS, EMBARGO_HOURS
from modelling.validation.reference_checks import reference_events, reference_metrics, reference_ap, check_preprocessing
from modelling.evaluation.threshold_selection import _choose_threshold
RUNS = FINAL_RUNS


HEADLINE_METRICS = (
    'median_lead_hours', 'episode_recall', 'false_alarms',
    'false_alarms_per_coin_month', 'average_precision',
)


def unique_cell(frame, **identity):
    """Require exactly one saved result for the requested identity."""
    selected = frame
    for key, value in identity.items():
        selected = selected.loc[selected[key].eq(value)]
    if len(selected) != 1:
        raise ValueError(f'Missing/duplicate result cell: {identity}')
    return selected.iloc[0]


def check_fit_identity(spec, info, model, config, horizon, year):
    """Reject fit metadata belonging to a different run or experimental cell."""
    expected = dict(run_id=info['run_id'], model=model, config=config,
                    horizon=horizon, year=year, refit_after_calibration=True,
                    threshold_policy=info['policy'])
    for key, value in expected.items():
        if spec.get(key) != value:
            raise ValueError(f'Fit identity mismatch: {key}: {spec.get(key)!r} != {value!r}')


def check_metrics(actual, saved, context):
    """Compare all headline metrics and the episode/alarm supporting counts."""
    for key in HEADLINE_METRICS + ('episodes_caught', 'targetable_episodes', 'alarm_events'):
        np.testing.assert_allclose(
            actual[key], saved[key], rtol=1e-10, atol=1e-12,
            equal_nan=True, err_msg=f'{context} {key}',
        )


def pooled_metric_values(parts):
    """Pool counts, exposure, scores, and individual true-alarm lead times.

    Pool scores and true-alarm lead times; sum exposure and episode counts.
    """
    pred = pd.concat([pred for pred, _, _ in parts], ignore_index=True)
    counts = {
        key: sum(result[key] for _, result, _ in parts)
        for key in ['episodes_caught', 'targetable_episodes', 'false_alarms',
                    'alarm_events', 'observed_coin_months']
    }
    leads = pd.concat([
        events.loc[events.is_true_alarm.astype(bool), 'lead_hours']
        for _, _, events in parts
    ], ignore_index=True)
    targets, months = counts['targetable_episodes'], counts['observed_coin_months']
    return dict(
        **counts,
        episode_recall=counts['episodes_caught'] / targets if targets else np.nan,
        false_alarms_per_coin_month=counts['false_alarms'] / months if months else np.nan,
        average_precision=reference_ap(pred.y_true, pred.y_score),
        median_lead_hours=float(leads.median()) if len(leads) else np.nan,
    )



def check_split_boundaries(partitions, episodes, year,
                           purge_hours=PURGE_HOURS, embargo_hours=EMBARGO_HOURS):
    """Check all four saved partitions against calendar and episode exclusions.

    Final train legitimately contains practice training and calibration rows.
    The practice partitions must be subsets of final train, not substitutes
    for its full eligible window.
    """
    calibration = pd.Timestamp(f'{year - 1}-01-01', tz='UTC')
    test = pd.Timestamp(f'{year}-01-01', tz='UTC')
    next_year = pd.Timestamp(f'{year + 1}-01-01', tz='UTC')
    purge = pd.Timedelta(hours=purge_hours)
    embargo = pd.Timedelta(hours=embargo_hours)
    windows = {
        'inner_train': (None, calibration - purge),
        'inner_val': (calibration + embargo, test - purge),
        'train': (None, test - purge),
        'test': (test + embargo, next_year),
    }
    excluded = set()
    for boundary in [calibration, test]:
        overlaps = (episodes.end >= boundary - purge) & (episodes.start <= boundary + embargo)
        excluded.update(episodes.loc[overlaps, 'episode_id'].astype(str))
    keys = {}
    for name, (lower, upper) in windows.items():
        frame = partitions[name]
        if frame.empty or frame[['coin', 'hour']].isna().any().any():
            raise ValueError(f'Empty/missing split origins: {name}')
        if frame.duplicated(['coin', 'hour']).any():
            raise ValueError(f'Duplicate split origins: {name}')
        if not isinstance(frame.hour.dtype, pd.DatetimeTZDtype) or str(frame.hour.dt.tz) != 'UTC':
            raise ValueError(f'Split timestamps must be UTC: {name}')
        if not frame.hour.lt(upper).all() or (lower is not None and not frame.hour.ge(lower).all()):
            raise ValueError(f'Purge/embargo boundary failed: {name}')
        if frame.episode_id.dropna().astype(str).isin(excluded).any():
            raise ValueError(f'Boundary-straddling episode retained: {name}')
        keys[name] = pd.MultiIndex.from_frame(frame[['coin', 'hour']])
    for name in ['inner_train', 'inner_val']:
        if not keys[name].isin(keys['train']).all():
            raise ValueError(f'Practice partition outside final training window: {name}')
    if keys['inner_train'].isin(keys['inner_val']).any():
        raise ValueError('Practice training/calibration overlap')
    if keys['train'].isin(keys['test']).any():
        raise ValueError('Final training/test overlap')


def verify(run, check_scores=False):
    """Audit one saved run's identities, splits, alarms, and metrics."""
    info = json.loads((run / 'run.json').read_text())
    if (info.get('status') != 'complete' or not info.get('refit_after_calibration')
            or info.get('policy') != 'exhaustive_lowest'):
        raise ValueError(f'Unsupported/incomplete refit: {run}')
    if info.get('purge_hours') != PURGE_HOURS or info.get('embargo_hours') != EMBARGO_HOURS:
        raise ValueError('Run purge/embargo settings do not match the maintained protocol')
    for path, digest in info['input_sha256'].items():
        if sha256(resolve_recorded_input(path)) != digest:
            raise ValueError(f'Input changed: {path}')
    eps = pd.read_parquet(evidence_dir(run) / 'episodes.parquet')
    yearly = pd.read_csv(run / 'by_year.csv', float_precision='round_trip')
    expected_cells = len(info['models']) * len(info['configs']) * len(info['horizons']) * len(info['years'])
    if len(yearly) != expected_cells:
        raise ValueError('Unexpected number of yearly result cells')
    rows = []
    pooled = {}
    for model in info['models']:
        for config in info['configs']:
            for h in info['horizons']:
                for year in info['years']:
                    folder = evidence_dir(run) / 'fits' / f'{model}_{config}_h{h}_{year}'
                    spec = json.loads((folder / 'fit.json').read_text())
                    check_fit_identity(spec, info, model, config, h, year)
                    for file, key in [('pipeline.joblib', 'pipeline_sha256'), ('practice_pipeline.joblib', 'practice_sha256')]:
                        if sha256(folder / file) != spec[key]:
                            raise ValueError(f'Changed pipeline: {folder / file}')
                    pred = pd.read_parquet(folder / 'test.parquet')
                    val = pd.read_parquet(folder / 'inner_val.parquet')
                    train = pd.read_parquet(folder / 'train.parquet')
                    inner_train = pd.read_parquet(
                        folder / 'inner_train.parquet', columns=['coin', 'hour', 'episode_id'])
                    check_split_boundaries(dict(inner_train=inner_train, inner_val=val,
                                                train=train, test=pred), eps, year)
                    cut, stats = _choose_threshold(val, eps, h, 1.0, policy='lowest_feasible')
                    if not pred.threshold.eq(spec['threshold']).all() or not val.threshold.eq(spec['threshold']).all():
                        raise ValueError(f'Prediction cutoff mismatch: {folder}')
                    if cut != spec['threshold']:
                        raise ValueError(f'Cutoff mismatch: {folder}')
                    ve = reference_events(val, eps, cut, h)
                    months = len(val.assign(month=val.hour.dt.strftime('%Y-%m'))[['coin', 'month']].drop_duplicates())
                    if int((~ve.is_true_alarm).sum()) > months:
                        raise ValueError('Calibration budget failed')
                    events = reference_events(pred, eps, cut, h)
                    actual = reference_metrics(pred, events, eps, h)
                    saved = unique_cell(yearly, model=model, config=config,
                                        horizon_hours=h, test_year=year)
                    check_metrics(actual, saved, folder)
                    stored = pd.read_csv(folder / 'alarms.csv', float_precision='round_trip')
                    keys = ['coin', 'alarm_hour', 'target_episode_id']
                    a = events[keys].copy()
                    b = stored[keys].copy()
                    b['alarm_hour'] = pd.to_datetime(b.alarm_hour, utc=True)
                    pd.testing.assert_frame_equal(a.sort_values(['coin', 'alarm_hour']).reset_index(drop=True), b.sort_values(['coin', 'alarm_hour']).reset_index(drop=True), check_dtype=False)
                    if check_scores:
                        bundle = joblib.load(folder / 'pipeline.joblib')
                        check_preprocessing(bundle, train, spec['features'])
                        X = pred[spec['features']]
                        imputer = bundle['imputer']
                        predicted = bundle['model'].predict_proba(imputer.transform(X) if imputer else X)[:, 1]
                        np.testing.assert_allclose(predicted, pred.y_score, atol=1e-12, rtol=1e-10)
                    pooled.setdefault((model, config, h), []).append((pred, actual, events))
                    rows.append(dict(model=model, config=config, horizon_hours=h, test_year=year))
    summary = pd.read_csv(run / 'summary.csv', float_precision='round_trip')
    if len(summary) != len(pooled):
        raise ValueError('Unexpected number of pooled result cells')
    for (model, config, h), parts in pooled.items():
        row = unique_cell(summary, model=model, config=config, horizon_hours=h)
        actual = pooled_metric_values(parts)
        check_metrics(actual, row, f'{run} pooled {model}/{config}/h{h}')
        np.testing.assert_equal(actual['observed_coin_months'], row.coin_months)
    return dict(path=str(run), run_id=info['run_id'], manifest_sha256=sha256(run / 'run.json'), status='passed', cells=len(rows), checks=['input/pipeline hashes', 'four-partition calendar bounds, 25h purge, 24h embargo, episode exclusions and subset relationships', 'validation threshold and budget', 'independent greedy alarms', 'independent tie-aware AP', 'all five yearly and pooled headline metrics', 'fit identities and unique result cells'], model_score_replay=check_scores, limitations=['Threshold candidate selection reuses the maintained engine; its independent oracle is covered by tests.', 'LR C selection is not rerun.', 'Split checks verify saved-row boundaries and exclusions; they do not reconstruct the complete eligible-row population.'])

def main():
    """Audit saved runs and write verification results as JSON."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, action='append')
    p.add_argument('--check-model-scores', action='store_true')
    p.add_argument('--out', type=Path, default=Path('reports/verification.json'))
    args = p.parse_args()
    results = []
    for run in args.run_dir or RUNS:
        print('Verifying', run, flush=True)
        results.append(verify(run, args.check_model_scores))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(status='passed', runs=results), indent=2) + '\n')
    print('Passed', sum((r['cells'] for r in results)), 'cells')
if __name__ == '__main__':
    main()
