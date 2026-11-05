"""Read-only independent alarm/AP verification of saved final refit runs.

Does not retrain or rewrite run evidence. JSON results go outside frozen runs.
"""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from modelling.corrected_xgb import sha256
from data_processing.splits import resolve_recorded_input
from modelling.reference_checks import reference_events, reference_metrics, reference_ap, check_preprocessing
from modelling.rolling_threshold import _choose_threshold
RUNS = [Path('artifacts/runs') / kind / name for kind, name in [('primary', 'REFIT-LOWEST-NEW'), ('primary', 'LR-REFIT-LOWEST-NEW'), ('supplementary', 'REFIT-LOWEST-H1'), ('supplementary', 'LR-REFIT-LOWEST-H1')]]

def verify(run, check_scores=False):
    info = json.loads((run / 'run.json').read_text())
    if info.get('status') != 'complete' or not info.get('refit_after_calibration'):
        raise ValueError(f'Unsupported/incomplete refit: {run}')
    for path, digest in info['input_sha256'].items():
        if sha256(resolve_recorded_input(path)) != digest:
            raise ValueError(f'Input changed: {path}')
    eps = pd.read_parquet(run / 'episodes.parquet')
    yearly = pd.read_csv(run / 'by_year.csv', float_precision='round_trip')
    rows = []
    pooled = {}
    for model in info['models']:
        for config in info['configs']:
            for h in info['horizons']:
                for year in info['years']:
                    folder = run / 'fits' / f'{model}_{config}_h{h}_{year}'
                    spec = json.loads((folder / 'fit.json').read_text())
                    for file, key in [('pipeline.joblib', 'pipeline_sha256'), ('practice_pipeline.joblib', 'practice_sha256')]:
                        if sha256(folder / file) != spec[key]:
                            raise ValueError(f'Changed pipeline: {folder / file}')
                    pred = pd.read_parquet(folder / 'test.parquet')
                    val = pd.read_parquet(folder / 'inner_val.parquet')
                    train = pd.read_parquet(folder / 'train.parquet')
                    if not pred.hour.dt.year.eq(year).all() or not train.hour.lt(pd.Timestamp(f'{year}-01-01', tz='UTC') - pd.Timedelta(hours=25)).all():
                        raise ValueError(f'Split boundary failed: {folder}')
                    if pd.MultiIndex.from_frame(train[['coin', 'hour']]).isin(pd.MultiIndex.from_frame(pred[['coin', 'hour']])).any():
                        raise ValueError('Train/test overlap')
                    cut, stats = _choose_threshold(val, eps, h, 1.0, policy='lowest_feasible')
                    if cut != spec['threshold']:
                        raise ValueError(f'Cutoff mismatch: {folder}')
                    ve = reference_events(val, eps, cut, h)
                    months = len(val.assign(month=val.hour.dt.strftime('%Y-%m'))[['coin', 'month']].drop_duplicates())
                    if int((~ve.is_true_alarm).sum()) > months:
                        raise ValueError('Calibration budget failed')
                    events = reference_events(pred, eps, cut, h)
                    actual = reference_metrics(pred, events, eps, h)
                    saved = yearly.loc[yearly.model.eq(model) & yearly.config.eq(config) & yearly.horizon_hours.eq(h) & yearly.test_year.eq(year)]
                    if len(saved) != 1:
                        raise ValueError('Missing/duplicate yearly cell')
                    for key in ['average_precision', 'episodes_caught', 'targetable_episodes', 'false_alarms', 'alarm_events', 'median_lead_hours']:
                        np.testing.assert_allclose(actual[key], saved.iloc[0][key], rtol=1e-10, atol=1e-12, equal_nan=True, err_msg=str(folder) + ' ' + key)
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
                    pooled.setdefault((model, config, h), []).append((pred, actual))
                    rows.append(dict(model=model, config=config, horizon_hours=h, test_year=year))
    summary = pd.read_csv(run / 'summary.csv')
    for (model, config, h), parts in pooled.items():
        row = summary.loc[summary.model.eq(model) & summary.config.eq(config) & summary.horizon_hours.eq(h)].iloc[0]
        pred = pd.concat([p for p, _ in parts], ignore_index=True)
        for key in ['episodes_caught', 'targetable_episodes', 'false_alarms']:
            np.testing.assert_equal(sum((m[key] for _, m in parts)), row[key])
        np.testing.assert_allclose(reference_ap(pred.y_true, pred.y_score), row.average_precision, rtol=1e-10, atol=1e-12)
    return dict(path=str(run), run_id=info['run_id'], manifest_sha256=sha256(run / 'run.json'), status='passed', cells=len(rows), checks=['input/pipeline hashes', 'train/test boundaries', 'validation threshold and budget', 'independent greedy alarms', 'independent tie-aware AP', 'yearly and pooled metrics'], model_score_replay=check_scores, limitations=['Threshold candidate selection reuses the maintained engine; its independent oracle is covered by tests.', 'LR C selection is not rerun.'])

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, action='append')
    p.add_argument('--check-model-scores', action='store_true')
    p.add_argument('--out', type=Path, default=Path('reports/final/verification.json'))
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
