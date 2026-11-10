"""Shared evaluation of alarm-rule changes using immutable final-run scores.

Cutoffs come from saved calibration, never test-year optimisation. Binary
rules are grouped using the main cooldown. They do not define new probability
scores: single-model AP uses its unchanged scores; OR AP is left undefined.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from data_processing.splits import canonical_input_hashes, resolve_recorded_input
from modelling.evaluation.alarm_events import build_alarm_events
from modelling.evaluation.metrics import metrics
from modelling.evaluation.targetability import targetable_ids
from modelling.utils.artifacts import evidence_dir
from modelling.utils.io import sha256


PARTS = ('inner_val', 'test')


def load_cells(runs, horizons, years=None):
    """Load compatible final fits and record the exact saved evidence used.

    Args:
        runs: Completed final-run directories.
        horizons: Forecast horizons to include.
        years: Optional test years; otherwise use each run's saved years.

    Returns:
        Fit records keyed by model, tier configuration, horizon and test year,
        plus source paths and hashes for an output manifest.

    Raises:
        ValueError: Runs, fit identities, scores or shared inputs disagree.
    """
    cells, sources, expected_inputs = {}, [], None
    for run in runs:
        run = Path(run)
        info = json.loads((run / 'run.json').read_text())
        selected_h = sorted(set(horizons) & set(info['horizons']))
        selected_y = sorted(set(info['years']) & set(
            info['years'] if years is None else years
        ))
        if not selected_h or not selected_y:
            continue
        if (info.get('status') != 'complete'
                or not info.get('refit_after_calibration')
                or info.get('policy') != 'exhaustive_lowest'):
            raise ValueError(f'Not a completed final refit: {run}')
        inputs = canonical_input_hashes(info['input_sha256'])
        if expected_inputs is not None and inputs != expected_inputs:
            raise ValueError('Runs use different dataset inputs')
        expected_inputs = inputs
        for path, digest in inputs.items():
            if sha256(resolve_recorded_input(path)) != digest:
                raise ValueError(f'Input changed: {path}')
        print(f'Loading saved scores from {run}', flush=True)
        evidence = evidence_dir(run)
        episodes_path = evidence / 'episodes.parquet'
        episodes = pd.read_parquet(episodes_path)
        episodes['start'] = pd.to_datetime(episodes.start, utc=True)
        sources.append(dict(path=str(run), run_id=info['run_id'],
                            manifest_sha256=sha256(run / 'run.json'),
                            episodes_sha256=sha256(episodes_path)))
        for model in info['models']:
            for config in info['configs']:
                for h in selected_h:
                    for year in selected_y:
                        key = (model, config, h, year)
                        if key in cells:
                            raise ValueError(f'Duplicate final fit: {key}')
                        folder = evidence / 'fits' / (
                            f'{model}_{config}_h{h}_{year}'
                        )
                        spec_path = folder / 'fit.json'
                        spec = json.loads(spec_path.read_text())
                        actual = (spec['model'], spec['config'],
                                  spec['horizon'], spec['year'])
                        if (actual != key or spec['run_id'] != info['run_id']
                                or not spec.get('refit_after_calibration')
                                or spec['threshold_policy'] != info['policy']):
                            raise ValueError(f'Fit identity mismatch: {folder}')
                        cutoff = float(spec['threshold'])
                        if np.isnan(cutoff) or cutoff == -np.inf:
                            raise ValueError(f'Invalid saved cutoff: {folder}')
                        parts, hashes = {}, {}
                        for part in PARTS:
                            path = folder / f'{part}.parquet'
                            frame = pd.read_parquet(path, columns=[
                                'coin', 'hour', 'horizon', 'y_true', 'y_score'
                            ])
                            frame['hour'] = pd.to_datetime(frame.hour, utc=True)
                            if (frame.empty
                                    or frame[['coin', 'hour']].isna().any().any()
                                    or frame.duplicated(['coin', 'hour']).any()
                                    or not np.isfinite(frame.y_score).all()
                                    or not frame.y_true.isin([0, 1]).all()
                                    or not frame.horizon.eq(h).all()):
                                raise ValueError(f'Invalid scored rows: {path}')
                            expected_year = year - 1 if part == 'inner_val' else year
                            if not frame.hour.dt.year.eq(expected_year).all():
                                raise ValueError(f'Wrong partition year: {path}')
                            parts[part] = frame.sort_values(
                                ['coin', 'hour']
                            ).reset_index(drop=True)
                            hashes[str(path)] = sha256(path)
                        hashes[str(spec_path)] = sha256(spec_path)
                        cells[key] = dict(parts=parts, episodes=episodes,
                                          cutoff=cutoff, run_id=info['run_id'])
                        sources.append(dict(fit=str(folder), sha256=hashes))
    if not cells:
        raise ValueError('No final fits match the requested scope')
    return cells, sources


def consecutive_flags(pred, cutoff, hours=2):
    """Require consecutive same-coin calendar hours above a fixed cutoff.

    Missing hours and partition starts break a sequence. No future score or
    outcome label is used. A one-hour requirement is the original raw rule.
    """
    if not isinstance(hours, int) or hours < 1:
        raise ValueError('hours must be a positive integer')
    if pred.duplicated(['coin', 'hour']).any():
        raise ValueError('Duplicate coin-hour keys')
    result = pd.Series(False, index=pred.index)
    for _, group in pred.groupby('coin', sort=False):
        group = group.sort_values('hour')
        active = group.y_score.ge(cutoff)
        flags = active.copy()
        for lag in range(1, hours):
            contiguous = group.hour.sub(group.hour.shift(lag)).eq(
                pd.Timedelta(hours=lag)
            )
            flags &= active.shift(lag, fill_value=False) & contiguous
        result.loc[group.index] = flags
    return result


def align_pair(first, second):
    """Require identical paired observations and labels; preserve key order."""
    keys = ['coin', 'hour']
    for frame in (first, second):
        if frame[keys].isna().any().any() or frame.duplicated(keys).any():
            raise ValueError('Missing or duplicate paired keys')
    first = first.sort_values(keys).reset_index(drop=True)
    second = second.sort_values(keys).reset_index(drop=True)
    if (not first[keys].equals(second[keys])
            or not first.y_true.equals(second.y_true)
            or not first.horizon.equals(second.horizon)):
        raise ValueError('Paired origins, labels or horizons disagree')
    return first, second


def assess(pred, episodes, flags, risk_scores=None):
    """Evaluate a binary alarm rule with unchanged origins and cooldown.

    Args:
        pred: Eligible scored rows for one year and horizon.
        episodes: Reference episodes with UTC start times.
        flags: Boolean rule decisions aligned with pred's row order.
        risk_scores: Optional unchanged single-model scores for reporting AP.

    Returns:
        Metric values and grouped alarms. AP is missing for binary OR rules.
    """
    flags = np.asarray(flags)
    if flags.dtype != bool or flags.shape != (len(pred),):
        raise ValueError('Rule flags must be a matching boolean vector')
    scored = pred.copy()
    scored['y_score'] = flags.astype(float)
    scored['threshold'] = .5
    _, events = build_alarm_events(scored, episodes, .5)
    targets = targetable_ids(pred, episodes, int(pred.horizon.iloc[0]))
    values = metrics(scored, events, targets)
    values['average_precision'] = (
        float(average_precision_score(pred.y_true, risk_scores))
        if risk_scores is not None and pred.y_true.sum() else np.nan
    )
    return values, events.rename(columns={
        "y_score": "rule_signal", "threshold": "rule_threshold"
    })


def export_checks(output, rows, records, sources, notes):
    """Write yearly/pooled alarm-rule results with evidence and AP scope.

    Args:
        output: Destination directory for derived reports only.
        rows: Yearly result records containing test and calibration metrics.
        records: Tuples of identifiers, original test rows, alarms and scores.
        sources: Saved evidence paths and hashes.
        notes: Method definitions and limitations for the output manifest.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / 'by_year.csv', index=False)
    groups = {}
    all_events = []
    for identity, pred, events, risk in records:
        key = tuple(identity.items())
        groups.setdefault(key, []).append((pred, events, risk))
        all_events.append(events.assign(**identity))
    summary = []
    for identity, parts in groups.items():
        ids = dict(identity)
        yearly = [row for row in rows if all(
            row[key] == value for key, value in ids.items()
        )]
        pred = pd.concat([part[0] for part in parts], ignore_index=True)
        alarms = pd.concat([part[1] for part in parts], ignore_index=True)
        count_keys = ['rows', 'positive_hours', 'targetable_episodes',
                      'episodes_caught', 'false_alarms', 'alarm_events',
                      'true_alarms', 'observed_coin_months']
        totals = {key: sum(row[key] for row in yearly) for key in count_keys}
        targets = totals['targetable_episodes']
        months = totals['observed_coin_months']
        risk = (np.concatenate([part[2] for part in parts])
                if all(part[2] is not None for part in parts) else None)
        summary.append(dict(
            **ids, **totals,
            episode_recall=totals['episodes_caught'] / targets if targets else np.nan,
            false_alarms_per_coin_month=totals['false_alarms'] / months,
            median_lead_hours=alarms.loc[
                alarms.is_true_alarm, 'lead_hours'
            ].median(),
            average_precision=(
                float(average_precision_score(pred.y_true, risk))
                if risk is not None and pred.y_true.sum() else np.nan
            ),
        ))
    pd.DataFrame(summary).to_csv(output / 'summary.csv', index=False)
    pd.concat(all_events, ignore_index=True).to_csv(
        output / 'alarm_events.csv', index=False
    )
    (output / 'manifest.json').write_text(json.dumps(dict(
        sources=sources, notes=notes,
        cutoff_policy='fixed saved calibration cutoffs; no test-year selection',
        average_precision='unchanged single-model scores for single-model rules; undefined for binary OR',
    ), indent=2) + '\n')
    (output / 'README.md').write_text(
        '# Alarm-rule robustness\n\n' + '\n\n'.join(notes) + '\n\n'
        'by_year.csv includes calibration and test metrics. summary.csv pools '
        'test years. AP for smoothing refers to the unchanged underlying model '
        'scores, not a newly fitted or smoothed probability. OR has no unique '
        'probability ranking, so its AP is blank. Compare recall, false alarms '
        'and lead time jointly; no improvement is guaranteed.\n'
    )
