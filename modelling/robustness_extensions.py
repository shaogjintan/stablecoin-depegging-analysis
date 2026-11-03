"""Saved-score robustness extensions: no fitting, validation-only calibration."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from numba import njit

from modelling.exact_threshold import exhaustive_counts
from modelling.verify_lr_rf_run import reference_events, reference_metrics, reference_targets, HOUR
from modelling.rolling_threshold import _choose_threshold

MODELS = ('logreg', 'random_forest', 'xgboost', 'persistence')
CONFIGS = ('market_matched', 'market_onchain')
PAIRS = (('logreg', 'xgboost'), ('logreg', 'random_forest'),
         ('random_forest', 'xgboost'), ('logreg', 'persistence'),
         ('xgboost', 'persistence'), ('random_forest', 'persistence'))
NAMES = dict(logreg='LR', random_forest='RF', xgboost='XGB', persistence='Pers', baseline='Prior')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(frame):
    frame = frame.copy()
    frame['hour'] = pd.to_datetime(frame.hour, utc=True).astype('datetime64[ns, UTC]')
    frame = frame.sort_values(['coin', 'hour']).reset_index(drop=True)
    assert not frame.duplicated(['coin', 'hour']).any()
    assert np.isfinite(frame.y_score).all()
    return frame


def smooth_scores(frame):
    # Two-hour confirmation stays eligible on later run hours, subject to cooldown.
    adjacent = frame.coin.eq(frame.coin.shift()) & frame.hour.diff().eq(pd.Timedelta(hours=1))
    return np.where(adjacent, np.minimum(frame.y_score, frame.y_score.shift()), -np.inf)


def oracle_flags(frame, flags, episodes, h):
    pred = frame.assign(y_score=np.asarray(flags, dtype=float), threshold=0.5)
    return reference_events(pred, episodes, 0.5, h)


def arrays(frame, episodes, h):
    times = frame.hour.astype('int64').to_numpy()
    ends = frame.groupby('coin', sort=False).size().cumsum().to_numpy(dtype=np.int64)
    targets = np.full(len(frame), -1, dtype=np.int64)
    skip = np.empty(len(frame), dtype=np.int64)
    ids = {str(ep): i for i, ep in enumerate(episodes.episode_id)}
    begin = 0
    for (coin, group), end in zip(frame.groupby('coin', sort=False), ends):
        eps = episodes.loc[episodes.coin.eq(coin)].sort_values('start')
        starts = eps.start.astype('int64').to_numpy()
        pos = np.searchsorted(starts, times[begin:end], side='right')
        valid = pos < len(starts)
        indexes = np.flatnonzero(valid)
        valid[indexes] &= starts[pos[indexes]] <= times[begin:end][indexes] + h * HOUR
        indexes = np.flatnonzero(valid)
        targets[begin + indexes] = [ids[str(eps.iloc[j].episode_id)] for j in pos[indexes]]
        skip[begin:end] = begin + np.searchsorted(times[begin:end], times[begin:end] + h * HOUR, side='right')
        begin = end
    return times, ends, targets, skip


def lowest_smoothed(frame, episodes, h, budget, data):
    scores = smooth_scores(frame)
    times, ends, targets, _ = data
    cut, caught, false, _ = exhaustive_counts(scores, times, ends, targets, h * HOUR)
    candidates = np.r_[np.unique(frame.y_score), np.inf]
    # Original score candidates matter: an interval can start below its next rolling-min score.
    indexes = np.searchsorted(-cut, -candidates, side='right') - 1
    nc, nf = np.zeros(len(candidates), dtype=int), np.zeros(len(candidates), dtype=int)
    valid = indexes >= 0
    nc[valid], nf[valid] = caught[indexes[valid]], false[indexes[valid]]
    choice = np.flatnonzero(nf <= budget)[0]
    return float(candidates[choice]), int(nc[choice]), int(nf[choice]), len(candidates)


def grid(scores, original):
    unique = np.unique(scores)
    if len(unique) > 200:
        quantiles = np.unique(np.r_[np.linspace(0, .95, 40), np.linspace(.95, .99, 40),
                                    np.linspace(.99, 1, 121)])
        unique = np.quantile(unique, quantiles, method='nearest')
    return np.unique(np.r_[unique, original, np.inf])


@njit
def next_flags(scores, thresholds):
    n = len(scores)
    result = np.empty((len(thresholds), n + 1), dtype=np.int32)
    for k in range(len(thresholds)):
        nxt = n
        result[k, n] = n
        for i in range(n - 1, -1, -1):
            if scores[i] >= thresholds[k]:
                nxt = i
            result[k, i] = nxt
    return result


@njit
def pair_counts(a, b, skip, targets, budget, n_episodes):
    caught = np.zeros((len(a), len(b)), dtype=np.int64)
    false = np.zeros_like(caught)
    seen = np.zeros(n_episodes, dtype=np.int64)
    epoch = 0
    n = len(skip)
    for i in range(len(a)):
        for j in range(len(b)):
            epoch += 1
            cursor = 0
            while cursor < n:
                pos = min(a[i, cursor], b[j, cursor])
                if pos == n:
                    break
                target = targets[pos]
                if target < 0:
                    false[i, j] += 1
                    if false[i, j] > budget:
                        break
                elif seen[target] != epoch:
                    seen[target] = epoch
                    caught[i, j] += 1
                cursor = skip[pos]
    return caught, false


def selected_pair(a, b, ga, gb, data, budget, n_episodes):
    nc, nf = pair_counts(a, b, data[3], data[2], budget, n_episodes)
    feasible = np.argwhere(nf <= budget)
    i, j = max(feasible, key=lambda x: (nc[tuple(x)], -nf[tuple(x)], ga[x[0]], gb[x[1]]))
    return float(ga[i]), float(gb[j]), int(nc[i, j]), int(nf[i, j]), len(ga) * len(gb)


def test_kernels():
    ep = pd.DataFrame(dict(coin=['A', 'B'], episode_id=['a', 'b'],
        start=pd.to_datetime(['2020-01-01 08:00Z', '2020-01-01 08:00Z'])))
    times = pd.to_datetime(['2020-01-01 00:00Z', '2020-01-01 01:00Z', '2020-01-01 03:00Z',
                           '2020-01-01 04:00Z', '2020-01-01 07:00Z', '2020-01-01 08:00Z'])
    f = normalize(pd.DataFrame(dict(coin=['A'] * 6 + ['B'] * 6, hour=list(times) * 2,
        y_score=[.4, .8, .9, .8, .9, .7] * 2, y_true=[0] * 12)))
    np.testing.assert_array_equal(smooth_scores(f) >= .5,
        [False, False, False, True, False, True] * 2)
    d = arrays(f, ep, 6)
    x, y = f.y_score.to_numpy(), f.y_score.to_numpy()[::-1].copy()
    ga, gb = np.r_[np.unique(x), np.inf], np.r_[np.unique(y), np.inf]
    nc, nf = pair_counts(next_flags(x, ga), next_flags(y, gb), d[3], d[2], 1000, len(ep))
    for i, ca in enumerate(ga):
        for j, cb in enumerate(gb):
            ev = oracle_flags(f, (x >= ca) | (y >= cb), ep, 6)
            assert nf[i, j] == (~ev.is_true_alarm.astype(bool)).sum()
            assert nc[i, j] == ev.target_episode_id.nunique()
    for budget in (0, 1, 2, 20):
        cut, nc, nf, _ = lowest_smoothed(f, ep, 6, budget, d)
        feasible = []
        for ca in ga:
            ev = oracle_flags(f, smooth_scores(f) >= ca, ep, 6)
            if (~ev.is_true_alarm.astype(bool)).sum() <= budget:
                feasible.append(ca)
        assert cut == min(feasible)
    return dict(pair_grid_all_cells=True, smoothing_gaps_and_coin_boundaries=True,
                exact_original_candidate_smoothing=True)


def run(session, out):
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    provenance = {}
    def read(path, columns=None):
        provenance[str(path)] = digest(path)
        return pd.read_parquet(path, columns=columns)
    def read_json(path):
        provenance[str(path)] = digest(path)
        return json.loads(path.read_text())
    final = session / 'REFIT-FULL-20261103'
    practice = session / 'OVERNIGHT-20261102-0232'
    cutoffs = session / 'THRESHOLD-LOWEST-VERIFIED-20261102'
    episodes = read(final / 'episodes.parquet')
    episodes['start'] = pd.to_datetime(episodes.start, utc=True).astype('datetime64[ns, UTC]')
    expected = pd.read_csv(final / 'pooled.csv')
    provenance[str(final / 'pooled.csv')] = digest(final / 'pooled.csv')
    tests = test_kernels()
    yearly, calibrations, checks, alarm_tables = [], [], [], []
    all_pred, all_events = {}, {}
    validation_original_checks = 0
    def record(variant, model, config, h, year, frame, events, **extra):
        key = variant, model, config, h
        metric = reference_metrics(frame.assign(threshold=.5, y_score=np.zeros(len(frame))), events, episodes, h)
        keep = ('rows', 'positive_hours', 'targetable_episodes', 'episodes_caught', 'episode_recall',
                'alarm_events', 'true_alarms', 'false_alarms', 'observed_coin_months',
                'false_alarms_per_coin_month', 'max_false_alarms_in_coin_month', 'median_lead_hours')
        yearly.append(dict(variant=variant, model=model, config=config, horizon_hours=h,
                           test_year=year, **{k: metric[k] for k in keep}, **extra))
        all_pred.setdefault(key, []).append(frame[['coin', 'hour', 'y_true']])
        all_events.setdefault(key, []).append(events)
        alarm_tables.append(events.assign(variant=variant, model=model, config=config,
                                         horizon_hours=h, test_year=year))
    for config in CONFIGS:
        for h in (6, 24):
            single_frames, single_events = {}, {}
            for year in range(2021, 2026):
                val, test, original = {}, {}, {}
                for model in (*MODELS, 'baseline'):
                    name = f'{model}_{config}_h{h}_{year}'
                    test[model] = normalize(read(final / 'fits' / name / 'test.parquet',
                        columns=['coin', 'hour', 'y_true', 'y_score', 'threshold']))
                    original[model] = float(test[model].threshold.iloc[0])
                    assert test[model].threshold.eq(original[model]).all()
                    assert test[model].hour.dt.year.eq(year).all()
                    ev = reference_events(test[model], episodes, original[model], h)
                    single_frames.setdefault(model, []).append(test[model])
                    single_events.setdefault(model, []).append(ev)
                    if model == 'baseline':
                        continue
                    source = (practice / 'XGB' / 'fits' / f'{config}_h{h}_{year}' if model == 'xgboost'
                              else practice / 'LR-RF' / 'fits' / name)
                    val[model] = normalize(read(source / 'inner_val.parquet',
                        columns=['coin', 'hour', 'y_true', 'y_score']))
                    val[model]['horizon'] = h
                    assert val[model].hour.dt.year.eq(year - 1).all()
                    source_fit = read_json(source / 'fit.json')
                    assert source_fit['fit_partition'] == 'inner_train'
                    assert source_fit['calibration_partition'] == 'inner_val'
                    evaluation = read_json(cutoffs / 'cells' / name / 'evaluation.json')
                    assert original[model] == float(evaluation['threshold'])
                    chosen, stats = _choose_threshold(val[model], episodes, h, 1., policy='lowest_feasible')
                    assert chosen == original[model], (name, chosen, original[model])
                    assert stats['false_alarms'] == evaluation['validation_false_alarms']
                    validation_original_checks += 1
                    record('single', model, config, h, year, test[model], ev, threshold=original[model])
                anchor = MODELS[0]
                for model in MODELS[1:]:
                    for data in (val, test):
                        pd.testing.assert_frame_equal(data[anchor][['coin', 'hour', 'y_true']],
                                                      data[model][['coin', 'hour', 'y_true']], check_exact=True)
                base = val[anchor]
                budget = len(base[['coin']].assign(month=base.hour.dt.strftime('%Y-%m')).drop_duplicates())
                data = arrays(base, episodes, h)
                grids, nexts = {}, {}
                for model in MODELS:
                    grids[model] = grid(val[model].y_score.to_numpy(), original[model])
                    nexts[model] = next_flags(val[model].y_score.to_numpy(), grids[model])
                    smooth = smooth_scores(test[model])
                    for variant in ('B1', 'B2'):
                        if variant == 'B1':
                            threshold = original[model]
                            n_candidates = 1
                        else:
                            threshold, nc, nf, n_candidates = lowest_smoothed(val[model], episodes, h, budget, data)
                        ve = oracle_flags(val[model], smooth_scores(val[model]) >= threshold, episodes, h)
                        vn = int((~ve.is_true_alarm.astype(bool)).sum())
                        vc = len(set(ve.target_episode_id.dropna()) & reference_targets(base, episodes, h))
                        if variant == 'B2':
                            assert (vc, vn) == (nc, nf)
                            assert vn <= budget
                        ev = oracle_flags(test[model], smooth >= threshold, episodes, h)
                        record(variant, model, config, h, year, test[model], ev, threshold=threshold)
                        calibrations.append(dict(variant=variant, model=model, config=config, horizon_hours=h,
                            test_year=year, validation_year=year-1, threshold_a=threshold,
                            validation_caught=vc, validation_false_alarms=vn, validation_coin_months=budget,
                            candidate_count=n_candidates))
                for a, b in PAIRS:
                    label = f'{a}+{b}'
                    for variant in ('A1', 'A2'):
                        if variant == 'A1':
                            ca, cb = original[a], original[b]
                            n_candidates = 1
                        else:
                            ca, cb, nc, nf, n_candidates = selected_pair(nexts[a], nexts[b], grids[a], grids[b],
                                                                        data, budget, len(episodes))
                        flags = (val[a].y_score >= ca) | (val[b].y_score >= cb)
                        ve = oracle_flags(base, flags, episodes, h)
                        vn = int((~ve.is_true_alarm.astype(bool)).sum())
                        vc = len(set(ve.target_episode_id.dropna()) & reference_targets(base, episodes, h))
                        if variant == 'A2':
                            assert (vc, vn) == (nc, nf), (label, config, h, year, vc, vn, nc, nf)
                            assert vn <= budget
                        flags = (test[a].y_score >= ca) | (test[b].y_score >= cb)
                        ev = oracle_flags(test[a], flags, episodes, h)
                        record(variant, label, config, h, year, test[a], ev, threshold_a=ca, threshold_b=cb)
                        calibrations.append(dict(variant=variant, model=label, config=config, horizon_hours=h,
                            test_year=year, validation_year=year-1, threshold_a=ca, threshold_b=cb,
                            validation_caught=vc, validation_false_alarms=vn, validation_coin_months=budget,
                            candidate_count=n_candidates, grid_a=len(grids[a]), grid_b=len(grids[b])))
                print(f'{config} h{h} {year} complete; {time.monotonic()-started:.1f}s', flush=True)
            for model in (*MODELS, 'baseline'):
                pred = pd.concat(single_frames[model], ignore_index=True)
                events = pd.concat(single_events[model], ignore_index=True)
                actual = reference_metrics(pred, events, episodes, h)
                row = expected.loc[expected.model.eq(model) & expected.config.eq(config) & expected.horizon_hours.eq(h)].iloc[0]
                for metric, value in actual.items():
                    np.testing.assert_allclose(value, row[metric], atol=1e-12, rtol=0, equal_nan=True,
                        err_msg=f'{model} {config} h{h} {metric}')
                checks.append(dict(model=model, config=config, horizon_hours=h, status='passed',
                    episodes_caught=actual['episodes_caught'], targetable_episodes=actual['targetable_episodes'],
                    false_alarms=actual['false_alarms'], metrics_checked=list(actual)))
    pooled = []
    for (variant, model, config, h), frames in all_pred.items():
        pred = pd.concat(frames, ignore_index=True).assign(y_score=0., threshold=.5)
        events = pd.concat(all_events[(variant, model, config, h)], ignore_index=True)
        m = reference_metrics(pred, events, episodes, h)
        assert m['observed_coin_months'] == 316
        for name in ('average_precision', 'raw_flagged_hours'):
            m.pop(name)
        pooled.append(dict(variant=variant, model=model, config=config, horizon_hours=h, **m))
    pooled = pd.DataFrame(pooled)
    singles = pooled.loc[pooled.variant.eq('single')].copy()
    combos = pooled.loc[pooled.variant.isin(['A1', 'A2'])].copy()
    smoothing = pooled.loc[pooled.variant.isin(['B1', 'B2'])].copy()
    for index, row in combos.iterrows():
        a, b = row.model.split('+')
        candidates = singles.loc[singles.config.eq(row.config) & singles.horizon_hours.eq(row.horizon_hours)
                                & singles.model.isin([a, b])]
        best = candidates.sort_values(['episodes_caught', 'false_alarms', 'model'], ascending=[False, True, True]).iloc[0]
        combos.loc[index, 'best_single_component'] = best.model
        for metric in ('episodes_caught', 'false_alarms', 'median_lead_hours'):
            combos.loc[index, 'best_single_' + metric] = best[metric]
            combos.loc[index, 'delta_' + metric] = row[metric] - best[metric]
    for index, row in smoothing.iterrows():
        base = singles.loc[singles.config.eq(row.config) & singles.horizon_hours.eq(row.horizon_hours)
                           & singles.model.eq(row.model)].iloc[0]
        for metric in ('episodes_caught', 'false_alarms', 'median_lead_hours'):
            smoothing.loc[index, 'unsmoothed_' + metric] = base[metric]
            smoothing.loc[index, 'delta_' + metric] = row[metric] - base[metric]
    singles.to_csv(out / 'pooled_singles.csv', index=False)
    combos.to_csv(out / 'pooled_combinations.csv', index=False)
    smoothing.to_csv(out / 'pooled_smoothing.csv', index=False)
    pd.DataFrame(yearly).to_csv(out / 'per_year.csv', index=False)
    pd.DataFrame(calibrations).to_csv(out / 'validation_calibration.csv', index=False)
    pd.concat(alarm_tables, ignore_index=True).to_csv(out / 'alarm_events.csv', index=False)
    unchanged = all(digest(Path(path)) == sha for path, sha in provenance.items())
    assert unchanged
    pd.DataFrame([dict(path=p, sha256=v) for p, v in provenance.items()]).to_csv(out / 'input_provenance.csv', index=False)
    check = dict(status='passed', single_model_pooled_checks=checks, validation_original_cutoffs_reproduced=validation_original_checks,
        kernel_checks=tests, chosen_extension_validation_metrics_oracle_checked=True,
        source_files_unchanged=unchanged, models_fitted=0, validation_year_only=True,
        pooled_observed_coin_months=316, seconds=time.monotonic()-started)
    (out / 'self_check.json').write_text(json.dumps(check, indent=2), encoding='utf-8')
    write_summary(out, singles, combos, smoothing)
    print(json.dumps({k:v for k,v in check.items() if k != 'single_model_pooled_checks'}, indent=2), flush=True)


def cell_text(row):
    lead = '—' if pd.isna(row.median_lead_hours) else f'{row.median_lead_hours:g}h'
    return f'{int(row.episodes_caught)}/{int(row.targetable_episodes)}; {int(row.false_alarms)}; {row.false_alarms_per_coin_month:.3f}; {lead}'


def write_summary(out, singles, combos, smooth):
    lines = ['# Saved-score robustness extensions — final-refit protocol', '',
        '**Sanity check: PASSED.** All 20 single-model/config/horizon pooled rows (including the constant prior) reproduce every numeric metric in REFIT-FULL pooled.csv within 1e-12; all 80 original validation cutoffs independently reproduced. Zero model fits. Sources unchanged (SHA-256 provenance).', '',
        '## Protocol and interpretation', '',
        '- Each 2021–2025 test year uses practice scores from the preceding validation year, then frozen cutoffs on saved final-refit test scores. No test score/label enters calibration. All models in each cell have exactly identical coin-hour/label keys.',
        '- Original/B2 cutoff candidates: every distinct original validation score plus +infinity (no alarms). B2 chooses the lowest feasible cutoff after smoothing, not maximum recall. The validation grouped-FA budget is its observed eligible coin-month count; this does not guarantee a test-year budget.',
        '- A1 ORs hourly flags at the original individual cutoffs, then groups once (not a union of separately grouped alarm events). A2 maximizes validation distinct episodes caught, then minimizes grouped FA, then maximizes the first and second named component cutoffs lexicographically, subject to the same single validation budget.',
        '- A2 uses each component’s distinct scores if <=200; otherwise ~200 distinct-score quantiles: 40 points through 95%, 40 from 95–99%, 121 from 99–100%, plus its original cutoff and +infinity. Full Cartesian grids are evaluated exactly; optimization is approximate relative to all distinct-score pairs. Tail density is fixed in advance, not tuned using test outcomes.',
        '- Smoothing is two-hour confirmation: the current and previous available eligible row for the same coin must both flag **and be exactly one UTC hour apart**. The first hour of a run cannot alarm; the second and later hours can, subject to cooldown. Missing/ineligible hours and coin/year/fold boundaries reset confirmation. Rolling minimum implements this rule; it does not use a future row.',
        '- Grouping and metrics reuse reference_events/reference_metrics. A further crossing within <=h hours of the last alarm is suppressed. A true alarm targets the next coin episode beginning strictly after alarm time and <=h hours ahead. Only targetable episodes count. Median lead is across true alarm events, as in the source results.',
        '- Cooldown resets for each annual test fold, matching source pooling. Pooled exposure is **316 coin-months** in each tier/horizon. Best single component is selected retrospectively for descriptive comparison only (caught first, fewer FA next); it does not calibrate extension cutoffs.',
        '- These are retrospective robustness checks on already-inspected test years, not a new untouched confirmatory experiment. Neither validation calibration nor joint budgets establish test-set feasibility.', '',
        '## Results', '', 'Every cell below is **caught/targetable; grouped FA; FA per coin-month; median lead**. A dash means no true alarms.', '']
    for h in (6, 24):
        for config, tier in zip(CONFIGS, ('Tier 1: market matched', 'Tier 2: market + on-chain')):
            lines += [f'### h{h} — {tier}', '', '| Single / smoothing | Original | B1: fixed cutoff | B2: validation recalibrated |', '|---|---|---|---|']
            for model in MODELS:
                filt = lambda d: d.loc[d.config.eq(config) & d.horizon_hours.eq(h) & d.model.eq(model)]
                base = filt(singles).iloc[0]
                b1 = filt(smooth).query("variant == 'B1'").iloc[0]
                b2 = filt(smooth).query("variant == 'B2'").iloc[0]
                lines.append(f'| {NAMES[model]} | {cell_text(base)} | {cell_text(b1)} | {cell_text(b2)} |')
            lines += ['', '| OR pair | Best single (original) | A1: individual cutoffs | A2: joint validation budget |', '|---|---|---|---|']
            for a, b in PAIRS:
                rows = combos.loc[combos.config.eq(config) & combos.horizon_hours.eq(h) & combos.model.eq(f'{a}+{b}')]
                a1, a2 = rows.query("variant == 'A1'").iloc[0], rows.query("variant == 'A2'").iloc[0]
                base = singles.loc[singles.config.eq(config) & singles.horizon_hours.eq(h) & singles.model.eq(a1.best_single_component)].iloc[0]
                lines.append(f'| {NAMES[a]}+{NAMES[b]} | {NAMES[base.model]}: {cell_text(base)} | {cell_text(a1)} | {cell_text(a2)} |')
            lines.append('')
    # These pre-specified examples describe all variants, not a test-selected winning policy.
    def fetch(d, variant, model, config='market_onchain', h=24):
        return d.loc[d.variant.eq(variant) & d.model.eq(model) & d.config.eq(config) & d.horizon_hours.eq(h)].iloc[0]
    lr = fetch(singles, 'single', 'logreg')
    a1, a2 = (fetch(combos, v, 'logreg+xgboost') for v in ('A1', 'A2'))
    b1, b2 = (fetch(smooth, v, 'logreg') for v in ('B1', 'B2'))
    lines += ['## Key findings', '',
        f'- Tier 2 h24 LR+XGB OR: naive {cell_text(a1)} versus LR {cell_text(lr)}; joint-budget {cell_text(a2)}. OR can change alarm timing through shared cooldown, so catches are not guaranteed to be a superset.',
        f'- Tier 2 h24 LR smoothing: fixed cutoff {cell_text(b1)}; validation recalibration {cell_text(b2)} versus unsmoothed {cell_text(lr)}. Recalibration must be reported separately from mechanical filtering.',
        f'- Across 24 pair/tier/horizon comparisons, A1 has more FA than the best component in {int((combos.query("variant == \'A1\'").delta_false_alarms > 0).sum())}; A2 catches more episodes in {int((combos.query("variant == \'A2\'").delta_episodes_caught > 0).sum())}. No averaging across overlapping tiers or horizons is appropriate.',
        '- A validation budget is not a test guarantee. All pooled FA rates use the same 316 observed coin-month denominator; see annual CSVs for distribution shifts and calibration grids.', '',
        '## Paste-ready slide 29', '',
        '**Title:** Robustness: combine alarms or require two-hour confirmation?', '',
        '**Box A — Model combination**',
        '- **Rule:** LR OR XGB hourly flags → shared horizon cooldown; compare individual cutoffs with validation-only joint-budget cutoffs.',
        f'- **Result (Tier 2, 24h):** LR {int(lr.episodes_caught)}/34 caught, {int(lr.false_alarms)} FA → naive OR {int(a1.episodes_caught)}/34, {int(a1.false_alarms)} FA; joint-budget OR {int(a2.episodes_caught)}/34, {int(a2.false_alarms)} FA.', '',
        '**Box B — Alarm smoothing**',
        '- **Rule:** LR must flag at two consecutive eligible UTC hours; alarm from the second hour, then shared cooldown.',
        f'- **Result (Tier 2, 24h):** Fixed cutoff {int(b1.episodes_caught)}/34 caught, {int(b1.false_alarms)} FA; validation-recalibrated {int(b2.episodes_caught)}/34, {int(b2.false_alarms)} FA (original {int(lr.episodes_caught)}/34, {int(lr.false_alarms)} FA).', '',
        '**Caveat:** Retrospective saved-score checks; no retraining or test-set calibration; joint search is a coarsened validation grid and validation budgets need not hold out of sample.', '',
        '## Files', '',
        '- pooled_singles.csv, pooled_combinations.csv, pooled_smoothing.csv: pooled metrics and explicit single-component deltas.',
        '- per_year.csv: annual test metrics and frozen cutoffs; validation_calibration.csv: validation metrics, exposure, cutoffs, and candidate counts.',
        '- alarm_events.csv: emitted annual alarm events; input_provenance.csv: SHA-256 hashes of all consumed data/cutoff/fit-metadata files.',
        '- self_check.json: exact single-model parity, validation cutoff reproduction, kernel/oracle checks, source immutability, no-fitting declaration.',
        '- evaluator.py: source snapshot; execute from repository root with `python -m modelling.robustness_extensions --session <S> --out <output>`.']
    (out / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'evaluator.py').write_bytes(Path(__file__).read_bytes())
    run(args.session, args.out)


if __name__ == '__main__':
    main()

