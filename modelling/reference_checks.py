"""Independent saved-score and preprocessing oracles, preserved from the original verifier."""
import numpy as np
import pandas as pd
from modelling.lr_training import RF_PARAMS
HOUR = 3600_000_000_000

def reference_ap(y, scores):
    y, scores = np.asarray(y, dtype=int), np.asarray(scores, dtype=float)
    assert len(y) and np.isfinite(scores).all()
    if y.sum() == 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    labels, ranks = y[order], scores[order]
    ends = np.r_[np.flatnonzero(ranks[1:] != ranks[:-1]), len(ranks) - 1]
    positive = labels.cumsum()[ends]
    gained = np.diff(np.r_[0, positive])
    return float(np.sum(gained * positive / (ends + 1)) / labels.sum())

def reference_events(pred, episodes, cutoff, h):
    rows = []
    cooldown = h * HOUR
    for coin, g in pred.groupby("coin", sort=True):
        e = episodes.loc[episodes.coin.eq(coin)].sort_values("start")
        starts = e.start.astype("int64").to_numpy()
        ids = e.episode_id.to_numpy()
        flagged = g.loc[g.y_score.ge(cutoff)].sort_values("hour")
        last = None
        for hour, score in flagged[["hour", "y_score"]].itertuples(index=False, name=None):
            t = hour.value
            if last is not None and t - last <= cooldown:
                continue
            last = t
            pos = np.searchsorted(starts, t, side="right")
            caught = pos < len(starts) and starts[pos] - t <= cooldown
            rows.append(dict(coin=coin, alarm_hour=hour, y_score=score, is_true_alarm=bool(caught),
                             target_episode_id=ids[pos] if caught else None,
                             lead_hours=(starts[pos] - t) / HOUR if caught else np.nan))
    return pd.DataFrame(rows, columns=["coin", "alarm_hour", "y_score", "is_true_alarm",
                                      "target_episode_id", "lead_hours"])

def reference_targets(pred, episodes, h):
    targets = set()
    for coin, g in pred.groupby("coin", sort=False):
        times = np.sort(g.hour.astype("int64").to_numpy())
        for ep, start in episodes.loc[episodes.coin.eq(coin), ["episode_id", "start"]].itertuples(
                index=False, name=None):
            pos = np.searchsorted(times, start.value, side="left") - 1
            if pos >= 0 and start.value - times[pos] <= h * HOUR:
                targets.add(ep)
    return targets

def reference_metrics(pred, events, episodes, h):
    targets = reference_targets(pred, episodes, h)
    caught = set(events.target_episode_id.dropna()) & targets
    true = events.is_true_alarm.astype(bool)
    false = events.loc[~true]
    months = len(pred[["coin", "hour"]].assign(month=pred.hour.dt.strftime("%Y-%m"))[
        ["coin", "month"]].drop_duplicates())
    costs = false.assign(month=false.alarm_hour.map(lambda t: t.strftime("%Y-%m"))).groupby(
        ["coin", "month"]).size()
    return dict(rows=len(pred), positive_hours=int(pred.y_true.sum()),
                positive_rate=float(pred.y_true.mean()), average_precision=reference_ap(pred.y_true, pred.y_score),
                targetable_episodes=len(targets), episodes_caught=len(caught),
                episode_recall=len(caught) / len(targets) if targets else np.nan,
                raw_flagged_hours=int(pred.y_score.ge(pred.threshold).sum()),
                alarm_events=len(events), true_alarms=int(true.sum()), false_alarms=len(false),
                observed_coin_months=months, false_alarms_per_coin_month=len(false) / months,
                max_false_alarms_in_coin_month=int(costs.max()) if len(costs) else 0,
                median_lead_hours=float(events.lead_hours.median()))

def assert_values(actual, expected):
    for k, v in expected.items():
        np.testing.assert_allclose(actual[k], v, atol=1e-12, rtol=0, equal_nan=True, err_msg=k)

def check_preprocessing(bundle, train, features):
    model, imputer = bundle["model"], bundle["imputer"]
    X = train[features]
    if imputer is not None:
        pd.testing.assert_series_equal(imputer.medians_, X.median())
        X = imputer.transform(X)
    if model.name == "logreg":
        assert model.penalty == "l1" and model.class_weight == "balanced" and model.winsor == .01
        X = X.copy()
        X["abs_peg_dev"] = X.peg_deviation.abs()
        X["dev_x_vol6"] = X.peg_deviation * X.realised_vol_6h
        assert list(X.columns) == model.cols_
        pd.testing.assert_series_equal(model.lo_, X.quantile(.01))
        pd.testing.assert_series_equal(model.hi_, X.quantile(.99))
        a = X.clip(model.lo_, model.hi_, axis=1).to_numpy()
        np.testing.assert_allclose(model.scaler_.mean_, a.mean(axis=0), atol=1e-12, rtol=1e-12)
        variance = a.var(axis=0)
        np.testing.assert_allclose(model.scaler_.var_, variance, atol=1e-12, rtol=1e-12)
        assert model.clf_.solver == "liblinear" and model.clf_.random_state == 42
        assert max(model.clf_.n_iter_) < model.clf_.max_iter
        np.testing.assert_array_equal(model.clf_.classes_, [0, 1])
    elif model.name == "random_forest":
        assert_values(model.clf_.get_params(), {k: v for k, v in RF_PARAMS.items() if isinstance(v, int)})
        for k, v in RF_PARAMS.items():
            assert model.clf_.get_params()[k] == v
        assert model.feature_names_ == features and len(model.clf_.estimators_) == 500
        np.testing.assert_array_equal(model.clf_.classes_, [0, 1])
    elif model.name == "baseline":
        assert model.positive_rate_ == train.y_true.mean()
