"""Metrics for the maintained grouped-alarm evaluation protocol.

Counts use cooldown-grouped alarms and observed calendar coin-months.
Average precision pools prediction scores; lead time is conditional on true
alarms. Calibration cutoff selection lives in threshold_selection.py.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def _targetable_episode_ids(predictions: pd.DataFrame,
                            episodes: pd.DataFrame,
                            horizon: int) -> set[str]:
    """Episodes that at least one prediction origin could forecast in-window."""
    targets: set[str] = set()
    starts = {
        coin: group.sort_values("start")
        for coin, group in episodes.groupby("coin", sort=False)
    }
    positives = predictions.loc[predictions["y_true"].eq(1)]
    delta = pd.Timedelta(hours=horizon)
    for row in positives.itertuples(index=False):
        coin_eps = starts.get(row.coin)
        if coin_eps is None:
            continue
        hit = coin_eps.loc[
            (coin_eps["start"] > row.hour)
            & (coin_eps["start"] <= row.hour + delta)
        ]
        if not hit.empty:
            targets.add(str(hit.iloc[0]["episode_id"]))
    return targets


def _observed_coin_months(predictions: pd.DataFrame) -> int:
    months = predictions[["coin", "hour"]].copy()
    months["coin_month"] = pd.to_datetime(months["hour"], utc=True).dt.strftime("%Y-%m")
    return int(months[["coin", "coin_month"]].drop_duplicates().shape[0])


def _false_alarm_stats(
    events: pd.DataFrame,
    predictions: pd.DataFrame,
    n_coin_months: int | None = None,
) -> tuple[int, int, float, int]:
    if n_coin_months is None:
        n_coin_months = _observed_coin_months(predictions)
    false_count = (
        int(events["is_true_alarm"].eq(False).sum()) if not events.empty else 0
    )
    rate = false_count / n_coin_months if n_coin_months else float("inf")
    if false_count:
        false_by_month = (
            events.loc[events["is_true_alarm"].eq(False)]
            .groupby(["coin", "coin_month"])
            .size()
        )
        max_false_by_month = int(false_by_month.max())
    else:
        max_false_by_month = 0
    return false_count, n_coin_months, rate, max_false_by_month


def metrics(pred, events, targets):
    caught = set(events.target_episode_id.dropna().astype(str)) & targets
    fa, months, rate, maximum = _false_alarm_stats(events, pred)
    return dict(rows=len(pred), positive_hours=int(pred.y_true.sum()),
                positive_rate=float(pred.y_true.mean()),
                average_precision=float(average_precision_score(pred.y_true, pred.y_score)),
                targetable_episodes=len(targets), episodes_caught=len(caught),
                episode_recall=len(caught) / len(targets) if targets else np.nan,
                raw_flagged_hours=int(pred.y_score.ge(pred.threshold).sum()),
                alarm_events=len(events), true_alarms=int(events.is_true_alarm.sum()),
                false_alarms=fa, observed_coin_months=months,
                false_alarms_per_coin_month=rate, max_false_alarms_in_coin_month=maximum,
                median_lead_hours=float(events.loc[events.is_true_alarm, "lead_hours"].median()))

