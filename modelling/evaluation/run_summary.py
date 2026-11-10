"""Shared pooled summaries without training or threshold-search dependencies."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def aggregate(rows, events, scores):
    """Pool yearly counts, prediction AP, alarms, and conditional lead times."""
    yearly = pd.DataFrame(rows)
    keys = ["model", "config", "horizon_hours", "policy"]
    summary = yearly.groupby(keys, as_index=False).agg(
        false_alarms=("false_alarms", "sum"), coin_months=("coin_months", "sum"),
        episodes_caught=("episodes_caught", "sum"),
        targetable_episodes=("targetable_episodes", "sum"),
        alarm_events=("alarm_events", "sum"), true_alarms=("true_alarms", "sum"),
        test_rows=("test_rows", "sum"))
    summary["episode_recall"] = summary.episodes_caught / summary.targetable_episodes
    summary["false_alarms_per_coin_month"] = summary.false_alarms / summary.coin_months
    summary["alarm_precision"] = summary.true_alarms / summary.alarm_events.replace(0, np.nan)
    summary["budget_fa_per_coin_month"] = 1.
    summary["average_precision"] = [
        average_precision_score(scores[(r.model, r.config, r.horizon_hours)].y_true,
                                scores[(r.model, r.config, r.horizon_hours)].y_score)
        for r in summary.itertuples()]
    all_events = pd.concat(events, ignore_index=True)
    true = all_events.loc[all_events.is_true_alarm]
    leads = true.groupby(keys).lead_hours.median().rename("median_lead_hours").reset_index()
    summary = summary.merge(leads, on=keys, how="left")
    return yearly, summary, all_events

