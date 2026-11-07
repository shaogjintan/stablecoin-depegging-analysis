"""Select calibration cutoffs under a coin-month false-alarm budget.

Select calibration cutoffs under an observed coin-month false-alarm budget.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from modelling.evaluation.alarm_events import build_alarm_events
from modelling.evaluation.metrics import (
    _targetable_episode_ids, _observed_coin_months, _false_alarm_stats,
)
POLICY_ALIASES = {"exact_max_recall": "max_validation_recall",
                  "exhaustive_lowest": "lowest_feasible"}


def normalize_policy(policy: str) -> str:
    """Map legacy policy names to canonical names; reject unknown policies."""
    policy = POLICY_ALIASES.get(policy, policy)
    if policy not in ("max_validation_recall", "lowest_feasible"):
        raise ValueError(f"Unknown threshold policy: {policy}")
    return policy


def _choose_threshold(
    validation: pd.DataFrame,
    episodes: pd.DataFrame,
    horizon: int,
    budget: float,
    policy: str = "max_validation_recall",
) -> tuple[float, dict]:
    """Select a calibration cutoff under the requested alarm-budget policy.

    Choose an explicit policy under the average validation FA/exposure budget.

    lowest_feasible selects the smallest feasible distinct score (or +inf).
    max_validation_recall maximizes catches, then fewer FA, then higher cutoff.
    Cooldown counts are NOT monotone. Never use test labels.
    """
    policy = normalize_policy(policy)
    scores = validation["y_score"].to_numpy(dtype=float)
    if not len(scores):
        raise ValueError("Validation fold has no scored rows")
    if not np.isfinite(scores).all():
        bad_count = int((~np.isfinite(scores)).sum())
        raise ValueError(
            f"Validation scores contain {bad_count} non-finite value(s); "
            "cannot select an alarm threshold"
        )
    if not np.isfinite(budget) or budget < 0:
        raise ValueError("Budget must be finite and nonnegative")
    target_ids = _targetable_episode_ids(validation, episodes, horizon)
    n_validation_coin_months = _observed_coin_months(validation)
    def evaluate(threshold: float) -> dict:
        """Calculate calibration alarm outcomes for one candidate cutoff."""
        _, events = build_alarm_events(validation, episodes, float(threshold))
        false_count, coin_months, rate, max_false_by_month = _false_alarm_stats(
            events, validation, n_validation_coin_months
        )
        caught = (
            set(events.loc[events["is_true_alarm"], "target_episode_id"].dropna().astype(str))
            if not events.empty else set()
        )
        recall = len(caught & target_ids) / len(target_ids) if target_ids else 0.0
        return {
            "threshold": float(threshold),
            "false_alarms": false_count,
            "coin_months": coin_months,
            "false_alarms_per_coin_month": rate,
            "max_false_alarms_in_coin_month": max_false_by_month,
            "episodes_targetable": len(target_ids),
            "episodes_warned": len(caught & target_ids),
            "episode_recall": recall,
            "alarm_events": len(events),
        }
    from modelling.evaluation.exact_threshold import exhaustive_counts
    frame = validation.sort_values(["coin", "hour"]).copy()
    frame["hour"] = pd.to_datetime(frame["hour"], utc=True)
    if frame.duplicated(["coin", "hour"]).any():
        raise ValueError("Duplicate validation coin-hour")
    times = frame.hour.astype("datetime64[ns, UTC]").astype("int64").to_numpy()
    targets = np.full(len(frame), -1, dtype=np.int64)
    ids = {ep: i for i, ep in enumerate(sorted(target_ids))}
    delta = horizon * 3600_000_000_000
    for coin, group in frame.groupby("coin", sort=False):
        eps = episodes.loc[episodes.coin.eq(coin)].sort_values("start")
        starts = pd.to_datetime(eps.start, utc=True).astype("datetime64[ns, UTC]").astype("int64").to_numpy()
        positions = np.flatnonzero(frame.coin.eq(coin))
        loc = np.searchsorted(starts, times[positions], side="right")
        for pos, j in zip(positions, loc):
            if j < len(starts) and starts[j] <= times[pos] + delta:
                targets[pos] = ids.get(str(eps.iloc[j].episode_id), -1)
    ends = frame.groupby("coin", sort=False).size().cumsum().to_numpy(dtype=np.int64)
    cutoffs, caught, false, _ = exhaustive_counts(
        frame.y_score.to_numpy(dtype=float), times, ends, targets, delta)
    cutoffs = np.r_[cutoffs, np.inf]
    caught, false = np.r_[caught, 0], np.r_[false, 0]
    feasible = np.flatnonzero(false <= budget * n_validation_coin_months)
    idx = (min(feasible, key=lambda i: cutoffs[i]) if policy == "lowest_feasible"
           else max(feasible, key=lambda i: (caught[i], -false[i], cutoffs[i])))
    best = evaluate(float(cutoffs[idx]))
    assert best["episodes_warned"] == caught[idx] and best["false_alarms"] == false[idx]
    best["candidate_count"] = len(cutoffs)
    best["threshold_policy"] = policy
    return best["threshold"], best
