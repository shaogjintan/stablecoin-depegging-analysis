"""Count cooldown-grouped alarm events from pooled out-of-sample predictions.

Run from the repository root, using a threshold chosen on validation data:

    python modelling/alarm_events.py \
        --predictions results/predictions/xgboost_t2_h24_recall_tuned_pooled_test.parquet \
        --threshold 0.25

The first flagged hour for a coin starts an alarm. Further flagged hours for
that coin through the next h hours (inclusive) are ignored. An alarm is true
when a new episode starts in (alarm_hour, alarm_hour + h]; otherwise it is a
false alarm. The script reports both raw flagged-hour counts and grouped alarm
event counts, overall and by coin, year, and coin-month.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "data_processing"))

from config import PROCESSED_DATA_DIR  # noqa: E402


REQUIRED_PREDICTION_COLUMNS = {"coin", "hour", "horizon", "y_true", "y_score"}
EVENT_COLUMNS = [
    "coin", "alarm_hour", "year", "coin_month", "horizon", "threshold",
    "y_score", "is_true_alarm", "target_episode_id", "target_episode_start",
    "lead_hours",
]
COUNT_COLUMNS = [
    "flagged_hours", "alarm_events", "true_alarms", "false_alarms",
    "episodes_warned",
]


def build_alarm_events(
    predictions: pd.DataFrame,
    episodes: pd.DataFrame,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (flagged_hours, alarm_events) for one model/horizon run."""
    missing = REQUIRED_PREDICTION_COLUMNS - set(predictions.columns)
    if missing:
        raise ValueError(f"Predictions missing columns: {sorted(missing)}")
    if predictions["horizon"].nunique(dropna=True) != 1:
        raise ValueError("Input must contain predictions for exactly one horizon")
    if "split" in predictions and not predictions["split"].eq("test").all():
        raise ValueError("Use a pooled out-of-sample test file containing only split='test'")
    if predictions.duplicated(["coin", "hour"]).any():
        raise ValueError("Predictions contain duplicate coin-hour rows")
    if predictions["y_score"].isna().any() or predictions["y_true"].isna().any():
        raise ValueError("Pooled predictions must have observed scores and labels")

    horizon_values = predictions["horizon"].dropna().unique()
    horizon = int(horizon_values[0])
    if horizon <= 0:
        raise ValueError(f"Invalid forecast horizon: {horizon}")

    pred = predictions.copy()
    pred["hour"] = pd.to_datetime(pred["hour"], utc=True)
    flagged = pred.loc[pred["y_score"] >= threshold].copy()
    flagged["year"] = flagged["hour"].dt.year
    flagged["coin_month"] = flagged["hour"].dt.strftime("%Y-%m")

    eps = episodes[["coin", "episode_id", "start"]].copy()
    eps["start"] = pd.to_datetime(eps["start"], utc=True)
    starts_by_coin = {}
    for coin, group in eps.groupby("coin", sort=False):
        ordered = group.sort_values("start")
        starts = ordered["start"].tolist()
        starts_by_coin[coin] = (
            np.fromiter((start.value for start in starts), dtype=np.int64),
            ordered["episode_id"].to_numpy(),
            starts,
        )

    event_rows = []
    cooldown = pd.Timedelta(hours=horizon)
    for coin, group in flagged.sort_values(["coin", "hour"]).groupby("coin", sort=False):
        last_alarm_hour = None
        episode_data = starts_by_coin.get(coin)
        for row in group.itertuples(index=False):
            alarm_hour = row.hour
            # Suppress flags during (alarm time, alarm time + h], matching the
            # event target's inclusive future window (t, t+h].
            if last_alarm_hour is not None and alarm_hour <= last_alarm_hour + cooldown:
                continue

            target_pos = None
            if episode_data is not None:
                episode_times, episode_ids, episode_starts = episode_data
                pos = int(np.searchsorted(episode_times, alarm_hour.value, side="right"))
                if pos < len(episode_times) and episode_times[pos] <= (alarm_hour + cooldown).value:
                    target_pos = pos
            is_true = target_pos is not None
            target_start = episode_data[2][target_pos] if is_true else pd.NaT
            target_id = episode_data[1][target_pos] if is_true else None
            event_rows.append({
                "coin": coin,
                "alarm_hour": alarm_hour,
                "year": alarm_hour.year,
                "coin_month": alarm_hour.strftime("%Y-%m"),
                "horizon": horizon,
                "threshold": float(threshold),
                "y_score": float(row.y_score),
                "is_true_alarm": is_true,
                "target_episode_id": target_id,
                "target_episode_start": target_start,
                "lead_hours": (
                    (target_start - alarm_hour) / pd.Timedelta(hours=1)
                    if is_true else float("nan")
                ),
            })
            last_alarm_hour = alarm_hour

    events = pd.DataFrame(event_rows, columns=EVENT_COLUMNS)
    events["is_true_alarm"] = events["is_true_alarm"].astype(bool)
    return flagged, events


def _group_counts(
    flagged: pd.DataFrame,
    events: pd.DataFrame,
    keys: list[str],
) -> pd.DataFrame:
    """Combine raw flagged-hour counts with cooldown-grouped event counts."""
    hours = flagged.groupby(keys, dropna=False).size().rename("flagged_hours")
    if events.empty:
        counts = pd.DataFrame(columns=COUNT_COLUMNS, index=hours.index)
    else:
        grouped = events.groupby(keys, dropna=False)
        counts = grouped.agg(
            alarm_events=("is_true_alarm", "size"),
            true_alarms=("is_true_alarm", "sum"),
            episodes_warned=("target_episode_id", "nunique"),
        )
        counts["false_alarms"] = counts["alarm_events"] - counts["true_alarms"]
    out = hours.to_frame().join(counts, how="outer")
    for col in COUNT_COLUMNS:
        if col not in out:
            out[col] = 0
    return out[COUNT_COLUMNS].fillna(0).astype(int).reset_index()


def summarize_alarm_events(
    flagged: pd.DataFrame,
    events: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the requested coin, year, and coin-month summaries."""
    by_coin = _group_counts(flagged, events, ["coin"])
    by_year = _group_counts(flagged, events, ["year"])
    by_coin_month = _group_counts(flagged, events, ["coin", "coin_month"])
    return by_coin, by_year, by_coin_month


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True,
                        help="one pooled OOS test predictions parquet file")
    parser.add_argument("--threshold", required=True, type=float,
                        help="alarm score threshold, selected using validation data")
    parser.add_argument("--out-dir", default="reports/alarm_events",
                        help="directory for the event list and summary CSVs")
    args = parser.parse_args(argv)

    prediction_path = Path(args.predictions)
    predictions = pd.read_parquet(prediction_path)
    episodes = pd.read_parquet(PROCESSED_DATA_DIR / "depeg_episodes.parquet")
    flagged, events = build_alarm_events(predictions, episodes, args.threshold)
    by_coin, by_year, by_coin_month = summarize_alarm_events(flagged, events)

    horizon = int(predictions["horizon"].iloc[0])
    summary = pd.DataFrame([{
        "run": prediction_path.stem,
        "horizon": horizon,
        "threshold": args.threshold,
        "prediction_rows": len(predictions),
        "flagged_hours": len(flagged),
        "alarm_events": len(events),
        "true_alarms": int(events["is_true_alarm"].sum()) if not events.empty else 0,
        "false_alarms": int((~events["is_true_alarm"]).sum()) if not events.empty else 0,
        "episodes_warned": int(events["target_episode_id"].nunique()) if not events.empty else 0,
    }])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = prediction_path.stem
    outputs = {
        "events": (events, f"{stem}_alarm_events.csv"),
        "summary": (summary, f"{stem}_alarm_event_summary.csv"),
        "by_coin": (by_coin, f"{stem}_alarm_events_by_coin.csv"),
        "by_year": (by_year, f"{stem}_alarm_events_by_year.csv"),
        "by_coin_month": (by_coin_month, f"{stem}_alarm_events_by_coin_month.csv"),
    }
    for name, (frame, filename) in outputs.items():
        path = out_dir / filename
        frame.to_csv(path, index=False)
        print(f"Saved {name}: {path}")

    print("\nAlarm event summary:")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
