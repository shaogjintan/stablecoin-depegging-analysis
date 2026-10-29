"""LOCAL-ONLY driver: classify depeg events into the deck's five types and
count caught/missed by type (counts only, per Prof Huang -- no per-type
detection rates).

Type rules (deck P8):
  - market-wide, simultaneous : >=2 coins, onsets <=1h apart
  - market-wide, staggered    : >=2 coins, onsets >1h apart
  - sustained single-coin     : 1 coin, >24h below $0.99 (episode n_hours)
  - short single-coin         : 1 coin, <=24h, and (>3h or low <$0.98)
  - brief wobble              : 1 coin, <=3h, low >=$0.98

Caught/missed is read from committed XGBoost rolling-threshold alarm events
(reports/rolling_thresholds/xgboost_*_test_alarm_events.csv); targetable
episodes come from the fixed-config pooled predictions.

Usage: python plans/event_types.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "data_processing"))

from config import PROCESSED_DATA_DIR  # noqa: E402


def classify(episodes: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """One row per event with its type."""
    rows = []
    for _, ev in events.iterrows():
        mem = episodes[episodes["event_id"] == ev["event_id"]]
        n_coins = mem["coin"].nunique()
        spread = (mem["start"].max() - mem["start"].min()) / pd.Timedelta(hours=1)
        if n_coins >= 2:
            typ = ("market-wide, simultaneous" if spread <= 1.0
                   else "market-wide, staggered")
        else:
            r = mem.iloc[0]
            if r["n_hours"] > 24:
                typ = "sustained single-coin"
            elif r["n_hours"] > 3 or r["min_price"] < 0.98:
                typ = "short single-coin"
            else:
                typ = "brief wobble"
        rows.append({"event_id": ev["event_id"], "type": typ,
                     "n_coins": int(n_coins), "onset_spread_hours": float(spread),
                     "n_episodes": int(len(mem))})
    return pd.DataFrame(rows)


def targetable_ids(pred_path: Path, episodes: pd.DataFrame, horizon: int) -> set[str]:
    """Episodes with >=1 positive prediction row (same rule as
    rolling_threshold._targetable_episode_ids)."""
    pred = pd.read_parquet(pred_path)
    starts = {c: g.sort_values("start") for c, g in episodes.groupby("coin", sort=False)}
    delta = pd.Timedelta(hours=horizon)
    ids: set[str] = set()
    for row in pred[pred["y_true"].eq(1)].itertuples(index=False):
        ce = starts.get(row.coin)
        if ce is None:
            continue
        hit = ce.loc[(ce["start"] > row.hour) & (ce["start"] <= row.hour + delta)]
        if not hit.empty:
            ids.add(str(hit.iloc[0]["episode_id"]))
    return ids


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--event-dir", default=str(REPO / "reports/event_tables"))
    ap.add_argument("--alarm-dir", default=str(REPO / "reports/rolling_thresholds"))
    ap.add_argument("--pred-dir", default=str(REPO / "results/predictions"))
    args = ap.parse_args()

    ev_dir, al_dir, pd_dir = Path(args.event_dir), Path(args.alarm_dir), Path(args.pred_dir)
    episodes = pd.read_csv(ev_dir / "episodes.csv", parse_dates=["start", "end"])
    events = pd.read_csv(ev_dir / "events.csv")
    types = classify(episodes, events)

    types.to_csv(ev_dir / "types_events.csv", index=False)
    summary = (types.groupby("type")
               .agg(events=("event_id", "size"), episodes=("n_episodes", "sum"))
               .reset_index())
    summary.to_csv(ev_dir / "types_summary.csv", index=False)
    print("Saved types_events.csv / types_summary.csv")
    print(summary.to_string(index=False))
    print(f"total episodes typed: {types['n_episodes'].sum()} (manifest has 79)")

    # caught / missed by type from committed XGBoost alarm events (real-time)
    epm = pd.read_parquet(PROCESSED_DATA_DIR / "depeg_episodes.parquet")
    epm["start"] = pd.to_datetime(epm["start"], utc=True)
    out_rows = []
    for tier in (1, 2):
        for horizon in (1, 6, 24):
            al = al_dir / f"xgboost_t{tier}_h{horizon}_xgb_d3_lr005_300_test_alarm_events.csv"
            if not al.exists():
                continue
            alarm = pd.read_csv(al)
            tgt = targetable_ids(
                pd_dir / f"xgboost_t{tier}_h{horizon}_xgb_d3_lr005_300_pooled_test.parquet",
                epm, horizon)
            caught_eps = set(alarm.loc[alarm["is_true_alarm"].eq(1),
                                       "target_episode_id"].dropna().astype(str))
            leads = alarm.loc[alarm["is_true_alarm"].eq(1), "lead_hours"].dropna()
            eps_typed = episodes.merge(types[["event_id", "type"]], on="event_id")
            for typ, g in eps_typed.groupby("type"):
                ids = set(g["episode_id"].astype(str))
                tgt_t = ids & tgt
                c = len(caught_eps & tgt_t)
                out_rows.append({
                    "tier": tier, "horizon": horizon, "type": typ,
                    "targetable_episodes": len(tgt_t),
                    "caught_episodes": c,
                    "caught_events": int(g.loc[g["episode_id"].astype(str).isin(caught_eps),
                                               "event_id"].nunique()),
                    "median_lead_hours": (
                        float(np.median(leads[alarm.loc[alarm["is_true_alarm"].eq(1),
                                                       "target_episode_id"].astype(str).isin(tgt_t)]))
                        if len(leads) else np.nan),
                })
    caught = pd.DataFrame(out_rows)
    caught.to_csv(ev_dir / "caught_by_type.csv", index=False)
    print("\nSaved caught_by_type.csv")
    print(caught.round(2).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
