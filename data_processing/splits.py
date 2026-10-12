"""
Expanding-window (walk-forward) cross-validation folds by calendar year.

For each test year Y from OOS_START_YEAR to OOS_END_YEAR:
  - train      = every hour before Y, minus a PURGE_HOURS purge at the end
  - test       = year Y, minus an EMBARGO_HOURS embargo at the start
  - inner_train / inner_val = the same split one year earlier (train on
    everything before Y-1, validate on Y-1), for nested tuning of
    hyperparameters and the decision threshold without touching year Y

Pooling the test predictions across all folds scores every episode from
OOS_START_YEAR onwards out of sample, while staying strictly chronological.

Leakage control (same rules as the old single 3-way split):
  - Purge: labels look up to 24h ahead, so the last 24h before each boundary
    are dropped from the earlier set -- no label peeks across the boundary.
  - Embargo: features use up to 24h of lookback, so the first 24h after each
    boundary are dropped from the later set -- no feature reaches back across it.
  - Episode-aware: any depeg episode overlapping a boundary window
    [boundary - purge, boundary + embargo] is dropped from every set in that fold.

Rows are returned unfiltered (unlabelled rows included); filter on
label_h{h}.notna() and apply any at-risk filtering in the modelling step.

Usage from modelling code (run from the repo root):
    import sys; sys.path.insert(0, "data_processing")
    from splits import load_master, expanding_window_folds
    master, episodes = load_master()
    for fold in expanding_window_folds(master, episodes):
        fold["year"], fold["train"], fold["test"], fold["inner_train"], fold["inner_val"]
"""
from __future__ import annotations

import pandas as pd
from config import (
    PROCESSED_DATA_DIR, EMBARGO_HOURS, PURGE_HOURS, OOS_START_YEAR, OOS_END_YEAR,
)


def load_master() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load master.parquet and depeg_episodes.parquet (outputs of step 03)."""
    master = pd.read_parquet(PROCESSED_DATA_DIR / "master.parquet")
    episodes = pd.read_parquet(PROCESSED_DATA_DIR / "depeg_episodes.parquet")
    return master, episodes


def _year_start(year: int) -> pd.Timestamp:
    return pd.Timestamp(f"{year}-01-01", tz="UTC")


def _straddling_episodes(episodes: pd.DataFrame, boundaries: list[pd.Timestamp]) -> set:
    """Episode IDs overlapping [boundary - purge, boundary + embargo] for any boundary."""
    out = set()
    for b in boundaries:
        lo = b - pd.Timedelta(hours=PURGE_HOURS)
        hi = b + pd.Timedelta(hours=EMBARGO_HOURS)
        hit = episodes[(episodes["end"] >= lo) & (episodes["start"] <= hi)]
        out.update(hit["episode_id"])
    return out


def expanding_window_folds(master: pd.DataFrame, episodes: pd.DataFrame,
                           start_year: int = OOS_START_YEAR,
                           end_year: int = OOS_END_YEAR) -> list[dict]:
    """Return one dict per test year Y with keys
    year, train, test, inner_train, inner_val, dropped_episodes."""
    purge = pd.Timedelta(hours=PURGE_HOURS)
    embargo = pd.Timedelta(hours=EMBARGO_HOURS)
    hour = master["hour"]

    folds = []
    for year in range(start_year, end_year + 1):
        b_test = _year_start(year)        # train | test boundary
        b_val = _year_start(year - 1)     # inner_train | inner_val boundary
        b_next = _year_start(year + 1)

        train = master[hour < b_test - purge]
        test = master[(hour >= b_test + embargo) & (hour < b_next)]
        inner_train = master[hour < b_val - purge]
        inner_val = master[(hour >= b_val + embargo) & (hour < b_test - purge)]

        drop = _straddling_episodes(episodes, [b_val, b_test])
        sets = {
            name: df[~df["episode_id"].isin(drop)].copy()
            for name, df in [("train", train), ("test", test),
                             ("inner_train", inner_train), ("inner_val", inner_val)]
        }
        folds.append({"year": year, **sets, "dropped_episodes": sorted(drop)})
    return folds
