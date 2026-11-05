"""
Expanding-window (walk-forward) cross-validation folds by calendar year.

For each test year Y from OOS_START_YEAR to OOS_END_YEAR:
  - train      = every hour before Y, minus a PURGE_HOURS purge at the end
  - test       = year Y, minus an EMBARGO_HOURS embargo at the start
  - inner_train / inner_val = the same split one year earlier (train on
    everything before Y-1, validate on Y-1), for nested tuning of
    hyperparameters and the decision threshold without touching year Y

Pooling the test predictions across folds scores every episode from
OOS_START_YEAR onwards out of sample, while staying strictly chronological.

Leakage control:
  - Purge: labels look up to 24h ahead plus one confirmation bar, so the last
    25h before each boundary are dropped from the earlier set.
  - Embargo: features use up to 24h of lookback, so the first 24h after each
    boundary are dropped from the later set -- no feature reaches back across it.
  - Episode-aware: any depeg episode overlapping a boundary window
    [boundary - purge, boundary + embargo] is dropped from every set in that fold.

Rows are returned unfiltered (unlabelled rows included); modelling/prepare_data.py
applies the tier- and horizon-specific eligibility filters.

Usage from modelling code (run from the repo root):
    import sys; sys.path.insert(0, "data_processing")
    from splits import load_master, expanding_window_folds
    master, episodes = load_master()
    for fold in expanding_window_folds(master, episodes):
        fold["year"], fold["train"], fold["test"], fold["inner_train"], fold["inner_val"]
"""
from __future__ import annotations

import pandas as pd
try:
    from .config import (
        PROCESSED_DATA_DIR, EMBARGO_HOURS, PURGE_HOURS, OOS_START_YEAR, OOS_END_YEAR,
        ACTIVE_MASTER_PATH, ACTIVE_EPISODES_PATH, ACTIVE_TYPED_EPISODES_PATH,
        MODELLING_DATA_VERSION,
    )
except ImportError:  # Direct script / historical top-level "splits" import.
    from config import (
        PROCESSED_DATA_DIR, EMBARGO_HOURS, PURGE_HOURS, OOS_START_YEAR, OOS_END_YEAR,
        ACTIVE_MASTER_PATH, ACTIVE_EPISODES_PATH, ACTIVE_TYPED_EPISODES_PATH,
        MODELLING_DATA_VERSION,
    )


def active_input_paths():
    """Resolve the committed typed manifest when no untyped manifest exists."""
    episodes_path = (ACTIVE_EPISODES_PATH if ACTIVE_EPISODES_PATH.exists()
                     else ACTIVE_TYPED_EPISODES_PATH)
    if not ACTIVE_MASTER_PATH.exists() or not episodes_path.exists():
        raise FileNotFoundError(
            "Corrected modelling inputs are missing. Run "
            "'python -m modelling.build_corrected_dataset' from the repo root."
        )
    return ACTIVE_MASTER_PATH, episodes_path


def load_master() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the versioned corrected master and its matching episode manifest."""
    master_path, episodes_path = active_input_paths()
    master = pd.read_parquet(master_path)
    episodes = pd.read_parquet(episodes_path)
    for name, frame in (("master", master), ("episodes", episodes)):
        if "dataset_version" not in frame or not frame["dataset_version"].eq(
                MODELLING_DATA_VERSION).all():
            raise ValueError(f"{name} does not match dataset version {MODELLING_DATA_VERSION}")
    master_columns = {"coin", "hour", "bar_start", "episode_id", "qualifying_depeg_hour"}
    episode_columns = {"coin", "start", "end", "episode_id"}
    for name, frame, columns in (("master", master, master_columns),
                                 ("episodes", episodes, episode_columns)):
        missing = columns - set(frame.columns)
        if missing:
            raise ValueError(f"{name} is missing columns {sorted(missing)}")
    for name, frame, columns in (("master", master, ("hour", "bar_start")),
                                 ("episodes", episodes, ("start", "end"))):
        for column in columns:
            dtype = frame[column].dtype
            if (not isinstance(dtype, pd.DatetimeTZDtype) or str(dtype.tz) != "UTC"
                    or frame[column].isna().any()):
                raise ValueError(f"{name}.{column} must contain UTC timestamps without missing values")
    if master.empty or master.duplicated(["coin", "hour"]).any():
        raise ValueError("Master must contain unique, nonempty coin-hour rows")
    if not (master.hour - master.bar_start).eq(pd.Timedelta(hours=1)).all():
        raise ValueError("Master does not use completed-bar close timestamps")
    if episodes.episode_id.isna().any() or episodes.episode_id.duplicated().any():
        raise ValueError("Episode IDs must be present and unique")
    if (not pd.api.types.is_bool_dtype(master.qualifying_depeg_hour)
            or master.qualifying_depeg_hour.isna().any()):
        raise ValueError("qualifying_depeg_hour must contain nonmissing boolean values")
    bounds = master.loc[master.qualifying_depeg_hour].groupby("episode_id").agg(
        coin=("coin", "first"), start=("hour", "min"), end=("hour", "max"))
    bounds["end"] += pd.Timedelta(hours=1)
    expected = episodes.set_index("episode_id")[["coin", "start", "end"]]
    if not bounds.sort_index().equals(expected.sort_index()):
        raise ValueError("Episode manifest IDs, coins or clocks do not match the master")
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
