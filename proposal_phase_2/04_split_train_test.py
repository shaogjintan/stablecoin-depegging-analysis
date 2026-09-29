"""
Purged Chronological Train/Validation/Test Split with Episode-Aware Embargo.

Owner: Sasinthiran (Week 7 Deliverable)

Features & Methodology:
1. Chronological Partitioning:
   - Train:      2018-09-01 -> 2022-12-31 23:00 (Includes May 2022 Terra/USDT crash)
   - Validation: 2023-01-01 -> 2023-12-31 23:00 (Includes March 2023 SVB/USDC depeg)
   - Test:       2024-01-01 -> 2025-12-31 23:00 (Out-of-sample forward evaluation)
2. Universal 24-Hour Purging:
   - Purges the final 24 hours of Train and Validation so that forward-looking
     labels (for h = 1, 6, 24) do not peek into subsequent evaluation windows.
3. 24-Hour Embargoing:
   - Gaps the first 24 hours of Validation and Test so that 24h rolling lookback
     features (Rogers-Satchell volatility, rolling returns, rolling volume) do
     not compute statistics over preceding partition hours.
4. Episode-Aware Boundary Purging:
   - If a depeg episode straddles a boundary window, it is entirely dropped
     from both sides to guarantee zero partial-event contamination.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from config import PROCESSED_DATA_DIR, FORECAST_HORIZONS_HOURS, EMBARGO_HOURS

# Split Boundary Timestamps (UTC)
TRAIN_END = pd.Timestamp("2022-12-31 23:00:00", tz="UTC")
VAL_START = pd.Timestamp("2023-01-01 00:00:00", tz="UTC")
VAL_END   = pd.Timestamp("2023-12-31 23:00:00", tz="UTC")
TEST_START= pd.Timestamp("2024-01-01 00:00:00", tz="UTC")


def load_dataset():
    """Load master dataset and episode manifest with filename fallbacks."""
    master_path = None
    for fname in ["master.parquet", "master_hourly_dataset.parquet", "master_hourly_dataset_fiat.parquet"]:
        p = PROCESSED_DATA_DIR / fname
        if p.exists():
            master_path = p
            break
    if master_path is None:
        raise FileNotFoundError(f"Could not find master parquet in {PROCESSED_DATA_DIR}")

    eps_path = None
    for fname in ["depeg_episodes.parquet", "depeg_episodes_fiat.parquet"]:
        p = PROCESSED_DATA_DIR / fname
        if p.exists():
            eps_path = p
            break
    if eps_path is None:
        raise FileNotFoundError(f"Could not find episodes parquet in {PROCESSED_DATA_DIR}")

    print(f"Loading master dataset from: {master_path}")
    print(f"Loading episode manifest from: {eps_path}")
    master = pd.read_parquet(master_path)
    episodes = pd.read_parquet(eps_path)
    return master, episodes


def purged_chronological_split(df: pd.DataFrame, episodes: pd.DataFrame, horizon_h: int, universal_purge_24h: bool = True):
    """
    Applies purged and embargoed chronological split across Train, Val, and Test.
    """
    df = df.copy()
    purge_hours = 24 if universal_purge_24h else horizon_h
    embargo_hours = EMBARGO_HOURS  # 24h

    # 1. Base time boundaries
    # Train: end minus purge_hours
    train_cutoff = TRAIN_END - pd.Timedelta(hours=purge_hours)
    # Val: start plus embargo_hours, end minus purge_hours
    val_embargoed_start = VAL_START + pd.Timedelta(hours=embargo_hours)
    val_cutoff = VAL_END - pd.Timedelta(hours=purge_hours)
    # Test: start plus embargo_hours
    test_embargoed_start = TEST_START + pd.Timedelta(hours=embargo_hours)

    train = df[df["hour"] <= train_cutoff].copy()
    val   = df[(df["hour"] >= val_embargoed_start) & (df["hour"] <= val_cutoff)].copy()
    test  = df[df["hour"] >= test_embargoed_start].copy()

    # 2. Episode-Aware Boundary Purging
    # Find episodes that touch boundary intervals [boundary - purge, boundary + embargo]
    boundary_windows = [
        (TRAIN_END - pd.Timedelta(hours=purge_hours), VAL_START + pd.Timedelta(hours=embargo_hours)),
        (VAL_END - pd.Timedelta(hours=purge_hours), TEST_START + pd.Timedelta(hours=embargo_hours)),
    ]

    straddling_episodes = set()
    for _, ep in episodes.iterrows():
        ep_start = ep["start"]
        ep_end = ep["end"]
        for b_start, b_end in boundary_windows:
            # Check overlap between [ep_start, ep_end] and [b_start, b_end]
            if not (ep_end < b_start or ep_start > b_end):
                straddling_episodes.add(ep["episode_id"])

    # Also drop any episode appearing across multiple partitions
    train_eps = set(train["episode_id"].dropna())
    val_eps   = set(val["episode_id"].dropna())
    test_eps  = set(test["episode_id"].dropna())

    overlap = (train_eps & val_eps) | (val_eps & test_eps) | (train_eps & test_eps)
    to_drop = straddling_episodes | overlap

    if to_drop:
        train = train[~train["episode_id"].isin(to_drop)].copy()
        val   = val[~val["episode_id"].isin(to_drop)].copy()
        test  = test[~test["episode_id"].isin(to_drop)].copy()

    return train, val, test


def main():
    master, episodes = load_dataset()

    print(f"\nExecuting Purged Chronological 3-Way Split (Embargo={EMBARGO_HOURS}h):")
    print(f"  Train:      2018-09-01 -> {TRAIN_END} (Purged last 24h)")
    print(f"  Validation: {VAL_START} -> {VAL_END} (Embargoed 24h, Purged last 24h)")
    print(f"  Test:       {TEST_START} -> 2025-12-31 (Embargoed 24h)")
    print("-" * 75)

    for h in FORECAST_HORIZONS_HOURS:
        train, val, test = purged_chronological_split(master, episodes, horizon_h=h, universal_purge_24h=True)

        # Save to processed directory
        train.to_parquet(PROCESSED_DATA_DIR / f"train_h{h}.parquet", index=False)
        val.to_parquet(PROCESSED_DATA_DIR / f"val_h{h}.parquet", index=False)
        test.to_parquet(PROCESSED_DATA_DIR / f"test_h{h}.parquet", index=False)

        col = f"label_h{h}"
        print(f"\n--- Horizon h = {h} Hours ---")
        for name, part in [("Train", train), ("Validation", val), ("Test", test)]:
            labelled = part[col].notna() if col in part.columns else pd.Series([False]*len(part))
            n_pos = int(part.loc[labelled, col].sum()) if labelled.sum() > 0 else 0
            pos_rate = (n_pos / labelled.sum()) * 100 if labelled.sum() > 0 else 0.0
            n_episodes = part["episode_id"].nunique()
            print(f"  {name:<10}: {len(part):>7,} rows | {labelled.sum():>7,} labelled | "
                  f"{n_pos:>5,} positive ({pos_rate:>5.2f}%) | {n_episodes:>3} episodes | "
                  f"{part['hour'].min():%Y-%m-%d} to {part['hour'].max():%Y-%m-%d}")

    print("\n[SUCCESS] Purged splits written to data/processed/ successfully.")


if __name__ == "__main__":
    main()
