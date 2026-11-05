"""
Empirical Verification of Embargo Duration, Autocorrelation Decay,
and Train-Validation-Test Episode Distribution.

Outputs:
1. Autocorrelation Function (ACF) decay across hourly lags (proving why 24h embargo is required).
2. Depeg episode duration and inter-episode gap statistics.
3. Expanding-window CV fold checks (see splits.py): train/test gap and shared
   episodes per fold, and out-of-sample episode counts by coin.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))
from config import (PROCESSED_DATA_DIR, FIAT_COINS, FORECAST_HORIZONS_HOURS, EMBARGO_HOURS,
                    PURGE_HOURS, OOS_START_YEAR, OOS_END_YEAR)
from splits import expanding_window_folds


def load_master_and_episodes():
    master_path = PROCESSED_DATA_DIR / "master.parquet"
    episodes_path = PROCESSED_DATA_DIR / "depeg_episodes.parquet"

    print(f"Loading master from: {master_path}")
    print(f"Loading episodes from: {episodes_path}")

    master = pd.read_parquet(master_path)
    episodes = pd.read_parquet(episodes_path)
    return master, episodes


def analyze_autocorrelation(master: pd.DataFrame):
    print("\n" + "=" * 60)
    print("EMPIRICAL TEST 1: AUTOCORRELATION DECAY OF FEATURES")
    print("=" * 60)
    
    lags = [1, 2, 4, 6, 12, 18, 24, 36, 48]
    acf_results = []

    for coin in ["USDT", "USDC", "DAI"]:
        cdf = master[master["coin"] == coin].sort_values("hour").copy()
        if len(cdf) == 0:
            continue

        # Look at 24h rolling price return and 24h transaction difference
        price_col = "close" if "close" in cdf.columns else "price"
        ret_24h = np.log(cdf[price_col] / cdf[price_col].shift(24))
        
        row_ret = {"coin": coin, "feature": "24h_Price_Return"}
        for lag in lags:
            val = ret_24h.autocorr(lag=lag)
            row_ret[f"lag_{lag}h"] = round(float(val), 4) if pd.notna(val) else np.nan
        acf_results.append(row_ret)

        if "tx_count" in cdf.columns and cdf["tx_count"].notna().sum() > 0:
            tx_diff = np.log1p(cdf["tx_count"]) - np.log1p(cdf["tx_count"].shift(24))
            row_tx = {"coin": coin, "feature": "24h_Tx_Count_Diff"}
            for lag in lags:
                val = tx_diff.autocorr(lag=lag)
                row_tx[f"lag_{lag}h"] = round(float(val), 4) if pd.notna(val) else np.nan
            acf_results.append(row_tx)

    acf_df = pd.DataFrame(acf_results)
    print(acf_df.to_string(index=False))
    return acf_df


def analyze_episodes_and_splits(master: pd.DataFrame, episodes: pd.DataFrame):
    print("\n" + "=" * 60)
    print("EMPIRICAL TEST 2: EXPANDING-WINDOW CV FOLD VALIDATION")
    print("=" * 60)
    print(f"For each year Y in {OOS_START_YEAR}-{OOS_END_YEAR}: train on everything before Y, test on Y")
    print(f"  Purge:   {PURGE_HOURS} hours before each boundary (no label overlap)")
    print(f"  Embargo: {EMBARGO_HOURS} hours after each boundary (no rolling feature memory leakage)")

    folds = expanding_window_folds(master, episodes)

    # Leakage checks: train ends >= purge before test starts, and no episode
    # has rows on both sides of a fold's boundary.
    print("\nLeakage checks per fold:")
    for f in folds:
        gap_h = (f["test"]["hour"].min() - f["train"]["hour"].max()).total_seconds() / 3600
        shared = set(f["train"]["episode_id"].dropna()) & set(f["test"]["episode_id"].dropna())
        ok = gap_h > PURGE_HOURS + EMBARGO_HOURS - 1 and not shared
        print(f"  Fold {f['year']}: train->test gap {gap_h:.0f}h, "
              f"shared episodes {len(shared)} -> {'OK' if ok else 'FAIL'}")

    # Out-of-sample episodes per fold, by coin (episodes with rows in the test year)
    print("\nOut-of-sample episodes per fold, by coin:")
    summary = pd.DataFrame({
        f"Test {f['year']}": (episodes[episodes["episode_id"].isin(f["test"]["episode_id"].dropna())]
                              ["coin"].value_counts())
        for f in folds
    }).reindex(FIAT_COINS).fillna(0).astype(int)
    summary["Pooled OOS"] = summary.sum(axis=1)
    print(summary.to_string())
    print(f"\nTotal episodes scored out of sample: {summary['Pooled OOS'].sum()} of {len(episodes)}")

    return summary


def main():
    master, episodes = load_master_and_episodes()
    analyze_autocorrelation(master)
    analyze_episodes_and_splits(master, episodes)


if __name__ == "__main__":
    main()
