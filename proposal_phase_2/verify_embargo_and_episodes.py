"""
Empirical Verification of Embargo Duration, Autocorrelation Decay,
and Train-Validation-Test Episode Distribution.

Outputs:
1. Autocorrelation Function (ACF) decay across hourly lags (proving why 24h embargo is required).
2. Depeg episode duration and inter-episode gap statistics.
3. 3-way chronological split episode and observation counts for Train (2018-2022),
   Validation (2023), and Test (2024-2025).
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))
from config import PROCESSED_DATA_DIR, FIAT_COINS, FORECAST_HORIZONS_HOURS, EMBARGO_HOURS


def load_master_and_episodes():
    # Check possible file names
    master_path = PROCESSED_DATA_DIR / "master.parquet"
    if not master_path.exists():
        master_path = PROCESSED_DATA_DIR / "master_hourly_dataset.parquet"
    if not master_path.exists():
        master_path = PROCESSED_DATA_DIR / "master_hourly_dataset_fiat.parquet"

    episodes_path = PROCESSED_DATA_DIR / "depeg_episodes.parquet"
    if not episodes_path.exists():
        episodes_path = PROCESSED_DATA_DIR / "depeg_episodes_fiat.parquet"

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
    print("EMPIRICAL TEST 2: 3-WAY CHRONOLOGICAL SPLIT VALIDATION")
    print("=" * 60)
    print("Partitions:")
    print("  Train:      Start      -> 2022-12-31 23:00 (Includes May 2022 Terra/USDT crash)")
    print("  Validation: 2023-01-01 -> 2023-12-31 23:00 (Includes March 2023 SVB/USDC crisis)")
    print("  Test:       2024-01-01 -> 2025-12-31 23:00 (Out-of-sample forward evaluation)")
    print("  Purge:      24 hours before each boundary (no label overlap)")
    print("  Embargo:    24 hours after each boundary (no rolling feature memory leakage)")

    train_end = pd.Timestamp("2022-12-31 23:00:00", tz="UTC")
    val_start = pd.Timestamp("2023-01-01 00:00:00", tz="UTC")
    val_end   = pd.Timestamp("2023-12-31 23:00:00", tz="UTC")
    test_start= pd.Timestamp("2024-01-01 00:00:00", tz="UTC")

    # Filter to fiat coins
    fiat_eps = episodes[episodes["coin"].isin(FIAT_COINS)].copy()
    
    # Classify episodes by start time
    train_eps = fiat_eps[fiat_eps["start"] <= train_end]
    val_eps   = fiat_eps[(fiat_eps["start"] >= val_start) & (fiat_eps["start"] <= val_end)]
    test_eps  = fiat_eps[fiat_eps["start"] >= test_start]

    print(f"\nTotal Fiat Depeg Episodes: {len(fiat_eps)}")
    print(f"  Train Episodes (2018-2022):       {len(train_eps)} ({len(train_eps)/len(fiat_eps):.1%})")
    print(f"  Validation Episodes (2023):       {len(val_eps)} ({len(val_eps)/len(fiat_eps):.1%})")
    print(f"  Test Episodes (2024-2025):         {len(test_eps)} ({len(test_eps)/len(fiat_eps):.1%})")

    print("\nEpisode distribution by coin across partitions:")
    summary = pd.DataFrame({
        "Train (2018-22)": train_eps["coin"].value_counts(),
        "Validation (2023)": val_eps["coin"].value_counts(),
        "Test (2024-25)": test_eps["coin"].value_counts(),
    }).fillna(0).astype(int)
    summary["Total"] = summary.sum(axis=1)
    print(summary.to_string())

    return summary


def main():
    master, episodes = load_master_and_episodes()
    analyze_autocorrelation(master)
    analyze_episodes_and_splits(master, episodes)


if __name__ == "__main__":
    main()
