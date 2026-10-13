"""
Build hourly on-chain features from Dune Analytics exports.

EXPECTED CSV SCHEMA (per file):
    hour, tx_count, total_volume, active_senders, active_receivers
  - `hour` is a string like "2018-01-22 08:00:00.000 UTC"
  - `total_volume` is native-token units, already decimal-adjusted
  - There is NO `coin` column -- the coin is inferred from the filename,
    matched case-insensitively against config.ALL_COINS (so "usdt.csv" ->
    "USDT", "crvusd.csv" -> "crvUSD").
"""
from __future__ import annotations

import pandas as pd
from config import DUNE_ONCHAIN_DIR, PROCESSED_DATA_DIR, ALL_COINS

OUT_PATH = PROCESSED_DATA_DIR / "hourly_onchain_features.parquet"

EXPECTED_COLUMNS = ["hour", "tx_count", "total_volume", "active_senders", "active_receivers"]

OPTIONAL_COLUMNS = ["total_volume_usd", "whale_tx_count", "median_transfer_usd", "transfer_gini"]


def main():
    if not DUNE_ONCHAIN_DIR.exists():
        raise FileNotFoundError(
            f"{DUNE_ONCHAIN_DIR} does not exist -- create it and drop your "
            f"per-coin Dune CSVs there (e.g. usdt.csv, dai.csv, ust.csv)."
        )

    lower_to_coin = {c.lower(): c for c in ALL_COINS}
    csv_files = sorted(DUNE_ONCHAIN_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSVs found in {DUNE_ONCHAIN_DIR}.")

    frames = []
    matched_coins = set()
    unmatched_files = []

    for path in csv_files:
        coin = lower_to_coin.get(path.stem.lower())
        if coin is None:
            unmatched_files.append(path.name)
            continue

        df = pd.read_csv(path)
        missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
        if missing_cols:
            print(f"WARNING: {path.name} is missing expected columns {missing_cols} -- skipping this file.")
            continue

        # "2018-01-22 08:00:00.000 UTC" -> tz-aware UTC timestamp. Strip the
        # trailing " UTC" (pandas' parser is inconsistent about combining a
        # literal "UTC" suffix with tz_localize), then localize explicitly,
        # tz-AWARE, matching hourly_price.parquet's `hour` column 
        hour_str = df["hour"].astype(str).str.replace(" UTC", "", regex=False)
        df["hour"] = pd.to_datetime(hour_str, format="%Y-%m-%d %H:%M:%S.%f").dt.tz_localize("UTC")

        df["coin"] = coin
        present_optional = [c for c in OPTIONAL_COLUMNS if c in df.columns]
        df = df[["coin", "hour", "tx_count", "total_volume", "active_senders",
                  "active_receivers"] + present_optional]
        frames.append(df)
        matched_coins.add(coin)
        print(f"  Loaded {path.name} -> {coin}: {len(df):,} coin-hour rows "
              f"(columns: {list(df.columns)}), "
              f"{df['hour'].min()} .. {df['hour'].max()}")

    if unmatched_files:
        print(f"\nFiles in {DUNE_ONCHAIN_DIR} that didn't match any coin in "
              f"config.ALL_COINS (skipped): {unmatched_files}")

    if not frames:
        raise RuntimeError(f"No usable CSVs matched a coin in config.ALL_COINS in {DUNE_ONCHAIN_DIR}.")

    hourly = pd.concat(frames, ignore_index=True, sort=False)
    hourly = hourly.sort_values(["coin", "hour"]).reset_index(drop=True)

    hourly.to_parquet(OUT_PATH, index=False)

    missing_coins = sorted(set(ALL_COINS) - matched_coins)
    print(f"\nBuilt {len(hourly):,} coin-hour rows -> {OUT_PATH}")
    print(f"Coins WITH on-chain data ({len(matched_coins)}/{len(ALL_COINS)}): {sorted(matched_coins)}")
    if missing_coins:
        print(f"Coins WITHOUT on-chain data yet (will show as NaN on-chain "
              f"features throughout 04/05, price-only): {missing_coins}")
    print(f"\n{hourly.groupby('coin')['hour'].agg(['min', 'max', 'count'])}")


if __name__ == "__main__":
    main()