"""
Build hourly on-chain features from Dune Analytics exports.

Extracts the per-coin CSV for each coin in config.FIAT_COINS from
config.DUNE_ONCHAIN_ZIP into config.DUNE_ONCHAIN_DIR (data/raw/dune_onchain),
then builds features from those extracted files only.

EXPECTED CSV SCHEMA (per file):
    hour, tx_count, tx_volume, tx_count_pct_change, tx_volume_pct_change, active_senders, active_receivers
  - `hour` is a string like "2018-01-22 08:00:00.000 UTC"
  - `tx_volume` is native-token units, already decimal-adjusted
  - There is NO `coin` column -- the coin is inferred from the filename,
    matched case-insensitively against config.FIAT_COINS (so "usdt.csv" ->
    "USDT").
"""
from __future__ import annotations

import zipfile
from pathlib import Path, PurePosixPath

import pandas as pd
from config import DUNE_ONCHAIN_ZIP, DUNE_ONCHAIN_DIR, PROCESSED_DATA_DIR, FIAT_COINS

OUT_PATH = PROCESSED_DATA_DIR / "hourly_stablecoin_onchain_features.parquet"

EXPECTED_COLUMNS = ["hour", "tx_count", "tx_volume", "tx_count_pct_change", "tx_volume_pct_change", "active_senders", "active_receivers"]

OPTIONAL_COLUMNS = ["total_volume_usd", "whale_tx_count", "median_transfer_usd", "transfer_gini"]


def extract_fiat_csvs() -> dict[str, Path]:
    """Extract each fiat coin's CSV from the Dune zip into DUNE_ONCHAIN_DIR.
    Returns {coin: extracted_path}."""
    if not DUNE_ONCHAIN_ZIP.exists():
        raise FileNotFoundError(
            f"{DUNE_ONCHAIN_ZIP} not found -- run from the repo root, where "
            f"dune_onchain.zip lives."
        )

    DUNE_ONCHAIN_DIR.mkdir(parents=True, exist_ok=True)
    lower_to_coin = {c.lower(): c for c in FIAT_COINS}
    extracted = {}

    with zipfile.ZipFile(DUNE_ONCHAIN_ZIP) as zf:
        for name in zf.namelist():
            member = PurePosixPath(name)
            if member.parts[0] == "__MACOSX" or member.suffix.lower() != ".csv":
                continue
            coin = lower_to_coin.get(member.stem.lower())
            if coin is None:
                continue
            out = DUNE_ONCHAIN_DIR / member.name
            out.write_bytes(zf.read(name))
            extracted[coin] = out
            print(f"  Extracted {name} -> {out}")

    return extracted

def add_pct_change_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add pct-change features to the on-chain DataFrame."""
    df = df.copy()
    for col in ["tx_count", "tx_volume"]:
        if col not in df.columns:
            continue
        df[f"{col}_pct_change"] = df[col].pct_change()
    return df

def load_coin_csv(coin: str, path: Path) -> pd.DataFrame | None:
    df = pd.read_csv(path)
    df = df.rename(columns={"total_volume": "tx_volume"})
    df["coin"] = coin
    df = add_pct_change_features(df)
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        print(f"WARNING: {path.name} is missing expected columns {missing_cols} -- skipping this file.")
        return None

    # "2018-01-22 08:00:00.000 UTC" -> tz-aware UTC timestamp. Strip the
    # trailing " UTC" (pandas' parser is inconsistent about combining a
    # literal "UTC" suffix with tz_localize), then localize explicitly,
    # tz-AWARE, matching hourly_price.parquet's `hour` column
    hour_str = df["hour"].astype(str).str.replace(" UTC", "", regex=False)
    df["hour"] = pd.to_datetime(hour_str, format="%Y-%m-%d %H:%M:%S.%f").dt.tz_localize("UTC")

    present_optional = [c for c in OPTIONAL_COLUMNS if c in df.columns]
    df = df[["coin", "hour", "tx_count", "tx_volume", "active_senders",
              "active_receivers"] + present_optional]
    print(f"  Loaded {path.name} -> {coin}: {len(df):,} coin-hour rows "
          f"(columns: {list(df.columns)}), "
          f"{df['hour'].min()} .. {df['hour'].max()}")
    return df


def main():
    extracted = extract_fiat_csvs()
    if not extracted:
        raise RuntimeError(f"No CSVs in {DUNE_ONCHAIN_ZIP} matched a coin in config.FIAT_COINS.")

    frames = []
    for coin, path in sorted(extracted.items()):
        df = load_coin_csv(coin, path)
        if df is not None:
            frames.append(df)

    if not frames:
        raise RuntimeError(f"None of the extracted CSVs in {DUNE_ONCHAIN_DIR} were usable.")

    hourly = pd.concat(frames, ignore_index=True, sort=False)
    hourly = hourly.sort_values(["coin", "hour"]).reset_index(drop=True)

    hourly.to_parquet(OUT_PATH, index=False)

    matched_coins = set(hourly["coin"])
    missing_coins = sorted(set(FIAT_COINS) - matched_coins)
    print(f"\nBuilt {len(hourly):,} coin-hour rows -> {OUT_PATH}")
    print(f"Coins WITH on-chain data ({len(matched_coins)}/{len(FIAT_COINS)}): {sorted(matched_coins)}")
    if missing_coins:
        print(f"Coins WITHOUT on-chain data yet (will show as NaN on-chain "
              f"features throughout 04/05, price-only): {missing_coins}")
    print(f"\n{hourly.groupby('coin')['hour'].agg(['min', 'max', 'count'])}")


if __name__ == "__main__":
    main()
