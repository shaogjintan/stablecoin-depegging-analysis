"""Build hourly on-chain features from Dune Analytics exports.

Extracts the per-coin CSV for each coin in config.FIAT_COINS from
config.DUNE_ONCHAIN_ZIP into config.DUNE_ONCHAIN_DIR (data/raw/dune_onchain),
then builds features from those extracted files only.

EXPECTED CSV SCHEMA (per file):
    hour, tx_count, total_volume, active_senders, active_receivers
  - `hour` is a string like "2018-01-22 08:00:00.000 UTC"
  - `total_volume` is native-token units, already decimal-adjusted; it is
    renamed to `tx_volume` on load
  - There is NO `coin` column -- the coin is inferred from the filename,
    matched case-insensitively against config.FIAT_COINS (so "usdt.csv" ->
    "USDT").

TIER 2 CHANGE FEATURES
Raw on-chain levels (tx_count, ...) differ by orders of magnitude between
coins and identify the coin, so they are NOT usable as Tier 2 model features.
This script therefore also builds 1h/6h/24h log1p-difference change features:

    {metric}_logdiff_{k}h = log1p(x_t) - log1p(x_{t-k})

Why log1p differences rather than pct_change:
  - defined at zero: pct_change gives +/-inf when the base is 0 or NaN, which
    is common on quiet coins (PAX/TUSD hours with 1-2 txs);
  - comparable scale across coins whose levels differ ~100x;
  - consistent with prepare_data.py's rule: changes/ratios/deviations only,
    no raw levels.
Each coin is reindexed onto its full hourly grid first, so the lag window is
a true calendar window even when the Dune export has gaps; the diff is NaN
whenever either endpoint is unobserved. Raw levels are still kept in the
output for reference.
"""
from __future__ import annotations

import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
from data_processing.config import (
    DUNE_ONCHAIN_DIR,
    DUNE_ONCHAIN_ZIP,
    FIAT_COINS,
    ONCHAIN_FEATURES_PATH,
)

OUT_PATH = ONCHAIN_FEATURES_PATH

EXPECTED_COLUMNS = [
    "hour", "tx_count", "tx_volume", "active_senders", "active_receivers"
]

OPTIONAL_COLUMNS = [
    "total_volume_usd", "whale_tx_count", "median_transfer_usd", "transfer_gini"
]

CHANGE_METRICS = ["tx_count", "tx_volume", "active_senders", "active_receivers"]
CHANGE_LAGS = [1, 6, 24]
CHANGE_COLUMNS = [
    f"{metric}_logdiff_{lag}h"
    for metric in CHANGE_METRICS for lag in CHANGE_LAGS
]


def extract_fiat_csvs() -> dict[str, Path]:
    """Extract fiat-coin CSVs and return their paths keyed by coin."""
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


def build_change_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the twelve on-chain log1p-difference change features.

    Computed per coin on the coin's full hourly grid (calendar hours), so each
    lag spans a true 1h/6h/24h window even when the coin's series has gaps or
    starts late. The diff is NaN whenever either endpoint is unobserved.
    Returns `df` with its original columns followed by the new change columns.
    """
    metrics = [c for c in CHANGE_METRICS if c in df.columns]
    if not metrics or "coin" not in df.columns or "hour" not in df.columns:
        return df.copy()

    frames = []
    for coin, g in df.groupby("coin", sort=False):
        g = g.set_index("hour")
        if not g.index.is_monotonic_increasing:
            g = g.sort_index()
        g = g[~g.index.duplicated(keep="first")]
        grid = pd.date_range(g.index.min(), g.index.max(), freq="h", tz="UTC")
        g = g.reindex(grid)
        # Unobserved grid rows still belong to this coin.
        g["coin"] = coin
        log1p = np.log1p(g[metrics].astype(float))
        for m in metrics:
            for k in CHANGE_LAGS:
                g[f"{m}_logdiff_{k}h"] = log1p[m] - log1p[m].shift(k)
        frames.append(g.reset_index().rename(columns={"index": "hour"}))

    out = pd.concat(frames, ignore_index=True, sort=False)
    orig = list(df.columns)
    return out[orig + [c for c in CHANGE_COLUMNS if c in out.columns]]


def load_coin_csv(coin: str, path: Path) -> pd.DataFrame | None:
    """Load one export, normalise UTC hours, and validate required columns."""
    df = pd.read_csv(path)
    df = df.rename(columns={"total_volume": "tx_volume"})
    df["coin"] = coin
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        print(
            f"WARNING: {path.name} is missing expected columns "
            f"{missing_cols} -- skipping this file."
        )
        return None

    # "2018-01-22 08:00:00.000 UTC" -> tz-aware UTC timestamp. Strip the
    # trailing " UTC" (pandas' parser is inconsistent about combining a
    # literal "UTC" suffix with tz_localize), then localize explicitly,
    # tz-AWARE, matching hourly_stablecoin.parquet's `hour` column
    hour_str = df["hour"].astype(str).str.replace(" UTC", "", regex=False)
    df["hour"] = pd.to_datetime(
        hour_str, format="%Y-%m-%d %H:%M:%S.%f"
    ).dt.tz_localize("UTC")

    present_optional = [c for c in OPTIONAL_COLUMNS if c in df.columns]
    df = df[["coin", "hour", "tx_count", "tx_volume", "active_senders",
              "active_receivers"] + present_optional]
    print(f"  Loaded {path.name} -> {coin}: {len(df):,} coin-hour rows "
          f"(columns: {list(df.columns)}), "
          f"{df['hour'].min()} .. {df['hour'].max()}")
    return df


def main():
    """Build and write on-chain change features from retained source exports."""
    extracted = extract_fiat_csvs()
    if not extracted:
        raise RuntimeError(
            f"No CSVs in {DUNE_ONCHAIN_ZIP} matched a coin in "
            "config.FIAT_COINS."
        )

    frames = []
    for coin, path in sorted(extracted.items()):
        df = load_coin_csv(coin, path)
        if df is not None:
            frames.append(df)

    if not frames:
        raise RuntimeError(
            f"None of the extracted CSVs in {DUNE_ONCHAIN_DIR} were usable."
        )

    hourly = pd.concat(frames, ignore_index=True, sort=False)
    hourly = hourly.sort_values(["coin", "hour"]).reset_index(drop=True)
    hourly = build_change_features(hourly)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_parquet(OUT_PATH, index=False)

    matched_coins = set(hourly["coin"])
    missing_coins = sorted(set(FIAT_COINS) - matched_coins)
    print(f"\nBuilt {len(hourly):,} coin-hour rows -> {OUT_PATH}")
    print(f"Change columns added: {CHANGE_COLUMNS}")
    print(
        f"Coins WITH on-chain data "
        f"({len(matched_coins)}/{len(FIAT_COINS)}): {sorted(matched_coins)}"
    )
    if missing_coins:
        print(f"Coins WITHOUT on-chain data yet (will show as NaN on-chain "
              "features in the canonical dataset, price-only): "
              f"{missing_coins}")
    print(f"\n{hourly.groupby('coin')['hour'].agg(['min', 'max', 'count'])}")


if __name__ == "__main__":
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
