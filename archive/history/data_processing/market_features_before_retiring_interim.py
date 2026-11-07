"""Calculate canonical market features and retain interim source generation.

calendar_market_features() is the authoritative calendar-based calculation
used by the canonical dataset builder. It preserves missing observations,
uses backward calendar lags, and masks volume changes across venue switches.

The command-line writer and build_price_features()/build_exchange_features()
retain the historical interim formulas. The canonical builder records their
source-completeness flags before calculating corrected model features. They
are deliberately separate: substituting the canonical formulas would change
those historical flags and the frozen dataset hash.

Default output: data/interim/hourly_stablecoin_market_features.parquet.
Running this writer replaces that interim file; normal result reproduction
uses retained interim files and does not invoke this command.
"""
import numpy as np
import pandas as pd
from data_processing.config import HOURLY_MARKET_PATH, MARKET_FEATURES_PATH

OUT_PATH = MARKET_FEATURES_PATH


def build_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Reproduce historical interim price features and completeness inputs.

    Use the DefiLlama reference price and group by coin so calculations never
    cross coin boundaries.
    """
    df = df.sort_values(["coin", "hour"]).copy()
    g = df.groupby("coin")["price"]

    df["peg_deviation"] = df["price"] - 1.0
    df["peg_deviation_bps"] = df["peg_deviation"] * 1e4
    df["return_1h"] = g.pct_change()
    df["return_6h"] = g.pct_change(6)
    df["return_24h"] = g.pct_change(24)
    df["log_return_1h"] = np.log(df["price"] / g.shift(1))
    df["log_return_6h"] = np.log(df["price"] / g.shift(6))
    df["log_return_24h"] = np.log(df["price"] / g.shift(24))
    # realized vol = rolling std of 1h returns; min_periods = window so an
    # incomplete window (e.g. right after a coin's start) stays NaN rather
    # than a noisy estimate from a handful of hours
    ret_by_coin = df.groupby("coin")["return_1h"]
    df["realised_vol_6h"] = ret_by_coin.transform(
        lambda s: s.rolling(6, min_periods=6).std()
    )
    df["realised_vol_24h"] = ret_by_coin.transform(
        lambda s: s.rolling(24, min_periods=24).std()
    )

    return df[["coin", "hour", "peg_deviation", "peg_deviation_bps",
               "return_1h", "log_return_1h", "return_6h", "log_return_6h",
               "return_24h", "log_return_24h",
               "realised_vol_6h", "realised_vol_24h"]]


def build_exchange_features(df: pd.DataFrame) -> pd.DataFrame:
    """Reproduce historical interim exchange-volume features.

    volume_pct_change_1h/6h/24h are plain % changes -- but exchange volume is
    often near zero for the thinner pairs, so a single quiet hour can make
    the change spike to +-inf; those get replaced with NaN.

    Every change is additionally masked for the hours right after a venue
    switch (hours_since_venue_start < window + 1), since acquisition can
    splice from one exchange to another mid-series (e.g. PAX, TUSD, USDC),
    and a pct_change computed across that boundary compares two different
    venues' volume levels rather than a real change in trading activity.
    """
    df = df.sort_values(["coin", "hour"]).copy()

    vol_by_coin = df.groupby("coin")["volume"]
    pct_1h = vol_by_coin.pct_change().replace([np.inf, -np.inf], np.nan)
    pct_6h = vol_by_coin.pct_change(6).replace([np.inf, -np.inf], np.nan)
    pct_24h = vol_by_coin.pct_change(24).replace([np.inf, -np.inf], np.nan)

    df["volume_pct_change_1h"] = pct_1h.where(
        df["hours_since_venue_start"] >= 2
    )
    df["volume_pct_change_6h"] = pct_6h.where(
        df["hours_since_venue_start"] >= 7
    )
    df["volume_pct_change_24h"] = pct_24h.where(
        df["hours_since_venue_start"] >= 25
    )

    # taker_buy_ratio and trades_pct_change_1h were removed here: Bitfinex
    # (USDT/DAI) reports neither, so both were 100% NaN for those coins and a
    # pooled tree model would learn the missingness as a coin identifier.
    # See module docstring.

    df = df.rename(columns={"volume": "trading_volume"})
    return df[["coin", "hour", "trading_volume",
               "volume_pct_change_1h", "volume_pct_change_6h",
               "volume_pct_change_24h",
               "venue"]]



def calendar_market_features(df):
    """Calculate canonical market features on a complete per-coin calendar.

    Args:
        df: Rows containing coin, hour, price, trading_volume, and
            hours_since_venue_start. Hours denote source bar starts here.

    Returns:
        A coin/hour-sorted hourly table with canonical peg stress, log
        returns, rolling volatility, and venue-masked volume changes.
        Unrelated columns are preserved; new grid rows remain missing.

    Raises:
        ValueError: The input contains duplicate coin-hour keys.
    """
    df = df.sort_values(["coin", "hour"]).copy()
    if df.duplicated(["coin", "hour"]).any():
        raise ValueError("Duplicate coin-hour")
    frames = []
    for coin, g in df.groupby("coin", sort=False):
        g = g.set_index("hour").reindex(
            pd.date_range(g.hour.min(), g.hour.max(), freq="h")
        )
        g.index.name = "hour"
        g["coin"] = coin
        p, v = g.price, g.trading_volume
        g["peg_deviation"] = p - 1
        g["peg_stress"] = np.clip(1 - p, 0, .01)
        ret = p.pct_change(fill_method=None)
        for h in (1, 6, 24):
            g[f"log_return_{h}h"] = np.log(p / p.shift(h))
            change = v.div(v.shift(h)).sub(1).replace([np.inf, -np.inf], np.nan)
            # Missing venue provenance is not evidence of an uninterrupted venue.
            g[f"volume_pct_change_{h}h"] = change.where(
                g.hours_since_venue_start >= h + 1
            )
        for h in (6, 24):
            g[f"realised_vol_{h}h"] = ret.rolling(h, min_periods=h).std()
        frames.append(g.reset_index())
    return pd.concat(frames, ignore_index=True)


def main():
    # The acquired table keeps price-only hours with missing exchange volume.
    """Write the historical interim market table from retained hourly data."""
    df = pd.read_parquet(HOURLY_MARKET_PATH)

    price_feats = build_price_features(df)
    exch_feats = build_exchange_features(df)

    merged = price_feats.merge(exch_feats, on=["coin", "hour"], how="left")
    merged = merged.sort_values(["coin", "hour"]).reset_index(drop=True)
    merged = merged.drop(columns=["venue"])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(OUT_PATH, index=False)

    print(f"Built {len(merged):,} coin-hour rows -> {OUT_PATH}")
    cov = merged.groupby("coin").agg(
        rows=("hour", "size"),
        realised_vol_24h_coverage=(
            "realised_vol_24h", lambda s: s.notna().mean()
        ),
        volume_pct_change_24h_coverage=(
            "volume_pct_change_24h", lambda s: s.notna().mean()
        ),
    )
    print(cov.round(3).to_string())


if __name__ == "__main__":
    main()
