"""
Build hourly MARKET features: returns, peg deviation, volatility and volume signals.

All rolling/pct-change features look backward only (no look-ahead). Volume-
change features are additionally masked across a venue switch (see 01's
`venue_switch` / `hours_since_venue_start` columns): a raw jump in exchange
volume right after a source switch reflects a change in data source, not a
change in the coin's actual trading activity, so those hours are set to NaN
rather than left in as a spurious signal.

OUTPUT: data/processed/hourly_market_features_fiat.parquet
  coin, hour, peg_deviation, peg_deviation_bps, return_1h, log_return_1h,
  return_6h, log_return_6h, return_24h, log_return_24h,
  realised_vol_6h, realised_vol_24h, trading_volume,
  volume_pct_change_1h, volume_pct_change_6h, volume_pct_change_24h,
  venue_switch

No trade-count or taker-buy features: Bitfinex (the source for USDT and DAI)
doesn't report either, so they'd be NaN for 100% of those two coins' rows. A
pooled tree model would learn that missingness pattern as a coin identifier
rather than a market signal, so trades_pct_change_1h and taker_buy_ratio are
not built. 
"""
import numpy as np
import pandas as pd
from config import PROCESSED_DATA_DIR

OUT_PATH = PROCESSED_DATA_DIR / "hourly_stablecoin_market_features.parquet"


def build_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Returns, peg deviation, and realised volatility from the DefiLlama
    reference price. Grouped by coin so nothing crosses a coin boundary."""
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
    df["realised_vol_6h"] = ret_by_coin.transform(lambda s: s.rolling(6, min_periods=6).std())
    df["realised_vol_24h"] = ret_by_coin.transform(lambda s: s.rolling(24, min_periods=24).std())

    return df[["coin", "hour", "peg_deviation", "peg_deviation_bps",
               "return_1h", "log_return_1h", "return_6h", "log_return_6h", "return_24h", "log_return_24h",
               "realised_vol_6h", "realised_vol_24h"]]


def build_exchange_features(df: pd.DataFrame) -> pd.DataFrame:
    """Volume signals from the exchange data.

    volume_pct_change_1h/6h/24h are plain % changes -- but exchange volume is
    often near zero for the thinner pairs, so a single quiet hour can make
    the change spike to +-inf; those get replaced with NaN.

    Every change is additionally masked for the hours right after a venue
    switch (hours_since_venue_start < window + 1), since 01's inner join can
    splice from one exchange to another mid-series (e.g. PAX, TUSD, USDC),
    and a pct_change computed across that boundary compares two different
    venues' volume levels rather than a real change in trading activity.
    """
    df = df.sort_values(["coin", "hour"]).copy()

    vol_by_coin = df.groupby("coin")["volume"]
    pct_1h = vol_by_coin.pct_change().replace([np.inf, -np.inf], np.nan)
    pct_6h = vol_by_coin.pct_change(6).replace([np.inf, -np.inf], np.nan)
    pct_24h = vol_by_coin.pct_change(24).replace([np.inf, -np.inf], np.nan)

    df["volume_pct_change_1h"] = pct_1h.where(df["hours_since_venue_start"] >= 2)
    df["volume_pct_change_6h"] = pct_6h.where(df["hours_since_venue_start"] >= 7)
    df["volume_pct_change_24h"] = pct_24h.where(df["hours_since_venue_start"] >= 25)

    # taker_buy_ratio and trades_pct_change_1h were removed here: Bitfinex
    # (USDT/DAI) reports neither, so both were 100% NaN for those coins and a
    # pooled tree model would learn the missingness as a coin identifier.
    # See module docstring.

    df = df.rename(columns={"volume": "trading_volume"})
    return df[["coin", "hour", "trading_volume",
               "volume_pct_change_1h", "volume_pct_change_6h", "volume_pct_change_24h",
               "venue"]]


def main():
    # 01 already inner-joins price and volume, so every row already has both
    # -- no separate left join needed here, unlike an earlier draft of this
    # script that assumed price and volume came from different tables.
    df = pd.read_parquet(PROCESSED_DATA_DIR / "hourly_stablecoin.parquet")

    price_feats = build_price_features(df)
    exch_feats = build_exchange_features(df)

    merged = price_feats.merge(exch_feats, on=["coin", "hour"], how="left")
    merged = merged.sort_values(["coin", "hour"]).reset_index(drop=True)
    merged = merged.drop(columns=["venue"])

    merged.to_parquet(OUT_PATH, index=False)

    print(f"Built {len(merged):,} coin-hour rows -> {OUT_PATH}")
    cov = merged.groupby("coin").agg(
        rows=("hour", "size"),
        realised_vol_24h_coverage=("realised_vol_24h", lambda s: s.notna().mean()),
        volume_pct_change_24h_coverage=("volume_pct_change_24h", lambda s: s.notna().mean()),
    )
    print(cov.round(3).to_string())


if __name__ == "__main__":
    main()