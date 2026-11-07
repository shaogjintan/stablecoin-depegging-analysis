"""Calculate calendar-based market features for interim and final tables.

Missing hours remain missing; lags refer to calendar hours. The command-line
writer uses the same calculation as the dataset builder. Historical formulas
are preserved under delete/archive_cleanup/history/data_processing/.

Run ``python -m data_processing.build_market_features`` to rebuild
data/interim/hourly_stablecoin_market_features.parquet from the retained
data/interim/hourly_stablecoin.parquet. This replaces the interim feature
file with the current calculation; it does not fetch data or train models.

The retained historical interim file can differ in simple-return diagnostics
and completeness. Verify frozen model evidence using its recorded input
hashes; a current rebuild need not reproduce historical file bytes. Normal
report reproduction reads retained inputs without running this writer.
"""
import numpy as np
import pandas as pd
from data_processing.config import HOURLY_MARKET_PATH, MARKET_FEATURES_PATH

OUT_PATH = MARKET_FEATURES_PATH


def calendar_market_features(df):
    """Calculate market features on a complete per-coin calendar.

    Args:
        df: Rows containing coin, hour, price, trading_volume, and
            hours_since_venue_start. Hours denote source bar starts here.

    Returns:
        A coin/hour-sorted hourly table with peg stress, log
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
        g["peg_deviation_bps"] = g["peg_deviation"] * 1e4
        g["peg_stress"] = np.clip(1 - p, 0, .01)
        ret = p.pct_change(fill_method=None)
        for h in (1, 6, 24):
            g[f"return_{h}h"] = p.pct_change(h, fill_method=None)
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
    """Write current calendar features from retained hourly observations."""
    df = pd.read_parquet(HOURLY_MARKET_PATH)
    df = df.rename(columns={"volume": "trading_volume"})
    features = calendar_market_features(df)
    columns = ["coin", "hour", "peg_deviation", "peg_deviation_bps",
               "return_1h", "log_return_1h", "return_6h", "log_return_6h",
               "return_24h", "log_return_24h", "realised_vol_6h",
               "realised_vol_24h", "trading_volume", "volume_pct_change_1h",
               "volume_pct_change_6h", "volume_pct_change_24h", "peg_stress"]
    merged = features[columns]

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
