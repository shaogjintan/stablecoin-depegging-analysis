"""Merge hourly sources, identify depeg episodes, and build forecast labels.

Definitions (thresholds and horizons are configured in config.py):
  - One observation is one coin-hour. A downward depeg hour has price below
    1 - DEPEG_BAND; above-peg prices do not count as depegs.
  - An episode begins with at least DEPEG_MIN_DURATION_HOURS consecutive
    below-band hours. Its recovery window starts at the first above-band hour
    after the last qualifying dip and lasts DEPEG_GAP_HOURS hours. A qualifying
    dip beginning during recovery joins the episode and restarts the clock.
    Short dips do not reset it. Recovery hours, including missing-price hours,
    are assigned the episode ID but are not counted in n_hours. n_hours counts
    only hours in qualifying dips, excluding short dips within recovery. The
    episode table's end is one hour after the final qualifying below-band hour,
    marking the start of recovery.
  - At any above-band hour t with a known price, including during recovery,
    label_h{h} is 1 if a qualifying below-band dip run starts anywhere in
    (t, t+h], even when that run is bridged into an existing episode or the
    price recovers before t+h. It is 0 if no run starts and the outcome window
    is observed. A known onset makes the label 1 even if a later price is
    missing. Without a known onset, a missing future price makes it NaN.
    Below-band hours, missing current
    prices, and horizons past the coin's final observation have NaN labels.
  - prediction_eligible_market and prediction_eligible_common mark
    forecast origins with a known above-band price and complete features for
    the corresponding feature set. Recovery rows can be eligible; below-band
    rows cannot. Their horizon-specific variants additionally require the
    target label to be observed, for training/evaluation sample selection.
    Live prediction eligibility should not depend on future label availability.

Train/test splitting lives in splits.py; model-ready reporting is in
modelling/reporting/partition_report.py.
"""
import numpy as np
import pandas as pd
from config import (
    PROCESSED_DATA_DIR, DEPEG_BAND, DEPEG_MIN_DURATION_HOURS, DEPEG_GAP_HOURS,
    FORECAST_HORIZONS_HOURS, FIAT_COINS,
)


def build_full_hourly_grid(pv: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for coin in sorted(pv["coin"].unique()):
        p = pv[pv["coin"] == coin]
        grid = pd.DataFrame({"hour": pd.date_range(p["hour"].min(), p["hour"].max(), freq="h", tz="UTC")})
        grid["coin"] = coin
        frames.append(grid)
    return pd.concat(frames, ignore_index=True)


def merge_sources(pv: pd.DataFrame, onchain: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    keys = ["coin", "hour"]
    for name, source in [("price/volume", pv), ("on-chain", onchain), ("market", market)]:
        duplicates = source.duplicated(keys, keep=False)
        if duplicates.any():
            examples = source.loc[duplicates, keys].drop_duplicates().head(5).to_dict("records")
            raise ValueError(f"{name} data has duplicate coin-hour keys; examples: {examples}")

    grid = build_full_hourly_grid(pv)
    pv = pv.drop(columns=["volume"]) # there exists trading volume column already in market features, so drop this one to avoid duplicate column name
    # drop raw trade count and taker-buy volume: Bitfinex (USDT/DAI) reports
    # neither, so they're 100% NaN for those coins and a pooled tree model
    # would learn the missingness as a coin identifier (see 02a docstring)
    merged = grid.merge(pv, on=keys, how="left", validate="one_to_one")
    merged = merged.merge(onchain, on=keys, how="left", validate="one_to_one")
    merged = merged.merge(market, on=keys, how="left", validate="one_to_one")
    merged = merged.sort_values(["coin", "hour"]).reset_index(drop=True)
    return merged


def flag_episodes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Use the raw price threshold shared with prediction eligibility. Exactly
    # $0.99 is on-band; only lower prices count as downward depegs.
    peg_threshold = 1 - DEPEG_BAND
    df["outside_band"] = df["price"] < peg_threshold
    df["episode_id"] = pd.array([None] * len(df), dtype="object")
    df["qualifying_depeg_hour"] = False
    df["qualifying_dip_start"] = False
    df["episode_end_hour"] = pd.Series(pd.NaT, index=df.index, dtype=df["hour"].dtype)
    for coin, g in df.groupby("coin"):
        idx = g.index
        outside = g["outside_band"].fillna(False).to_numpy()
        # First collect only qualifying consecutive below-band runs.
        runs = []
        i = 0
        while i < len(outside):
            if not outside[i]:
                i += 1
                continue

            # Find the complete consecutive below-band run before deciding
            # whether it qualifies. A short run after an episode is ignored
            # completely and therefore cannot move last_out.
            run_start = i
            while i < len(outside) and outside[i]:
                i += 1
            run_end = i - 1
            run_length = run_end - run_start + 1
            if run_length >= DEPEG_MIN_DURATION_HOURS:
                runs.append((run_start, run_end))
                df.loc[idx[run_start:run_end + 1], "qualifying_depeg_hour"] = True
                df.loc[idx[run_start], "qualifying_dip_start"] = True

        # Cluster qualifying runs when the next run starts within the recovery
        # window, measured from the first above-band hour after the prior run.
        # Extend each cluster through its full recovery window, even if no later
        # qualifying run occurs. Recovery rows retain the event ID, but remain
        # usable forecast origins whenever their price is above the band.
        clusters = []
        for run_start, run_end in runs:
            if clusters:
                prev_start, prev_end, prev_last_out = clusters[-1]
                recovery_start = prev_last_out + 1
                if run_start - recovery_start <= DEPEG_GAP_HOURS:
                    clusters[-1] = (prev_start, run_end, run_end)
                    continue
            clusters.append((run_start, run_end, run_end))

        for episode_num, (episode_start, last_out, _) in enumerate(clusters, start=1):
            # The first above-band hour is last_out + 1; include 24 hours
            # from that anchor, making the next independent onset eligible
            # only after the recovery window has elapsed.
            episode_end = min(last_out + DEPEG_GAP_HOURS + 1, len(outside) - 1)
            df.loc[idx[episode_start:episode_end + 1], "episode_id"] = f"{coin}_{episode_num}"
            # The manifest's end marks the first hour after the final
            # qualifying dip, when recovery begins (not the recovery-window end).
            df.loc[idx[last_out], "episode_end_hour"] = (
                g.iloc[last_out]["hour"] + pd.Timedelta(hours=1)
            )

    df["in_episode"] = df["episode_id"].notna()
    return df


def build_episode_table(df: pd.DataFrame) -> pd.DataFrame:
    eps = (
        df[df["qualifying_depeg_hour"]]
        .groupby(["coin", "episode_id"])
        .agg(start=("hour", "min"), end=("episode_end_hour", "max"), n_hours=("qualifying_depeg_hour", "sum"),
             min_price=("price", "min"), max_price=("price", "max"))
        .reset_index()
        .sort_values("start")
    )
    return eps


def build_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Label whether any qualifying dip run starts within each future horizon.

    A dip that starts within the window remains a positive outcome even if its
    price recovers before the horizon endpoint.
    """
    df = df.copy()
    dip_starts = (
        df.loc[df["qualifying_dip_start"]]
        .groupby("coin")["hour"]
        .apply(list)
        .to_dict()
    )
    max_hour_per_coin = df.groupby("coin")["hour"].max().to_dict()

    for h in FORECAST_HORIZONS_HOURS:
        col = f"label_h{h}"
        df[col] = np.nan
        for coin in FIAT_COINS:
            if coin not in df["coin"].unique():
                continue
            starts_arr = np.array(sorted(dip_starts.get(coin, [])))
            mask_coin = df["coin"] == coin
            hours = df.loc[mask_coin, "hour"].to_numpy()
            prices = df.loc[mask_coin, "price"].to_numpy()
            last_hour = max_hour_per_coin[coin]
            labels = np.full(len(hours), np.nan)
            for i, t in enumerate(hours):
                window_end = t + pd.Timedelta(hours=h)
                # right-censored: the lookahead window runs past the last
                # observed hour, so "no episode found" can't be trusted as a
                # true negative -- leave it NaN rather than defaulting to 0.
                if window_end > last_hour:
                    continue
                # A known qualifying-dip onset (including a bridged dip) makes
                # this positive even if a later price in the window is missing.
                hit = np.any((starts_arr > t) & (starts_arr <= window_end)) if len(starts_arr) else False
                if hit:
                    labels[i] = 1.0
                    continue

                # Without a known onset, any missing future price could hide
                # a qualifying dip run, including within a recovery window.
                future_slice = slice(i + 1, i + h + 1)
                if pd.isna(prices[future_slice]).any():
                    continue
                labels[i] = 0.0
            df.loc[mask_coin, col] = labels
        # Forecast origins must have an above-threshold observed price.
        # Recovery rows can be origins; below-threshold rows cannot.
        df.loc[df["outside_band"], col] = np.nan
        # no observed price at t -> can't tell if t is above the threshold
        df.loc[df["price"].isna(), col] = np.nan

    return df

def flag_feature_samples(
    df: pd.DataFrame,
    market_features: list,
    onchain_features: list
) -> pd.DataFrame:
    df = df.copy()
    required_features = market_features + onchain_features
    missing = [col for col in required_features if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    df["market_sample"] = df[market_features].notna().all(axis=1)
    # `common_sample` is the established name for rows complete across both
    # feature groups; keep it as the sole market + on-chain sample column.
    df["common_sample"] = df[required_features].notna().all(axis=1)
    return df


def flag_prediction_eligibility(df: pd.DataFrame) -> pd.DataFrame:
    """Flag prediction origins separately for market and market+on-chain.

    Base flags depend only on information available at the row's hour.
    Horizon-specific flags additionally require an observed label, making them
    convenient for training/evaluation sample selection.
    """
    df = df.copy()
    above_threshold = df["price"].ge(1 - DEPEG_BAND)
    feature_samples = {
        "market": "market_sample",
        "common": "common_sample",
    }
    for model, sample_col in feature_samples.items():
        eligible_col = f"prediction_eligible_{model}"
        df[eligible_col] = df[sample_col] & above_threshold

    for h in FORECAST_HORIZONS_HOURS:
        label_col = f"label_h{h}"
        if label_col not in df.columns:
            raise ValueError(f"Missing label column: {label_col}")
        for model in feature_samples:
            eligible_col = f"prediction_eligible_{model}_h{h}"
            df[eligible_col] = df[f"prediction_eligible_{model}"] & df[label_col].notna()
    return df

def main():
    price = pd.read_parquet(PROCESSED_DATA_DIR / "hourly_stablecoin.parquet")
    price = price.rename(columns={"timestamp": "hour"})
    onchain = pd.read_parquet(PROCESSED_DATA_DIR / "hourly_stablecoin_onchain_features.parquet")
    market = pd.read_parquet(PROCESSED_DATA_DIR / "hourly_stablecoin_market_features.parquet")

    merged = merge_sources(price, onchain, market)
    merged = flag_episodes(merged)
    episodes = build_episode_table(merged)
    merged = build_labels(merged)
    # Match the model's Tier 1 inputs in modelling/prepare_data.py. The
    # derived peg_stress feature is a deterministic transform of peg_deviation.
    market_features = [
        "peg_deviation", "log_return_1h", "log_return_6h", "log_return_24h",
        "realised_vol_6h", "realised_vol_24h", "volume_pct_change_1h",
        "volume_pct_change_6h", "volume_pct_change_24h",
    ]
    onchain_metrics = ["tx_count","tx_volume","active_senders","active_receivers"]
    onchain_lags = [1, 6, 24]
    onchain_features = [f"{metric}_logdiff_{lag}h" for metric in onchain_metrics for lag in onchain_lags]

    merged = flag_feature_samples(merged, market_features, onchain_features)
    merged = flag_prediction_eligibility(merged)
    coverage = merged.groupby("coin").agg(
        total_hours=("hour", "size"),
        market_hours=("market_sample", "sum"),
        common_hours=("common_sample", "sum"),
    )
    coverage["market_retained_pct"] = 100 * coverage["market_hours"] / coverage["total_hours"]
    coverage["common_retained_pct"] = 100 * coverage["common_hours"] / coverage["total_hours"]
    print("\nFEATURE SAMPLE COVERAGE")
    print(coverage.round(2).to_string())

    merged.to_parquet(PROCESSED_DATA_DIR / "master.parquet", index=False)
    merged.to_csv(PROCESSED_DATA_DIR / "master.csv", index=False)
    episodes.to_parquet(PROCESSED_DATA_DIR / "depeg_episodes.parquet", index=False)
    episodes.to_csv(PROCESSED_DATA_DIR / "depeg_episodes.csv", index=False)

    print(f"Master dataset: {len(merged):,} coin-hour rows")
    print(f"Depeg episodes found: {len(episodes)}")
    print(episodes)

if __name__ == "__main__":
    main()
