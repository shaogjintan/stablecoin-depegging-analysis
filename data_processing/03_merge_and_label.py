"""
Merge hourly price & volume data + market features + on-chain features, 
define depeg episodes explicitly
and build forecasting labels for each horizon h.

Definitions (edit in config.py, not here):
  - One observation = one coin-hour.
  - "Depegged" hour: price < 0.99 
    Downward only -- trading above the peg is not counted as a depeg.
  - "Depeg episode": starts with >= DEPEG_MIN_DURATION_HOURS (default: 2)
    consecutive depegged hours. After that, any dip (of any length) within
    DEPEG_GAP_HOURS (default: 24) of the episode's last depegged hour is
    bridged into it, so same-coin episodes are always separated by more than
    DEPEG_GAP_HOURS above the band. Bridged gap hours belong to the episode
    (excluded from labels) but don't count towards n_hours, which is hours
    spent below the band.
  - Label for horizon h, at a NORMAL (non-depegged) hour t: 1 if a new
    depeg episode starts anywhere in (t, t+h], else 0. Hours already inside
    an ongoing episode, hours with no price, and hours whose look-ahead runs
    past the coin's last observation get NaN.

Train/test splitting and the embargo flag live in 05_split_train_test.py.
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
    grid = build_full_hourly_grid(pv)
    pv = pv.drop(columns=["volume"]) # there exists trading volume column already in market features, so drop this one to avoid duplicate column name
    merged = grid.merge(pv, on=["coin", "hour"], how="left")
    merged = merged.merge(onchain, on=["coin", "hour"], how="left")
    merged = merged.merge(market, on=["coin", "hour"], how="left")
    merged = merged.sort_values(["coin", "hour"]).reset_index(drop=True)
    merged["is_stablecoin"] = merged["coin"].isin(FIAT_COINS)
    return merged


def flag_episodes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["outside_band"] = df["peg_deviation"] < -DEPEG_BAND  # downward depegs only
    df.loc[~df["is_stablecoin"], "outside_band"] = False  # never flag auxiliary coins

    df["episode_id"] = pd.array([None] * len(df), dtype="object")
    for coin, g in df[df["is_stablecoin"]].groupby("coin"):
        idx = g.index
        outside = g["outside_band"].fillna(False).to_numpy()
        # identify runs of True values. Once a run qualifies as an episode
        # (>= DEPEG_MIN_DURATION_HOURS consecutive below-band hours), any dip
        # within DEPEG_GAP_HOURS of its last below-band hour is bridged into it
        # (inactivity gap to merge continuous ongoing crash)
        run_id = np.zeros(len(outside), dtype=int)
        current = 0
        last_out = None
        streak = 0          # consecutive below-band hours ending at last_out
        qualified = False   # current run has had a >= MIN-hour consecutive dip
        valid_runs = set()
        for i in range(len(outside)):
            if outside[i]:
                if last_out == i - 1:
                    streak += 1
                elif qualified and i - last_out - 1 <= DEPEG_GAP_HOURS:
                    run_id[last_out + 1:i] = current  # fill the bridged gap
                    streak = 1
                else:
                    current += 1
                    streak = 1
                    qualified = False
                run_id[i] = current
                last_out = i
                if streak >= DEPEG_MIN_DURATION_HOURS:
                    qualified = True
                    valid_runs.add(current)
        for i, r in enumerate(run_id):
            if r in valid_runs:
                df.loc[idx[i], "episode_id"] = f"{coin}_{r}"

    df["in_episode"] = df["episode_id"].notna()
    return df


def build_episode_table(df: pd.DataFrame) -> pd.DataFrame:
    eps = (
        df[df["in_episode"]]
        .groupby(["coin", "episode_id"])
        .agg(start=("hour", "min"), end=("hour", "max"), n_hours=("outside_band", "sum"),
             min_price=("price", "min"), max_price=("price", "max"))
        .reset_index()
        .sort_values("start")
    )
    return eps


def build_labels(df: pd.DataFrame, episodes: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ep_starts = episodes.groupby("coin")["start"].apply(list).to_dict()
    max_hour_per_coin = df.groupby("coin")["hour"].max().to_dict()

    for h in FORECAST_HORIZONS_HOURS:
        col = f"label_h{h}"
        df[col] = np.nan
        for coin in FIAT_COINS:
            if coin not in df["coin"].unique():
                continue
            starts_arr = np.array(sorted(ep_starts.get(coin, [])))
            mask_coin = df["coin"] == coin
            hours = df.loc[mask_coin, "hour"].to_numpy()
            last_hour = max_hour_per_coin[coin]
            labels = np.full(len(hours), np.nan)
            for i, t in enumerate(hours):
                window_end = t + pd.Timedelta(hours=h)
                # right-censored: the lookahead window runs past the last
                # observed hour, so "no episode found" can't be trusted as a
                # true negative -- leave it NaN rather than defaulting to 0.
                if window_end > last_hour:
                    continue
                hit = np.any((starts_arr > t) & (starts_arr <= window_end)) if len(starts_arr) else False
                labels[i] = 1.0 if hit else 0.0
            df.loc[mask_coin, col] = labels
        # exclude hours already inside an active episode from this label
        df.loc[df["in_episode"], col] = np.nan
        # no observed price at t -> can't tell if t is normal, so no label
        df.loc[df["price"].isna(), col] = np.nan

    return df


def main():
    price = pd.read_parquet(PROCESSED_DATA_DIR / "hourly_stablecoin.parquet")
    price = price.rename(columns={"timestamp": "hour"})
    onchain = pd.read_parquet(PROCESSED_DATA_DIR / "hourly_stablecoin_onchain_features.parquet")
    market = pd.read_parquet(PROCESSED_DATA_DIR / "hourly_stablecoin_market_features.parquet")

    merged = merge_sources(price, onchain, market)
    merged = flag_episodes(merged)
    episodes = build_episode_table(merged)
    merged = build_labels(merged, episodes)

    merged.to_parquet(PROCESSED_DATA_DIR / "master.parquet", index=False)
    merged.to_csv(PROCESSED_DATA_DIR / "master.csv", index=False)
    episodes.to_parquet(PROCESSED_DATA_DIR / "depeg_episodes.parquet", index=False)
    episodes.to_csv(PROCESSED_DATA_DIR / "depeg_episodes.csv", index=False)

    print(f"Master dataset: {len(merged):,} coin-hour rows")
    print(f"Depeg episodes found: {len(episodes)}")
    print(episodes)


if __name__ == "__main__":
    main()