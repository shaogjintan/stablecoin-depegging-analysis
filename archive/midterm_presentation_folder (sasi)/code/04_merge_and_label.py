"""
Merge hourly price + on-chain features, define depeg episodes explicitly,
and build forecasting labels for each horizon h.

Definitions (edit in config.py, not here):
  - One observation = one coin-hour.
  - "Depegged" hour: |close - 1.00| > DEPEG_BAND (default: outside $0.99-$1.01).
  - "Depeg episode": a maximal run of consecutive depegged hours whose length
    is >= DEPEG_MIN_DURATION_HOURS (default: 2). 
  - Label for horizon h, at a NORMAL (non-depegged) hour t: 1 if a new
    depeg episode starts anywhere in (t, t+h], else 0. Hours already inside
    an ongoing episode are excluded from the training/eval set for that
    horizon.
"""
import numpy as np
import pandas as pd
from config import (
    PROCESSED_DATA_DIR, DEPEG_BAND, DEPEG_MIN_DURATION_HOURS,
    FORECAST_HORIZONS_HOURS, EMBARGO_HOURS, COINS, AUXILIARY_COINS,
)


def build_full_hourly_grid(price: pd.DataFrame, onchain: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for coin in sorted(set(price["coin"]) | set(onchain["coin"])):
        p = price[price["coin"] == coin]
        o = onchain[onchain["coin"] == coin]
        if p.empty and o.empty:
            continue
        lo = min(p["hour"].min() if not p.empty else pd.Timestamp.max.tz_localize("UTC"),
                  o["hour"].min() if not o.empty else pd.Timestamp.max.tz_localize("UTC"))
        hi = max(p["hour"].max() if not p.empty else pd.Timestamp.min.tz_localize("UTC"),
                  o["hour"].max() if not o.empty else pd.Timestamp.min.tz_localize("UTC"))
        grid = pd.DataFrame({"hour": pd.date_range(lo, hi, freq="h", tz="UTC")})
        grid["coin"] = coin
        frames.append(grid)
    return pd.concat(frames, ignore_index=True)


def merge_sources(price: pd.DataFrame, onchain: pd.DataFrame) -> pd.DataFrame:
    grid = build_full_hourly_grid(price, onchain)
    merged = grid.merge(price, on=["coin", "hour"], how="left")
    merged = merged.merge(onchain, on=["coin", "hour"], how="left")
    merged = merged.sort_values(["coin", "hour"]).reset_index(drop=True)
    merged["is_stablecoin"] = merged["coin"].isin(COINS)
    return merged


def flag_episodes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["price_dev"] = df["close"] - 1.0
    df["outside_band"] = df["price_dev"].abs() > DEPEG_BAND
    df.loc[~df["is_stablecoin"], "outside_band"] = False  # never flag auxiliary coins

    df["episode_id"] = pd.array([None] * len(df), dtype="object")
    for coin, g in df[df["is_stablecoin"]].groupby("coin"):
        idx = g.index
        outside = g["outside_band"].fillna(False).to_numpy()
        # identify runs of consecutive True values
        run_id = np.zeros(len(outside), dtype=int)
        current = 0
        for i in range(len(outside)):
            if outside[i]:
                if i == 0 or not outside[i - 1]:
                    current += 1
                run_id[i] = current
            else:
                run_id[i] = 0
        # keep only runs >= DEPEG_MIN_DURATION_HOURS
        run_lengths = pd.Series(run_id).value_counts()
        valid_runs = {r for r in run_lengths.index if r != 0 and run_lengths[r] >= DEPEG_MIN_DURATION_HOURS}
        for i, r in enumerate(run_id):
            if r in valid_runs:
                df.loc[idx[i], "episode_id"] = f"{coin}_{r}"

    df["in_episode"] = df["episode_id"].notna()
    return df


def build_episode_table(df: pd.DataFrame) -> pd.DataFrame:
    eps = (
        df[df["in_episode"]]
        .groupby(["coin", "episode_id"])
        .agg(start=("hour", "min"), end=("hour", "max"), n_hours=("hour", "count"),
             min_price=("close", "min"), max_price=("close", "max"))
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
        for coin in COINS:
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

    return df


def apply_embargo(df: pd.DataFrame, episodes: pd.DataFrame) -> pd.DataFrame:
    """Flag hours that fall within EMBARGO_HOURS of any episode's start/end.
    Use this column to purge windows around split boundaries so a single
    episode never has hours on both sides of a train/test cut."""
    df = df.copy()
    df["near_episode_boundary"] = False
    for _, ep in episodes.iterrows():
        lo = ep["start"] - pd.Timedelta(hours=EMBARGO_HOURS)
        hi = ep["end"] + pd.Timedelta(hours=EMBARGO_HOURS)
        mask = (df["coin"] == ep["coin"]) & (df["hour"] >= lo) & (df["hour"] <= hi)
        df.loc[mask, "near_episode_boundary"] = True
    return df


def chronological_split(df: pd.DataFrame, cutoff: str, horizon_h: int):
    """Purged chronological split: train = hours <= cutoff - h (so no label
    look-ahead crosses the cutoff), test = hours > cutoff. Then drop any
    episode that has rows on both sides entirely from both sides."""
    cutoff = pd.Timestamp(cutoff, tz="UTC")
    train = df[df["hour"] <= cutoff - pd.Timedelta(hours=horizon_h)].copy()
    test = df[df["hour"] > cutoff].copy()

    train_eps = set(train["episode_id"].dropna())
    test_eps = set(test["episode_id"].dropna())
    straddling = train_eps & test_eps
    if straddling:
        train = train[~train["episode_id"].isin(straddling)]
        test = test[~test["episode_id"].isin(straddling)]
    return train, test


def main():
    price = pd.read_parquet(PROCESSED_DATA_DIR / "hourly_price.parquet")
    price = price.rename(columns={"timestamp": "hour"})
    onchain = pd.read_parquet(PROCESSED_DATA_DIR / "hourly_onchain_features.parquet")

    merged = merge_sources(price, onchain)
    merged = flag_episodes(merged)
    episodes = build_episode_table(merged)
    merged = build_labels(merged, episodes)
    merged = apply_embargo(merged, episodes)

    merged.to_parquet(PROCESSED_DATA_DIR / "master_hourly_dataset.parquet", index=False)
    merged.to_csv(PROCESSED_DATA_DIR / "master_hourly_dataset.csv", index=False)
    episodes.to_parquet(PROCESSED_DATA_DIR / "depeg_episodes.parquet", index=False)
    episodes.to_csv(PROCESSED_DATA_DIR / "depeg_episodes.csv", index=False)

    print(f"Master dataset: {len(merged):,} coin-hour rows")
    print(f"Depeg episodes found: {len(episodes)}")
    print(episodes)


if __name__ == "__main__":
    main()