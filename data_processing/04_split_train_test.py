"""
Purged chronological train/test split of the labelled master dataset.

Reads the outputs of 04_merge_and_label.py and, for each forecast horizon h,
writes a train and test file:
  - train = hours <= SPLIT_CUTOFF - h (no label look-ahead crosses the cutoff)
  - test  = hours >  SPLIT_CUTOFF
  - any episode with rows on both sides is dropped from both sides
  - `near_episode_boundary` flags hours within EMBARGO_HOURS of an episode's
    start/end, for purging windows around split boundaries

All rows are kept (including unlabelled ones) so lookback features can still be
computed on contiguous history; filter on label_h{h}.notna() when training.
"""
import pandas as pd
from config import PROCESSED_DATA_DIR, FORECAST_HORIZONS_HOURS, EMBARGO_HOURS, SPLIT_CUTOFF


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
    master = pd.read_parquet(PROCESSED_DATA_DIR / "master_hourly_dataset_fiat.parquet")
    episodes = pd.read_parquet(PROCESSED_DATA_DIR / "depeg_episodes_fiat.parquet")

    master = apply_embargo(master, episodes)

    print(f"Split cutoff: {SPLIT_CUTOFF} (embargo {EMBARGO_HOURS}h)")
    for h in FORECAST_HORIZONS_HOURS:
        train, test = chronological_split(master, SPLIT_CUTOFF, h)
        train.to_parquet(PROCESSED_DATA_DIR / f"train_h{h}_fiat.parquet", index=False)
        test.to_parquet(PROCESSED_DATA_DIR / f"test_h{h}_fiat.parquet", index=False)

        col = f"label_h{h}"
        for name, part in [("train", train), ("test", test)]:
            labelled = part[col].notna()
            print(f"  h={h:<2} {name:<5}: {len(part):,} rows, {labelled.sum():,} labelled, "
                  f"{int(part.loc[labelled, col].sum()):,} positive "
                  f"({part.loc[labelled, col].mean():.2%}), "
                  f"{part['episode_id'].nunique()} episodes, "
                  f"{part['hour'].min():%Y-%m-%d} .. {part['hour'].max():%Y-%m-%d}")


if __name__ == "__main__":
    main()
