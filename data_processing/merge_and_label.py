"""Merge source tables and write the canonical corrected dataset directly.

Run ``python -m data_processing.merge_and_label`` to validate existing outputs,
or add ``--rebuild`` to reconstruct from the hourly snapshot and interim feature
tables. Use ``--out-dir`` for a fresh output pair. The shared builder applies
completed-bar clocks, merged-onset labels and causal eligibility in memory;
no legacy master or episode file is written. Episode helpers below are reused
by the corrected-data preparation function.
"""
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from data_processing.config import (
    DEPEG_BAND, DEPEG_MIN_DURATION_HOURS, DEPEG_GAP_HOURS,
)


def build_full_hourly_grid(pv: pd.DataFrame) -> pd.DataFrame:
    """Return a complete hourly grid within each coin's observed range."""
    frames = []
    for coin in sorted(pv["coin"].unique()):
        p = pv[pv["coin"] == coin]
        grid = pd.DataFrame({
            "hour": pd.date_range(
                p["hour"].min(), p["hour"].max(), freq="h", tz="UTC"
            )
        })
        grid["coin"] = coin
        frames.append(grid)
    return pd.concat(frames, ignore_index=True)


def merge_sources(
    pv: pd.DataFrame, onchain: pd.DataFrame, market: pd.DataFrame
) -> pd.DataFrame:
    """Merge source tables onto unique coin-hour keys, preserving missingness.

    Args:
        pv: Acquired reference prices and exchange volume.
        onchain: Hourly on-chain levels and changes.
        market: Derived market features, including trading_volume.

    Returns:
        A complete coin-hour grid with the three sources joined by key.
    """
    keys = ["coin", "hour"]
    for name, source in [
        ("price/volume", pv), ("on-chain", onchain), ("market", market)
    ]:
        duplicates = source.duplicated(keys, keep=False)
        if duplicates.any():
            examples = (
                source.loc[duplicates, keys].drop_duplicates()
                .head(5).to_dict("records")
            )
            raise ValueError(
                f"{name} data has duplicate coin-hour keys; "
                f"examples: {examples}"
            )

    grid = build_full_hourly_grid(pv)
    # Market features already contain trading_volume.
    pv = pv.drop(columns=["volume"])
    # Only selected feature columns are retained by the dataset builder.
    merged = grid.merge(pv, on=keys, how="left", validate="one_to_one")
    merged = merged.merge(onchain, on=keys, how="left", validate="one_to_one")
    merged = merged.merge(market, on=keys, how="left", validate="one_to_one")
    merged = merged.sort_values(["coin", "hour"]).reset_index(drop=True)
    return merged


def flag_episodes(df: pd.DataFrame) -> pd.DataFrame:
    """Flag qualifying dips and merge them under the recovery rule."""
    df = df.copy()
    # Use the raw price threshold shared with prediction eligibility. Exactly
    # $0.99 is on-band; only lower prices count as downward depegs.
    peg_threshold = 1 - DEPEG_BAND
    df["outside_band"] = df["price"] < peg_threshold
    df["episode_id"] = pd.array([None] * len(df), dtype="object")
    df["qualifying_depeg_hour"] = False
    df["qualifying_dip_start"] = False
    df["episode_end_hour"] = pd.Series(
        pd.NaT, index=df.index, dtype=df["hour"].dtype
    )
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
                df.loc[
                    idx[run_start:run_end + 1], "qualifying_depeg_hour"
                ] = True
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

        for episode_num, (episode_start, last_out, _) in enumerate(
            clusters, start=1
        ):
            # The first above-band hour is last_out + 1; include 24 hours
            # from that anchor, making the next independent onset eligible
            # only after the recovery window has elapsed.
            episode_end = min(last_out + DEPEG_GAP_HOURS + 1, len(outside) - 1)
            df.loc[
                idx[episode_start:episode_end + 1], "episode_id"
            ] = f"{coin}_{episode_num}"
            # The manifest ends at recovery onset, one hour after the
            # final qualifying dip; it does not include the recovery window.
            df.loc[idx[last_out], "episode_end_hour"] = (
                g.iloc[last_out]["hour"] + pd.Timedelta(hours=1)
            )

    df["in_episode"] = df["episode_id"].notna()
    return df


def build_episode_table(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize flagged depeg episodes and their coin-specific time bounds."""
    eps = (
        df[df["qualifying_depeg_hour"]]
        .groupby(["coin", "episode_id"])
        .agg(
            start=("hour", "min"), end=("episode_end_hour", "max"),
            n_hours=("qualifying_depeg_hour", "sum"),
             min_price=("price", "min"), max_price=("price", "max"))
        .reset_index()
        .sort_values("start")
    )
    return eps


def flag_feature_samples(
    df: pd.DataFrame,
    market_features: list,
    onchain_features: list
) -> pd.DataFrame:
    """Mark current calendar-feature completeness for each feature group.

    market_sample and common_sample record nonmissing current features.
    These flags describe completeness only. Prediction eligibility also
    requires the causal risk mask and an observed horizon-specific label.
    """
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


def main():
    """Build corrected outputs directly without persisting a legacy master."""
    from data_processing.build_dataset import main as build_main
    build_main()


if __name__ == "__main__":
    main()
