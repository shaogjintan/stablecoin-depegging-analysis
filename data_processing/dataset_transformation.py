"""Build causal features, onset labels, and prediction eligibility.

Completed-bar features, onset labels, and causal eligibility transformations.
"""
import numpy as np
import pandas as pd
from data_processing.build_market_features import calendar_market_features
from data_processing.config import (
    DEPEG_BAND,
    DEPEG_GAP_HOURS,
    DEPEG_MIN_DURATION_HOURS,
    FORECAST_HORIZONS_HOURS,
)
from data_processing.feature_schema import MARKET_FEATURES, ONCHAIN_FEATURES
from data_processing.merge_and_label import flag_episodes, build_episode_table



def causal_risk_mask(
    df, recovery_hours=DEPEG_GAP_HOURS, band=DEPEG_BAND,
    min_duration=DEPEG_MIN_DURATION_HOURS,
):
    """Identify causal at-risk prediction rows at bar close.

    At bar close, exclude below-band and confirmed-dip recovery; no final end.
    """
    out = pd.Series(False, index=df.index)
    for _, g in df.groupby("coin", sort=False):
        run = 0
        confirmed = False
        recovery_start = None
        for row in g.itertuples():
            below = pd.notna(row.price) and row.price < 1 - band
            if below:
                run += 1
                if run >= min_duration:
                    confirmed = True
                out.at[row.Index] = False
                continue
            if confirmed:
                recovery_start = row.hour
            run, confirmed = 0, False
            out.at[row.Index] = (
                pd.notna(row.price) and row.price >= 1 - band
                and (recovery_start is None or
                     row.hour > recovery_start + pd.Timedelta(hours=recovery_hours))
            )
    return out


def episode_labels(
    df, episodes, horizons=tuple(FORECAST_HORIZONS_HOURS), band=DEPEG_BAND,
    min_duration=DEPEG_MIN_DURATION_HOURS,
):
    """Label new merged episode onsets and observed negative windows.

    New merged starts only; negative windows include onset-confirmation bar.
    """
    df = df.copy()
    for h in horizons:
        df[f"label_h{h}"] = np.nan
    for coin, g in df.groupby("coin", sort=False):
        times = g.hour.astype("datetime64[ns, UTC]").astype("int64").to_numpy()
        starts = episodes.loc[episodes.coin.eq(coin), "start"].astype(
            "datetime64[ns, UTC]").astype("int64").to_numpy(copy=True)
        starts.sort()
        missing = g.price.isna().astype(int).to_numpy()
        cumulative = np.r_[0, missing.cumsum()]
        for h in horizons:
            end = times + h * 3600_000_000_000
            left = np.searchsorted(starts, times, side="right")
            right = np.searchsorted(starts, end, side="right")
            positive = right > left
            i = np.arange(len(g))
            # Minimum-duration=2 requires the bar after a putative horizon-end onset.
            complete = i + h + min_duration - 1 < len(g)
            hi = np.minimum(i + h + min_duration, len(g))
            observed = complete & ((cumulative[hi] - cumulative[i + 1]) == 0)
            labels = np.where(positive, 1., np.where(observed, 0., np.nan))
            labels[g.price.isna() | g.price.lt(1 - band)] = np.nan
            df.loc[g.index, f"label_h{h}"] = labels
    return df


def prepare_corrected(master):
    """Build causal features and labels, then shift clocks to bar close."""
    if "bar_start" in master or "observable_at_risk" in master:
        raise ValueError("Input is already corrected; refusing to shift bar clocks twice")
    master = calendar_market_features(master)
    master = flag_episodes(master)
    episodes = build_episode_table(master)
    master = episode_labels(master, episodes)
    master["observable_at_risk"] = causal_risk_mask(master)
    for name, features in (("market", MARKET_FEATURES),
                           ("market_onchain", MARKET_FEATURES + ONCHAIN_FEATURES)):
        base = master.observable_at_risk & np.isfinite(master[features]).all(axis=1)
        master[f"prediction_eligible_{name}"] = base
        for h in FORECAST_HORIZONS_HOURS:
            master[f"prediction_eligible_{name}_h{h}"] = base & master[f"label_h{h}"].notna()
    # Source hour denotes bar start; every operational clock is its close.
    master["bar_start"] = master.hour
    master["hour"] += pd.Timedelta(hours=1)
    for col in ("start", "end"):
        episodes[col] += pd.Timedelta(hours=1)
    return master, episodes

