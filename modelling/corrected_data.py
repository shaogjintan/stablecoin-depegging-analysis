"""Versioned XGB risk set; never overwrites the committed legacy master."""
from __future__ import annotations

import numpy as np
import pandas as pd

from modelling.prepare_data import MARKET_FEATURES, ONCHAIN_FEATURES, META_COLS
from data_processing.merge_and_label import flag_episodes, build_episode_table
from data_processing.splits import expanding_window_folds


def calendar_market_features(df):
    df = df.sort_values(["coin", "hour"]).copy()
    if df.duplicated(["coin", "hour"]).any():
        raise ValueError("Duplicate coin-hour")
    frames = []
    for coin, g in df.groupby("coin", sort=False):
        g = g.set_index("hour").reindex(pd.date_range(g.hour.min(), g.hour.max(), freq="h"))
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
            g[f"volume_pct_change_{h}h"] = change.where(g.hours_since_venue_start >= h + 1)
        for h in (6, 24):
            g[f"realised_vol_{h}h"] = ret.rolling(h, min_periods=h).std()
        frames.append(g.reset_index())
    return pd.concat(frames, ignore_index=True)


def causal_risk_mask(df, recovery_hours=24, band=.01, min_duration=2):
    """At bar close, exclude below-band and confirmed-dip recovery; no final end."""
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


def episode_labels(df, episodes, horizons=(1, 6, 24), band=.01, min_duration=2):
    """New merged starts only; negative windows include onset-confirmation bar."""
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
    master = calendar_market_features(master)
    master = flag_episodes(master)
    episodes = build_episode_table(master)
    master = episode_labels(master, episodes)
    master["observable_at_risk"] = causal_risk_mask(master)
    for name, features in (("market", MARKET_FEATURES),
                           ("market_onchain", MARKET_FEATURES + ONCHAIN_FEATURES)):
        base = master.observable_at_risk & np.isfinite(master[features]).all(axis=1)
        master[f"prediction_eligible_{name}"] = base
        for h in (1, 6, 24):
            master[f"prediction_eligible_{name}_h{h}"] = base & master[f"label_h{h}"].notna()
    # Source hour denotes bar start; every operational clock is its close.
    master["bar_start"] = master.hour
    master["hour"] += pd.Timedelta(hours=1)
    for col in ("start", "end"):
        episodes[col] += pd.Timedelta(hours=1)
    return master, episodes


def corrected_folds(master, episodes, horizon, tier):
    features = MARKET_FEATURES if tier == 1 else MARKET_FEATURES + ONCHAIN_FEATURES
    name = "market" if tier == 1 else "market_onchain"
    sample = master[master[f"prediction_eligible_{name}_h{horizon}"]]
    folds = []
    for f in expanding_window_folds(sample, episodes):
        out = {"year": f["year"], "horizon": horizon, "tier": tier, "features": features}
        if "dataset_version" in master:
            out["dataset_version"] = master.dataset_version.iloc[0]
        for part in ("train", "inner_train", "inner_val", "test"):
            d = f[part].sort_values(["coin", "hour"]).reset_index(drop=True)
            out[f"X_{part}"] = d[features].astype(float)
            out[f"y_{part}"] = d[f"label_h{horizon}"].astype(int)
            meta_columns = META_COLS + (["dataset_version"] if "dataset_version" in d else [])
            out[f"{part}_meta"] = d[meta_columns].copy()
        assert out["inner_train_meta"].hour.max() < out["inner_val_meta"].hour.min()
        assert out["inner_val_meta"].hour.max() < out["test_meta"].hour.min()
        folds.append(out)
    return folds
