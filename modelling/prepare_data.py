"""
Prepare model-ready data for each expanding-window CV fold.

For a horizon h and feature tier, each fold returns X / y for:
  train, test               -> fit on train, predict test (pool test across folds)
  inner_train, inner_val    -> nested tuning (hyperparameters, threshold) inside the fold
plus `test_meta` (coin, hour, episode_id) so pooled OOS predictions can be
traced back to coins and dates.

Folds come from data_processing/splits.py (24h purge, 24h embargo,
boundary-straddling episodes dropped). This module only selects the at-risk
sample and the feature columns -- it doesn't change master.parquet.

AT-RISK SAMPLE (the risk set)
The label is the ONSET of a NEW depeg episode. A coin can only start a new
episode while it is on peg, so an hour is at risk only if:
  1. it is not inside an ongoing episode (in_episode == False), and
  2. it is not in the recovery window: the RECOVERY_HOURS after an episode's
     last below-band hour. Recovery = 24h back above $0.99 (DEPEG_GAP_HOURS;
     slide 10), since any dip within that window is merged into the same
     episode and can't be a new onset. The coin re-enters the risk set only
     once it has recovered.
  3. it has a label (label_h{h} not NaN: has a price and isn't right-censored).
Rows outside the risk set are dropped from every set (train, test, inner).
Set RECOVERY_HOURS = 0 to keep recovery-window hours (robustness check).

FEATURES
Only changes, ratios and deviations -- no raw levels. Raw trading volume and
on-chain counts differ by orders of magnitude between coins (e.g. median
hourly volume DAI ~70 vs BUSD ~10M), so a pooled tree model would learn
the level as a coin identifier rather than a market signal.
Not used as features: coin, hour, price (= peg_deviation + 1), duplicate
scalings (peg_deviation_bps, simple returns), venue metadata, depeg flags,
episode IDs and labels.

Missing feature values are left as NaN. XGBoost handles NaN natively; for
Lasso / Random Forest, impute inside the modelling code, fitted on the
training set only.

Usage (run from the repo root):
    from modelling.prepare_data import load_folds
    for fold in load_folds(horizon=24, tier=1):
        model.fit(fold["X_train"], fold["y_train"])
        p = model.predict_proba(fold["X_test"])[:, 1]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_processing"))
from config import DEPEG_GAP_HOURS, FORECAST_HORIZONS_HOURS  # noqa: E402
from splits import load_master, expanding_window_folds  # noqa: E402

RECOVERY_HOURS = DEPEG_GAP_HOURS  # 24h

# Tier 1: market only (Lee et al. 2025 style)
MARKET_FEATURES = [
    "peg_deviation",
    "log_return_1h", "log_return_6h", "log_return_24h",
    "realised_vol_6h", "realised_vol_24h",
    "volume_pct_change_1h", "volume_pct_change_6h", "volume_pct_change_24h",
]

# Tier 2: market + on-chain.
# TODO: these are raw hourly LEVELS, which identify the coin (median tx_count
# PAX ~12 vs USDT ~5,400). Replace with 1h/6h/24h change features built in
# 02b before using Tier 2 results.
ONCHAIN_FEATURES = [
    "tx_count", "tx_volume", "active_senders", "active_receivers",
]

TIERS = {1: MARKET_FEATURES, 2: MARKET_FEATURES + ONCHAIN_FEATURES}
META_COLS = ["coin", "hour", "episode_id"]


def recovery_window_mask(df: pd.DataFrame, episodes: pd.DataFrame,
                         recovery_hours: int = RECOVERY_HOURS) -> pd.Series:
    """True for hours in (episode end, episode end + recovery_hours] of the same coin."""
    mask = pd.Series(False, index=df.index)
    if recovery_hours <= 0:
        return mask
    for _, ep in episodes.iterrows():
        lo = ep["end"]
        hi = ep["end"] + pd.Timedelta(hours=recovery_hours)
        mask |= (df["coin"] == ep["coin"]) & (df["hour"] > lo) & (df["hour"] <= hi)
    return mask


def at_risk_mask(df: pd.DataFrame, episodes: pd.DataFrame, horizon: int,
                 recovery_hours: int = RECOVERY_HOURS) -> pd.Series:
    """Rows in the risk set for horizon h (see module docstring)."""
    return (
        ~df["in_episode"]
        & ~recovery_window_mask(df, episodes, recovery_hours)
        & df[f"label_h{horizon}"].notna()
    )


def _xy(part: pd.DataFrame, features: list[str], horizon: int):
    return part[features].astype(float), part[f"label_h{horizon}"].astype(int)


def load_folds(horizon: int = 24, tier: int = 1,
               recovery_hours: int = RECOVERY_HOURS) -> list[dict]:
    """One dict per CV fold with X/y for train, test, inner_train, inner_val,
    and test_meta (coin, hour, episode_id) for the test rows."""
    if horizon not in FORECAST_HORIZONS_HOURS:
        raise ValueError(f"horizon must be one of {FORECAST_HORIZONS_HOURS}")
    features = TIERS[tier]

    master, episodes = load_master()
    keep = at_risk_mask(master, episodes, horizon, recovery_hours)
    master = master[keep]

    out = []
    for f in expanding_window_folds(master, episodes):
        fold = {"year": f["year"], "horizon": horizon, "tier": tier, "features": features}
        for name in ["train", "test", "inner_train", "inner_val"]:
            fold[f"X_{name}"], fold[f"y_{name}"] = _xy(f[name], features, horizon)
        fold["test_meta"] = f["test"][META_COLS].reset_index(drop=True)
        out.append(fold)
    return out


def main():
    master, episodes = load_master()
    n = len(master)
    in_ep = master["in_episode"]
    recov = recovery_window_mask(master, episodes)
    print(f"master: {n:,} rows | in episode: {in_ep.sum():,} | "
          f"recovery window ({RECOVERY_HOURS}h): {recov.sum():,}")

    for h in FORECAST_HORIZONS_HOURS:
        keep = at_risk_mask(master, episodes, h)
        print(f"\n=== h = {h}: at-risk rows {keep.sum():,} of {n:,} "
              f"({keep.mean():.1%}), positives {int(master.loc[keep, f'label_h{h}'].sum()):,} ===")
        for f in load_folds(horizon=h, tier=1):
            yt, ys = f["y_train"], f["y_test"]
            print(f"  fold {f['year']}: train {len(yt):>7,} rows / {yt.sum():>5,} pos | "
                  f"test {len(ys):>6,} rows / {ys.sum():>4,} pos | "
                  f"inner_val {len(f['y_inner_val']):>6,} rows / {f['y_inner_val'].sum():>4,} pos")

    print("\nFeature NaN share in at-risk rows (h=24):")
    keep = at_risk_mask(master, episodes, 24)
    print(master.loc[keep, TIERS[2]].isna().mean().round(3).to_string())


if __name__ == "__main__":
    main()
