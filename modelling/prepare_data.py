"""
Prepare model-ready data for each expanding-window CV fold.

For a horizon h and feature tier, each fold returns X / y for:
  train, test               -> fit on train, predict test (pool test across folds)
  inner_train, inner_val    -> nested tuning (hyperparameters, threshold) inside the fold
plus `test_meta` (coin, hour, episode_id) so pooled OOS predictions can be
traced back to coins and dates.

Folds come from data_processing/splits.py (24h purge, 24h embargo,
boundary-straddling episodes dropped). This module only selects the sample
and the feature columns -- it doesn't change master.parquet.

SAMPLE (team convention since PR #11, 2026-10-14)
The canonical sample is `prediction_eligible_h{h} == True` from
master.parquet: rows with complete features (common_sample), an above-band
price, not inside an episode (in_episode covers the recovery window since
PR #11), and an observed label_h{h} (not right-censored). load_folds() uses
it automatically and falls back to the legacy at-risk mask (at_risk_mask)
on older masters.

FEATURES
Only changes, ratios and deviations -- no raw levels. Raw trading volume and
on-chain counts differ by orders of magnitude between coins (e.g. median
hourly volume DAI ~70 vs BUSD ~10M), so a pooled tree model would learn
the level as a coin identifier rather than a market signal.
Tier 1 also includes `peg_stress = clip(-peg_deviation, 0, 0.01)`, a derived
one-split "stressed regime" feature computed here (pipeline untouched).
Tier 2 adds the 12 on-chain log1p-difference change features built in
02b_build_onchain_features.py (see ONCHAIN_FEATURES below).
Not used as features: coin, hour, price (= peg_deviation + 1), duplicate
scalings (peg_deviation_bps, simple returns), venue metadata, depeg flags,
episode IDs and labels.

Missing feature values are left as NaN. XGBoost handles NaN natively; for
Lasso / Random Forest, impute inside the modelling code, fitted on the
training set only. Note: the eligible sample already excludes rows with
missing features, so NaNs are rare inside folds.

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
    "peg_stress",  # clip(-peg_deviation, 0, 0.01): one-split "stressed" regime
    "log_return_1h", "log_return_6h", "log_return_24h",
    "realised_vol_6h", "realised_vol_24h",
    "volume_pct_change_1h", "volume_pct_change_6h", "volume_pct_change_24h",
]

# Tier 2: market + on-chain. On-chain columns are 1h/6h/24h log1p-difference
# change features ({metric}_logdiff_{k}h = log1p(x_t) - log1p(x_{t-k})), built
# in 02b_build_onchain_features.py on the full hourly grid (true calendar
# lags). Raw levels (tx_count, ...) are NOT features: their scale differs by
# orders of magnitude between coins (median tx_count PAX ~12 vs USDT ~5,400)
# and would act as a coin identifier. Apply with:
#   python data_processing/02b_build_onchain_features.py --update-master
ONCHAIN_FEATURES = [
    f"{metric}_logdiff_{k}h"
    for metric in ("tx_count", "tx_volume", "active_senders", "active_receivers")
    for k in (1, 6, 24)
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
    """Rows in the risk set for horizon h (see module docstring).

    Legacy sample. Since PR #11 (2026-10-14) the canonical sample is
    prediction_eligible_h{h} from master.parquet: feature-complete rows
    (common_sample), above-band price, not in an episode, with an observed
    label. load_folds() uses that automatically; this remains for
    robustness checks."""
    return (
        ~df["in_episode"]
        & ~recovery_window_mask(df, episodes, recovery_hours)
        & df[f"label_h{horizon}"].notna()
    )


def prediction_eligible_mask(df: pd.DataFrame, horizon: int) -> pd.Series | None:
    """The canonical training/evaluation sample flag (team convention, PR #11),
    or None when the column is absent (older master versions)."""
    col = f"prediction_eligible_h{horizon}"
    if col in df.columns:
        return df[col].astype(bool)
    return None


def _xy(part: pd.DataFrame, features: list[str], horizon: int):
    return part[features].astype(float), part[f"label_h{horizon}"].astype(int)


def load_folds(horizon: int = 24, tier: int = 1,
               recovery_hours: int = RECOVERY_HOURS) -> list[dict]:
    """One dict per CV fold with X/y for train, test, inner_train, inner_val,
    test_meta (coin, hour, episode_id) for the test rows, and
    inner_val_meta (same columns) for the inner validation rows."""
    if horizon not in FORECAST_HORIZONS_HOURS:
        raise ValueError(f"horizon must be one of {FORECAST_HORIZONS_HOURS}")
    features = TIERS[tier]

    master, episodes = load_master()
    master = master.copy()
    master["peg_stress"] = np.clip(-master["peg_deviation"], 0.0, 0.01)
    missing = [c for c in features if c not in master.columns]
    if missing:
        raise KeyError(
            f"Tier {tier} features missing from master.parquet: {missing}. "
            "Run: python data_processing/02b_build_onchain_features.py --update-master"
        )
    keep = prediction_eligible_mask(master, horizon)
    if keep is None:  # pre-PR-#11 master: fall back to the legacy risk set
        keep = at_risk_mask(master, episodes, horizon, recovery_hours)
    master = master[keep]

    out = []
    for f in expanding_window_folds(master, episodes):
        fold = {"year": f["year"], "horizon": horizon, "tier": tier, "features": features}
        for name in ["train", "test", "inner_train", "inner_val"]:
            fold[f"X_{name}"], fold[f"y_{name}"] = _xy(f[name], features, horizon)
        fold["test_meta"] = f["test"][META_COLS].reset_index(drop=True)
        fold["inner_val_meta"] = f["inner_val"][META_COLS].reset_index(drop=True)
        out.append(fold)
    return out


def main():
    master, episodes = load_master()
    master = master.copy()
    master["peg_stress"] = np.clip(-master["peg_deviation"], 0.0, 0.01)
    n = len(master)
    in_ep = master["in_episode"]
    print(f"master: {n:,} rows | in episode: {in_ep.sum():,} | "
          f"prediction_eligible: {int(master['prediction_eligible'].sum()):,}")

    for h in FORECAST_HORIZONS_HOURS:
        keep = prediction_eligible_mask(master, h)
        print(f"\n=== h = {h}: eligible rows {keep.sum():,} of {n:,} "
              f"({keep.mean():.1%}), positives {int(master.loc[keep, f'label_h{h}'].sum()):,} ===")
        for f in load_folds(horizon=h, tier=1):
            yt, ys = f["y_train"], f["y_test"]
            print(f"  fold {f['year']}: train {len(yt):>7,} rows / {yt.sum():>5,} pos | "
                  f"test {len(ys):>6,} rows / {ys.sum():>4,} pos | "
                  f"inner_val {len(f['y_inner_val']):>6,} rows / {f['y_inner_val'].sum():>4,} pos")

    print("\nPer-feature NaN share in eligible rows (h=24, tier 2):")
    keep = prediction_eligible_mask(master, 24)
    print(master.loc[keep, TIERS[2]].isna().mean().round(3).to_string())


if __name__ == "__main__":
    main()
