"""Diagnose on-chain feature scales, missingness, and coin identity.

    python -m tests.diagnose_onchain_features

1. Coin-identity probe: a RandomForest predicting `coin` from
   (a) the raw on-chain levels  vs  (b) the 12 log1p-diff change features.
   Raw levels carry the coin's identity (median tx_count PAX ~12 vs
   USDT ~5,400), so (a) should be near-perfect and (b) markedly lower. The
   probe uses a temporal split (train <= 2023-12-31, test >= 2024-01-01) so
   memorising coin levels does not pass just because of in-sample rows.

2. Numerical hygiene: no +/-inf in the new columns; per-coin 1st/99th
   percentiles of each change feature are printed so cross-coin scale
   comparability can be eyeballed.

3. NaN pattern: the change features are NaN exactly where an endpoint of the
   lag window is unobserved (coin starts, TUSD's late Dune start, gaps).

This manual diagnostic retains its historical retrospective risk mask for
comparison continuity. It does not define the canonical training sample.

Exit code is non-zero only on hard failures (missing columns, +/-inf).
The accuracy comparison is informational: the plan records the before/after
numbers rather than enforcing a threshold.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "data_processing"))

from data_processing.config import DEPEG_GAP_HOURS
from data_processing.feature_schema import ONCHAIN_FEATURES
from data_processing.splits import load_master  # noqa: E402

RAW_LEVELS = ["tx_count", "tx_volume", "active_senders", "active_receivers"]
SPLIT_HOUR = pd.Timestamp("2024-01-01", tz="UTC")


RECOVERY_HOURS = DEPEG_GAP_HOURS


def recovery_window_mask(df: pd.DataFrame, episodes: pd.DataFrame,
                         recovery_hours: int = RECOVERY_HOURS) -> pd.Series:
    """Mark same-coin rows within the post-episode recovery window.

    True for hours in (episode end, episode end + recovery_hours] of the same
    coin.
    """
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

    This retrospective mask is retained only for historical diagnostic
    comparisons. Production training uses canonical eligibility flags.
    """
    return (
        ~df["in_episode"]
        & ~recovery_window_mask(df, episodes, recovery_hours)
        & df[f"label_h{horizon}"].notna()
    )



def coin_probe(master: pd.DataFrame, features: list[str], label: str,
               split: str = "temporal") -> dict:
    """Train a small RF to predict coin from `features`."""
    df = master[["hour", "coin"] + features].dropna()
    y = df["coin"]
    X = df[features].astype(float)
    if split == "temporal":
        tr = df["hour"] < SPLIT_HOUR
    else:  # random, stratified
        from sklearn.model_selection import train_test_split
        tr = pd.Series(False, index=df.index)
        idx_tr, _ = train_test_split(
            df.index, test_size=0.3, random_state=42, stratify=y)
        tr.loc[idx_tr] = True
    if tr.sum() < 100 or (~tr).sum() < 100:
        raise RuntimeError(f"{label}: too few rows in {split} split ({tr.sum()}/{ (~tr).sum()})")
    model = RandomForestClassifier(
        n_estimators=100, max_depth=8, min_samples_leaf=5,
        random_state=42, n_jobs=-1,
    )
    model.fit(X[tr], y[tr])
    pred = model.predict(X[~tr])
    return {
        "label": label,
        "split": split,
        "n_features": len(features),
        "rows_kept": len(df),
        "train_rows": int(tr.sum()),
        "test_rows": int((~tr).sum()),
        "accuracy": float(accuracy_score(y[~tr], pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y[~tr], pred)),
    }


def main() -> int:
    """Run coin-identity probes and on-chain feature hygiene diagnostics."""
    master, episodes = load_master()
    at_risk = at_risk_mask(master, episodes, horizon=24)
    m = master.loc[at_risk].reset_index(drop=True)

    missing_raw = [c for c in RAW_LEVELS if c not in m.columns]
    missing_chg = [c for c in ONCHAIN_FEATURES if c not in m.columns]
    if missing_chg:
        raise SystemExit(
            f"FAIL: change features missing from master.parquet: {missing_chg}. "
            "Regenerate interim features with python -m "
            "data_processing.build_onchain_features, then rebuild the dataset "
            "into a fresh directory with data_processing.build_dataset."
        )
    if missing_raw:
        raise SystemExit(f"FAIL: raw on-chain columns missing: {missing_raw}")

    print("=" * 78)
    print("1. Coin-identity probe (RandomForest, chance balanced accuracy = 1/6)")
    print("=" * 78)
    for split in ("temporal", "random"):
        print(f"\n--- {split} split ---")
        for features, label in [
            (RAW_LEVELS, "raw levels"),
            (ONCHAIN_FEATURES, "12 log1p-diff change features"),
        ]:
            r = coin_probe(m, features, label, split=split)
            print(f"{label}: acc={r['accuracy']:.3f} balanced={r['balanced_accuracy']:.3f} "
                  f"(rows kept {r['rows_kept']:,}, test {r['test_rows']:,})")

    # 2. Numerical hygiene
    print("\n" + "=" * 78)
    print("2. Numerical hygiene")
    print("=" * 78)
    inf_counts = m[ONCHAIN_FEATURES].isin([np.inf, -np.inf]).sum()
    print(f"+/-inf counts per change feature:\n{inf_counts.to_string()}")
    assert inf_counts.sum() == 0, "FAIL: +/-inf found in change features"
    print("-> no +/-inf: PASS")

    print("\nPer-coin 1st percentile of each change feature:")
    print(m.groupby("coin")[ONCHAIN_FEATURES].quantile(0.01).round(3).to_string())
    print("\nPer-coin 99th percentile of each change feature:")
    print(m.groupby("coin")[ONCHAIN_FEATURES].quantile(0.99).round(3).to_string())

    # 3. NaN pattern
    print("\n" + "=" * 78)
    print("3. NaN share of change features in at-risk rows (h=24)")
    print("=" * 78)
    print(m[ONCHAIN_FEATURES].isna().mean().round(4).to_string())
    print("\nChecks completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
