"""Prepare canonical eligible samples for expanding-window model training.

Eligibility flags, labels, feature columns, and completed-bar timestamps are
read from the canonical dataset. This module does not rebuild features or
fall back to retrospective risk masks. Feature tiers live in feature_schema;
splits enforces the 25h purge, 24h embargo, and episode-boundary exclusions.

Use corrected_folds() when the dataset is already loaded, or load_folds()
for the same preparation with canonical input loading included.
"""
from __future__ import annotations

import pandas as pd

from data_processing.config import FORECAST_HORIZONS_HOURS
from data_processing.feature_schema import META_COLS, TIERS
from data_processing.splits import expanding_window_folds, load_master


def prediction_eligible_mask(
    df: pd.DataFrame, horizon: int, tier: int = 1
) -> pd.Series:
    """Return the required canonical tier/horizon eligibility flag.

    Args:
        df: Canonical master containing boolean prediction-eligibility flags.
        horizon: Forecast horizon in hours.
        tier: Feature tier, 1 for market or 2 for market plus on-chain.

    Returns:
        Boolean row mask with the same index as df.

    Raises:
        ValueError: The horizon, tier, or eligibility values are invalid.
        KeyError: The canonical eligibility column is missing.
    """
    if horizon not in FORECAST_HORIZONS_HOURS:
        raise ValueError(f"horizon must be one of {FORECAST_HORIZONS_HOURS}")
    if tier not in TIERS:
        raise ValueError(f"tier must be one of {tuple(TIERS)}")
    name = "market" if tier == 1 else "market_onchain"
    column = f"prediction_eligible_{name}_h{horizon}"
    if column not in df:
        raise KeyError(f"Missing canonical eligibility column: {column}")
    mask = df[column]
    if not pd.api.types.is_bool_dtype(mask) or mask.isna().any():
        raise ValueError(f"Eligibility must be non-missing boolean: {column}")
    return mask.astype(bool)


def corrected_folds(master, episodes, horizon, tier):
    """Prepare eligible feature, label, and metadata partitions by test year.

    Args:
        master: Canonical master table with stored eligibility and labels.
        episodes: Matching canonical episode manifest.
        horizon: Forecast horizon in hours.
        tier: Feature tier, 1 for market or 2 for market plus on-chain.

    Returns:
        Fold dictionaries containing features, labels, and metadata for
        train, inner_train, inner_val, and test, sorted by coin and hour.
    """
    sample = master[prediction_eligible_mask(master, horizon, tier)]
    features = TIERS[tier]
    folds = []
    for fold in expanding_window_folds(sample, episodes):
        out = dict(year=fold["year"], horizon=horizon, tier=tier,
                   features=features)
        if "dataset_version" in master:
            out["dataset_version"] = master.dataset_version.iloc[0]
        for part in ("train", "inner_train", "inner_val", "test"):
            rows = fold[part].sort_values(["coin", "hour"]).reset_index(
                drop=True
            )
            out[f"X_{part}"] = rows[features].astype(float)
            out[f"y_{part}"] = rows[f"label_h{horizon}"].astype(int)
            meta_columns = META_COLS + (
                ["dataset_version"] if "dataset_version" in rows else []
            )
            out[f"{part}_meta"] = rows[meta_columns].copy()
        assert out["inner_train_meta"].hour.max() < (
            out["inner_val_meta"].hour.min()
        )
        assert out["inner_val_meta"].hour.max() < out["test_meta"].hour.min()
        folds.append(out)
    return folds


def load_folds(horizon: int = 24, tier: int = 1) -> list[dict]:
    """Load canonical inputs and prepare the same folds as corrected_folds.

    Args:
        horizon: Forecast horizon in hours.
        tier: Feature tier, 1 for market or 2 for market plus on-chain.

    Returns:
        Canonical feature, label, and metadata partitions for each test year.
    """
    master, episodes = load_master()
    return corrected_folds(master, episodes, horizon, tier)


def main():
    """Print canonical sample counts and fold sizes without modifying data."""
    master, episodes = load_master()
    print(f"master: {len(master):,} rows")
    for horizon in FORECAST_HORIZONS_HOURS:
        keep = prediction_eligible_mask(master, horizon, tier=1)
        positives = int(master.loc[keep, f"label_h{horizon}"].sum())
        print(f"\nh={horizon}: {int(keep.sum()):,} eligible market rows; "
              f"{positives:,} positives")
        for fold in corrected_folds(master, episodes, horizon, tier=1):
            train, test = fold["y_train"], fold["y_test"]
            calibration = fold["y_inner_val"]
            print(f"  fold {fold['year']}: train {len(train):,}; "
                  f"test {len(test):,}; calibration {len(calibration):,}")
    keep = prediction_eligible_mask(master, 24, tier=2)
    print("\nPer-feature NaN share in eligible rows (h=24, tier 2):")
    print(master.loc[keep, TIERS[2]].isna().mean().round(3).to_string())


if __name__ == "__main__":
    main()
