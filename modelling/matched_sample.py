"""Matched coin-hour populations and fixed forest parameters."""
import numpy as np
import pandas as pd

RANDOM_FOREST_FIXED_PARAMS = {
    "n_estimators": 500, "max_depth": 8, "min_samples_leaf": 50,
    "max_features": "sqrt", "class_weight": "balanced_subsample",
    "random_state": 42,
}

def _common_sample_fold(tier1_fold: dict, tier2_fold: dict) -> dict:
    """Restrict a Tier 1 fold to the exact Tier 2 eligible coin-hours."""
    if int(tier1_fold["year"]) != int(tier2_fold["year"]):
        raise ValueError("Tier 1 and Tier 2 folds must refer to the same test year")
    out = dict(tier1_fold)
    for part in ("train", "inner_train", "inner_val", "test"):
        meta1 = tier1_fold[f"{part}_meta"].copy()
        meta2 = tier2_fold[f"{part}_meta"].copy()
        meta1["hour"] = pd.to_datetime(meta1["hour"], utc=True)
        meta2["hour"] = pd.to_datetime(meta2["hour"], utc=True)
        common_keys = set(zip(meta2["coin"], meta2["hour"]))
        mask = np.fromiter(
            ((coin, hour) in common_keys
             for coin, hour in zip(meta1["coin"], meta1["hour"])),
            dtype=bool, count=len(meta1),
        )
        if int(mask.sum()) != len(meta2):
            raise ValueError(
                f"Tier 2 {part} rows are not a subset of Tier 1 for year "
                f"{tier1_fold['year']} (h={tier1_fold['horizon']})"
            )
        positions = np.flatnonzero(mask)
        out[f"X_{part}"] = tier1_fold[f"X_{part}"].iloc[positions].reset_index(drop=True)
        out[f"y_{part}"] = tier1_fold[f"y_{part}"].iloc[positions].reset_index(drop=True)
        out[f"{part}_meta"] = meta1.iloc[positions].reset_index(drop=True)
    return out
