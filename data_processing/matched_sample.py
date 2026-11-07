"""Compare feature tiers on the same eligible coin-hour population.

The market-only comparison uses the subset eligible for market plus on-chain
features in every training, calibration, and test partition. This controls
sample composition when measuring the contribution of on-chain features.
Matching uses coin-hour keys, not the historical common_sample column.
"""
import numpy as np
import pandas as pd


def _common_sample_fold(tier1_fold: dict, tier2_fold: dict) -> dict:
    """Restrict a market-only fold to the on-chain fold's eligible keys.

    Args:
        tier1_fold: Market-only features, labels, and partition metadata.
        tier2_fold: Market plus on-chain fold for the same year and horizon.

    Returns:
        A new market-only fold retaining its original row order and feature
        columns, restricted to the on-chain population in each partition.

    Raises:
        ValueError: Fold identities, unique keys, row counts, subsets, or
            labels disagree. Different row orders between tiers are allowed.
    """
    if int(tier1_fold["year"]) != int(tier2_fold["year"]):
        raise ValueError("Tier 1 and Tier 2 folds must refer to the same test year")
    if int(tier1_fold["horizon"]) != int(tier2_fold["horizon"]):
        raise ValueError("Tier 1 and Tier 2 folds must use the same horizon")
    out = dict(tier1_fold)
    for part in ("train", "inner_train", "inner_val", "test"):
        meta1 = tier1_fold[f"{part}_meta"].copy()
        meta2 = tier2_fold[f"{part}_meta"].copy()
        meta1["hour"] = pd.to_datetime(meta1["hour"], utc=True)
        meta2["hour"] = pd.to_datetime(meta2["hour"], utc=True)
        for tier, fold, meta in [(1, tier1_fold, meta1),
                                 (2, tier2_fold, meta2)]:
            if meta[["coin", "hour"]].isna().any().any():
                raise ValueError(f"Tier {tier} {part} has missing keys")
            if meta.duplicated(["coin", "hour"]).any():
                raise ValueError(f"Tier {tier} {part} has duplicate keys")
            if (len(fold[f"X_{part}"]) != len(meta)
                    or len(fold[f"y_{part}"]) != len(meta)):
                raise ValueError(f"Tier {tier} {part} has unaligned row counts")
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
        labels1 = pd.Series(
            tier1_fold[f"y_{part}"].iloc[positions].to_numpy(),
            index=pd.MultiIndex.from_frame(meta1.iloc[positions][["coin", "hour"]]),
        )
        labels2 = pd.Series(
            tier2_fold[f"y_{part}"].to_numpy(),
            index=pd.MultiIndex.from_frame(meta2[["coin", "hour"]]),
        ).reindex(labels1.index)
        if not labels1.eq(labels2).all():
            raise ValueError(f"Tier 1 and Tier 2 {part} labels disagree")
        out[f"X_{part}"] = tier1_fold[f"X_{part}"].iloc[positions].reset_index(drop=True)
        out[f"y_{part}"] = tier1_fold[f"y_{part}"].iloc[positions].reset_index(drop=True)
        out[f"{part}_meta"] = meta1.iloc[positions].reset_index(drop=True)
    return out
