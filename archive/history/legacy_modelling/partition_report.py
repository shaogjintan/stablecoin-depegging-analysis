"""Report the actual model-ready expanding-window folds.

This calls ``modelling.prepare_data.load_folds`` for every tier and forecast
horizon, so counts reflect the exact eligibility filtering, purge, embargo,
and episode-boundary handling used by training. It writes one summary row per
fold, partition, tier, and horizon to ``data/processed/cv_folds_summary.csv``.

Run from the repository root:
    python modelling/reporting/partition_report.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data_processing.config import (  # noqa: E402
    FORECAST_HORIZONS_HOURS,
    OOS_END_YEAR,
    OOS_START_YEAR,
    PROCESSED_DATA_DIR,
)
from modelling.prepare_data import load_folds  # noqa: E402

OUT_PATH = PROCESSED_DATA_DIR / "cv_folds_summary.csv"
PARTITIONS = ("train", "test", "inner_train", "inner_val")


def _partition_summary(fold: dict, name: str) -> dict:
    """Summarize one already-filtered partition returned by load_folds."""
    labels = fold[f"y_{name}"]
    meta = fold[f"{name}_meta"]
    return {
        "fold_year": fold["year"],
        "tier": fold["tier"],
        "set": name,
        "horizon_h": fold["horizon"],
        "start": meta["hour"].min(),
        "end": meta["hour"].max(),
        "rows": len(labels),
        "labelled": int(labels.notna().sum()),
        "positives": int(labels.sum()),
        # This means episode IDs present in rows, not forecastable/caught episodes.
        "episode_ids_present": meta["episode_id"].nunique(),
    }


def main() -> int:
    rows = []
    print(f"Model-ready expanding-window CV: {OOS_START_YEAR}-{OOS_END_YEAR}")

    for tier in (1, 2):
        for horizon in FORECAST_HORIZONS_HOURS:
            folds = load_folds(horizon=horizon, tier=tier)
            print(f"\n--- Tier {tier}, horizon {horizon}h ---")
            pooled_test_rows = 0
            pooled_test_positives = 0
            pooled_test_episode_ids: set[str] = set()

            for fold in folds:
                print(f"\nFold {fold['year']} (inner validation {fold['year'] - 1}):")
                for name in PARTITIONS:
                    row = _partition_summary(fold, name)
                    rows.append(row)
                    print(
                        f"  {name:<11}: {row['rows']:>7,} model rows | "
                        f"{row['positives']:>4} positives | "
                        f"{row['episode_ids_present']:>3} episode IDs present | "
                        f"{row['start']:%Y-%m-%d} to {row['end']:%Y-%m-%d}"
                    )
                pooled_test_rows += len(fold["y_test"])
                pooled_test_positives += int(fold["y_test"].sum())
                pooled_test_episode_ids.update(
                    fold["test_meta"]["episode_id"].dropna().astype(str)
                )

            print(
                f"\nPooled OOS: {pooled_test_rows:,} model rows | "
                f"{pooled_test_positives:,} positives | "
                f"{len(pooled_test_episode_ids)} episode IDs present"
            )

    pd.DataFrame(rows).to_csv(OUT_PATH, index=False)
    print(f"\nSaved model-ready fold summary -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
