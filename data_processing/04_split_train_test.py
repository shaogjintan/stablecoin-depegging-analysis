"""
Expanding-window (walk-forward) CV split by year, with purge, embargo and
episode-aware boundary handling.

Replaces the old single split (Train 2018-22 / Validation 2023 / Test 2024-25),
which left only 2 depeg episodes (both TUSD) in the test set. Now, for each
year Y from OOS_START_YEAR (2021) to OOS_END_YEAR (2025):
  - train on everything before Y (last 24h purged)
  - test on year Y (first 24h embargoed)
  - inner_train / inner_val = the same split one year earlier, for nested
    tuning of hyperparameters and the decision threshold
Pooling the test predictions scores every episode from 2021 onwards out of
sample. The fold logic itself lives in splits.py so modelling code can import
it; see that file for the purge/embargo/episode rules.

This script does NOT write per-fold datasets (they would be ~30 large,
mostly duplicated files). It prints a per-fold summary and writes:
  data/processed/cv_folds_summary.csv  -- boundaries, rows, positives and
                                          episodes per fold, set and horizon
"""
import pandas as pd
from config import PROCESSED_DATA_DIR, FORECAST_HORIZONS_HOURS, EMBARGO_HOURS, PURGE_HOURS
from splits import load_master, expanding_window_folds

OUT_PATH = PROCESSED_DATA_DIR / "cv_folds_summary.csv"


def summarise(name: str, part: pd.DataFrame, year: int) -> list[dict]:
    rows = []
    for h in FORECAST_HORIZONS_HOURS:
        col = f"label_h{h}"
        labelled = part[col].notna()
        rows.append({
            "fold_year": year, "set": name, "horizon_h": h,
            "start": part["hour"].min(), "end": part["hour"].max(),
            "rows": len(part), "labelled": int(labelled.sum()),
            "positives": int(part.loc[labelled, col].sum()),
            "episodes": part["episode_id"].nunique(),
        })
    return rows


def main():
    master, episodes = load_master()
    folds = expanding_window_folds(master, episodes)

    print(f"Expanding-window CV: {len(folds)} folds "
          f"({folds[0]['year']}-{folds[-1]['year']}), purge={PURGE_HOURS}h, embargo={EMBARGO_HOURS}h")

    summary = []
    for f in folds:
        y = f["year"]
        print(f"\n=== Fold {y}: train < {y}, test = {y} "
              f"(inner: train < {y - 1}, validate = {y - 1}) ===")
        if f["dropped_episodes"]:
            print(f"  Dropped boundary-straddling episodes: {f['dropped_episodes']}")
        for name in ["train", "test", "inner_train", "inner_val"]:
            part = f[name]
            rows = summarise(name, part, y)
            summary += rows
            pos = ", ".join(f"h{r['horizon_h']}={r['positives']}" for r in rows)
            print(f"  {name:<11}: {len(part):>7,} rows | {part['episode_id'].nunique():>3} episodes | "
                  f"positives {pos} | {part['hour'].min():%Y-%m-%d} to {part['hour'].max():%Y-%m-%d}")

    # pooled out-of-sample totals (test sets are disjoint years)
    pooled = pd.concat([f["test"] for f in folds])
    pos = ", ".join(f"h{h}={int(pooled[f'label_h{h}'].sum())}" for h in FORECAST_HORIZONS_HOURS)
    print(f"\nPooled OOS ({folds[0]['year']}-{folds[-1]['year']}): {len(pooled):,} rows | "
          f"{pooled['episode_id'].nunique()} episodes scored out of sample | positives {pos}")

    pd.DataFrame(summary).to_csv(OUT_PATH, index=False)
    print(f"\nSaved fold summary -> {OUT_PATH}")


if __name__ == "__main__":
    main()
