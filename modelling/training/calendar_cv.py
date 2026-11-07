"""Training-only calendar cross-validation splits."""
import numpy as np
import pandas as pd
from data_processing.config import PURGE_HOURS, EMBARGO_HOURS



def _calendar_cv_folds(X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame,
                       episodes: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    """Create chronological, calendar-year folds using actual UTC timestamps."""
    timestamps = pd.to_datetime(meta["hour"], utc=True).reset_index(drop=True)
    years = timestamps.dt.year
    y = pd.Series(y).reset_index(drop=True)
    episode_ids = meta["episode_id"].reset_index(drop=True)
    observed_years = sorted(int(v) for v in years.dropna().unique())
    valid, skipped = [], []

    for val_year in observed_years:
        boundary = pd.Timestamp(f"{val_year}-01-01", tz="UTC")
        next_boundary = pd.Timestamp(f"{val_year + 1}-01-01", tz="UTC")
        train_end = boundary - pd.Timedelta(hours=PURGE_HOURS)
        val_start = boundary + pd.Timedelta(hours=EMBARGO_HOURS)

        train_mask = timestamps < train_end
        val_mask = (timestamps >= val_start) & (timestamps < next_boundary)

        # Match the repository's episode-aware boundary rule: remove an entire
        # episode if it overlaps the purge/embargo interval around this cut.
        lo = boundary - pd.Timedelta(hours=PURGE_HOURS)
        hi = boundary + pd.Timedelta(hours=EMBARGO_HOURS)
        crossing = episodes.loc[
            (episodes["end"] >= lo) & (episodes["start"] <= hi), "episode_id"
        ]
        excluded = set(crossing)
        if excluded:
            train_mask &= ~episode_ids.isin(excluded)
            val_mask &= ~episode_ids.isin(excluded)

        tr_idx = np.flatnonzero(train_mask.to_numpy())
        va_idx = np.flatnonzero(val_mask.to_numpy())
        val_pos = int(y.iloc[va_idx].sum()) if len(va_idx) else 0
        train_pos = int(y.iloc[tr_idx].sum()) if len(tr_idx) else 0
        reason = None
        if not len(va_idx):
            reason = "no_validation_rows"
        elif val_pos == 0:
            reason = "no_depeg_positives"
        elif train_pos == 0 or train_pos == len(tr_idx):
            reason = "training_years_do_not_contain_both_classes"

        detail = {
            "validation_year": val_year,
            "train_rows": int(len(tr_idx)),
            "validation_rows": int(len(va_idx)),
            "validation_positives": val_pos,
            "training_positives": train_pos,
        }
        if reason:
            skipped.append({**detail, "reason": reason})
        else:
            valid.append({**detail, "train_idx": tr_idx, "val_idx": va_idx})
    return valid, skipped

