"""Choose alarm thresholds on a rolling validation year, then score test years.

For each expanding-window test year Y, the selector uses the immediately prior
year's inner-validation rows to choose the lowest candidate score threshold
that meets an average false-alarm budget (default: at most one false alarm per
observed coin-month). The same model fit on `inner_train` scores both
`inner_val` and test year Y, so the validation cutoff is applied to scores from
the same fitted model. A false alarm is counted using the cooldown/event
definition in `modelling/alarm_events.py`.

The saved predictions from `train.py` are not used here. This script fits a
separate model on `inner_train`, uses `inner_val` for early stopping and
threshold selection, then scores the test rows with that same fitted model.
This keeps score scales aligned, at the cost of not fitting the test-scoring
model on the additional `inner_val` year.

Run from the repository root, after training the selected model/tier/horizon:

    python modelling/rolling_threshold.py \
        --model xgboost --tier 2 --horizon 24 --tag recall_tuned

The budget is an average across observed coin-months in the validation year;
some coin-months may have more than one false alarm. Test-year counts are
reported but are not constrained by validation threshold selection.

Outputs (CSV files under `reports/rolling_thresholds/` by default; change with
`--out-dir`):
  - `{model}_t{tier}_h{horizon}_{tag}_rolling_thresholds.csv`: one row per
    test year, including the selected threshold, its validation false-alarm
    rate and episode recall, and the resulting test metrics.
  - `{model}_t{tier}_h{horizon}_{tag}_test_alarm_events.csv`: individual
    test alarm events, with each alarm's score, coin, time, and whether a new
    episode began within the forecast horizon.
  - `{model}_t{tier}_h{horizon}_{tag}_test_by_coin.csv`: test alarm and
    false-alarm counts summarized by coin.
  - `{model}_t{tier}_h{horizon}_{tag}_test_by_year.csv`: the same counts
    summarized by test year.
  - `{model}_t{tier}_h{horizon}_{tag}_test_by_coin_month.csv`: the same
    counts summarized by coin and calendar month.

The alarm summary files also report flagged hours separately from grouped
alarm events, so repeated threshold crossings during cooldown are not counted
as new alarms.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "data_processing"))

from config import (  # noqa: E402
    FORECAST_HORIZONS_HOURS,
    OOS_END_YEAR,
    OOS_START_YEAR,
    PROCESSED_DATA_DIR,
)
from modelling.alarm_events import build_alarm_events, summarize_alarm_events  # noqa: E402
from modelling.models import get_model  # noqa: E402
from modelling.models.base import MedianImputer  # noqa: E402
from modelling.prepare_data import load_folds  # noqa: E402


def _default_tag(model: str, tier: int, horizon: int) -> str:
    return f"{model}_t{tier}_h{horizon}"


def _info_path(model: str, tier: int, horizon: int, year: int,
               model_dir: Path) -> Path:
    stem = f"{model}_t{tier}_h{horizon}_{year}"
    return model_dir / f"{stem}.json"


def _fold_model_params(run_info: dict, model_name: str) -> dict:
    """Recover the selected constructor parameters from train.py's run log."""
    if not run_info.get("tune"):
        return dict(run_info.get("model_params") or {})
    log = run_info.get("tune_log") or {}
    metric = log.get("tune_metric") or run_info.get("tune_metric")
    combos = log.get("combos", [])
    score_col = f"inner_{metric}"
    scored = [entry for entry in combos if entry.get(score_col) is not None]
    if not scored:
        raise ValueError(
            f"Tuned run log has no candidate scores for {model_name} ({metric})"
        )
    best = max(scored, key=lambda entry: float(entry[score_col]))
    params = dict(best["params"])
    best_iteration = best.get("best_iteration")
    if best_iteration is not None:
        params["n_estimators"] = max(int(best_iteration) + 1, 1)
    return params


def _validation_and_test_scores(
    fold: dict, model_name: str, params: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Fit once on inner_train and score inner_val and test with that model."""
    model = get_model(model_name, **params)
    X_train = fold["X_inner_train"]
    X_val = fold["X_inner_val"]
    X_test = fold["X_test"]
    imp = None
    if model.needs_imputation:
        imp = MedianImputer().fit(X_train)
        X_train = imp.transform(X_train)
        X_val = imp.transform(X_val)
        X_test = imp.transform(X_test)
    model.fit(
        X_train,
        fold["y_inner_train"],
        X_val=X_val,
        y_val=fold["y_inner_val"],
    )
    val_scores = model.predict_proba(X_val)[:, 1]
    test_scores = model.predict_proba(X_test)[:, 1]
    return val_scores, test_scores


def _validation_prediction_frame(fold: dict, scores: np.ndarray,
                                 horizon: int) -> pd.DataFrame:
    frame = fold["inner_val_meta"].copy()
    frame["horizon"] = horizon
    frame["y_true"] = fold["y_inner_val"].reset_index(drop=True)
    frame["y_score"] = scores
    return frame


def _targetable_episode_ids(predictions: pd.DataFrame,
                            episodes: pd.DataFrame,
                            horizon: int) -> set[str]:
    """Episodes that at least one prediction origin could forecast in-window."""
    targets: set[str] = set()
    starts = {
        coin: group.sort_values("start")
        for coin, group in episodes.groupby("coin", sort=False)
    }
    positives = predictions.loc[predictions["y_true"].eq(1)]
    delta = pd.Timedelta(hours=horizon)
    for row in positives.itertuples(index=False):
        coin_eps = starts.get(row.coin)
        if coin_eps is None:
            continue
        hit = coin_eps.loc[
            (coin_eps["start"] > row.hour)
            & (coin_eps["start"] <= row.hour + delta)
        ]
        if not hit.empty:
            targets.add(str(hit.iloc[0]["episode_id"]))
    return targets


def _observed_coin_months(predictions: pd.DataFrame) -> int:
    months = predictions[["coin", "hour"]].copy()
    months["coin_month"] = pd.to_datetime(months["hour"], utc=True).dt.strftime("%Y-%m")
    return int(months[["coin", "coin_month"]].drop_duplicates().shape[0])


def _false_alarm_stats(
    events: pd.DataFrame,
    predictions: pd.DataFrame,
    n_coin_months: int | None = None,
) -> tuple[int, int, float, int]:
    if n_coin_months is None:
        n_coin_months = _observed_coin_months(predictions)
    false_count = (
        int((~events["is_true_alarm"]).sum()) if not events.empty else 0
    )
    rate = false_count / n_coin_months if n_coin_months else float("inf")
    if false_count:
        false_by_month = (
            events.loc[~events["is_true_alarm"]]
            .groupby(["coin", "coin_month"])
            .size()
        )
        max_false_by_month = int(false_by_month.max())
    else:
        max_false_by_month = 0
    return false_count, n_coin_months, rate, max_false_by_month


def _choose_threshold(
    validation: pd.DataFrame,
    episodes: pd.DataFrame,
    horizon: int,
    budget: float,
) -> tuple[float, dict]:
    """Choose the lowest threshold meeting the average false-alarm budget.

    Every distinct observed score is a candidate because predictions are
    flagged with `score >= threshold`. False-alarm count is non-increasing as
    the threshold rises, so binary search finds the lowest feasible score.
    """
    scores = validation["y_score"].to_numpy(dtype=float)
    if not len(scores):
        raise ValueError("Validation fold has no scored rows")
    if not np.isfinite(scores).all():
        bad_count = int((~np.isfinite(scores)).sum())
        raise ValueError(
            f"Validation scores contain {bad_count} non-finite value(s); "
            "cannot select an alarm threshold"
        )
    candidates = np.append(np.unique(scores), float("inf"))

    target_ids = _targetable_episode_ids(validation, episodes, horizon)
    n_validation_coin_months = _observed_coin_months(validation)
    def evaluate(threshold: float) -> dict:
        _, events = build_alarm_events(validation, episodes, float(threshold))
        false_count, coin_months, rate, max_false_by_month = _false_alarm_stats(
            events, validation, n_validation_coin_months
        )
        caught = (
            set(events.loc[events["is_true_alarm"], "target_episode_id"].dropna().astype(str))
            if not events.empty else set()
        )
        recall = len(caught & target_ids) / len(target_ids) if target_ids else 0.0
        return {
            "threshold": float(threshold),
            "false_alarms": false_count,
            "coin_months": coin_months,
            "false_alarms_per_coin_month": rate,
            "max_false_alarms_in_coin_month": max_false_by_month,
            "episodes_targetable": len(target_ids),
            "episodes_warned": len(caught & target_ids),
            "episode_recall": recall,
            "alarm_events": len(events),
        }

    # Infinity is always feasible (it flags no rows), so find the first
    # feasible index in the ascending candidate list.
    low, high = 0, len(candidates) - 1
    while low < high:
        mid = (low + high) // 2
        stats = evaluate(float(candidates[mid]))
        if stats["false_alarms_per_coin_month"] <= budget:
            high = mid
        else:
            low = mid + 1
    best = evaluate(float(candidates[low]))
    return best["threshold"], best


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True,
                        help="registered model name used by modelling/train.py")
    parser.add_argument("--tier", type=int, choices=[1, 2], required=True)
    parser.add_argument("--horizon", type=int, choices=FORECAST_HORIZONS_HOURS, required=True)
    parser.add_argument("--tag", default=None,
                        help="training run tag (defaults to MODEL_tTIER_hHORIZON)")
    parser.add_argument("--false-alarms-per-coin-month", type=float, default=1.0,
                        help="maximum average validation false alarms per observed coin-month (default: 1)")
    parser.add_argument("--model-dir", default=str(REPO_ROOT / "results/models"))
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "reports/rolling_thresholds"))
    args = parser.parse_args(argv)

    tag = args.tag or _default_tag(args.model, args.tier, args.horizon)
    model_dir, out_dir = Path(args.model_dir), Path(args.out_dir)
    if args.false_alarms_per_coin_month < 0:
        parser.error("--false-alarms-per-coin-month must be non-negative")

    folds = load_folds(horizon=args.horizon, tier=args.tier)
    folds_by_year = {fold["year"]: fold for fold in folds}
    episodes = pd.read_parquet(PROCESSED_DATA_DIR / "depeg_episodes.parquet").copy()
    episodes["start"] = pd.to_datetime(episodes["start"], utc=True)

    threshold_rows = []
    test_flagged_parts = []
    test_event_parts = []
    for year in range(OOS_START_YEAR, OOS_END_YEAR + 1):
        if year not in folds_by_year:
            continue
        fold = folds_by_year[year]
        info_path = _info_path(
            args.model, args.tier, args.horizon, year, model_dir
        )
        if not info_path.exists():
            raise FileNotFoundError(
                f"Missing fold model metadata for {year}: expected {info_path}. "
                "Train this model/tier/horizon first."
            )
        run_info = json.loads(info_path.read_text())
        if run_info.get("tag") != tag:
            raise ValueError(
                f"Fold {year} artifacts have tag {run_info.get('tag')!r}, expected {tag!r}. "
                "Retrain the requested run; fold outputs are overwritten between tags."
            )

        params = _fold_model_params(run_info, args.model)
        val_scores, test_scores = _validation_and_test_scores(
            fold, args.model, params
        )
        validation = _validation_prediction_frame(fold, val_scores, args.horizon)
        threshold, val_stats = _choose_threshold(
            validation, episodes, args.horizon,
            args.false_alarms_per_coin_month,
        )

        test = fold["test_meta"].copy()
        test["hour"] = pd.to_datetime(test["hour"], utc=True)
        test["horizon"] = args.horizon
        test["split"] = "test"
        test["y_true"] = fold["y_test"].reset_index(drop=True)
        test["y_score"] = test_scores
        test_flagged, test_events = build_alarm_events(test, episodes, threshold)
        test_target_ids = _targetable_episode_ids(test, episodes, args.horizon)
        test_caught = (
            set(test_events.loc[test_events["is_true_alarm"], "target_episode_id"].dropna().astype(str))
            if not test_events.empty else set()
        )
        test_false, test_coin_months, test_false_rate, test_max_false_by_month = (
            _false_alarm_stats(test_events, test, _observed_coin_months(test))
        )
        threshold_rows.append({
            "test_year": year,
            "validation_year": year - 1,
            "model": args.model,
            "tier": args.tier,
            "horizon": args.horizon,
            "tag": tag,
            "threshold": threshold,
            "validation_false_alarms": val_stats["false_alarms"],
            "validation_coin_months": val_stats["coin_months"],
            "validation_false_alarms_per_coin_month": val_stats["false_alarms_per_coin_month"],
            "validation_max_false_alarms_in_coin_month": val_stats["max_false_alarms_in_coin_month"],
            "validation_episode_recall": val_stats["episode_recall"],
            "validation_alarm_events": val_stats["alarm_events"],
            "validation_flagged_hours": int((validation["y_score"] >= threshold).sum()),
            "test_false_alarms": test_false,
            "test_coin_months": test_coin_months,
            "test_false_alarms_per_coin_month": test_false_rate,
            "test_max_false_alarms_in_coin_month": test_max_false_by_month,
            "test_alarm_events": len(test_events),
            "test_flagged_hours": int((test["y_score"] >= threshold).sum()),
            "test_episodes_targetable": len(test_target_ids),
            "test_episodes_warned": len(test_caught & test_target_ids),
            "test_episode_recall": (
                len(test_caught & test_target_ids) / len(test_target_ids)
                if test_target_ids else np.nan
            ),
        })
        test_flagged_parts.append(test_flagged)
        test_event_parts.append(test_events)
        print(
            f"test {year} (validation {year - 1}): threshold={threshold:.6g}; "
            f"validation false alarms={val_stats['false_alarms']}/"
            f"{val_stats['coin_months']} coin-months; "
            f"test false alarms={test_false}/{test_coin_months} coin-months; "
            f"test episode recall={threshold_rows[-1]['test_episode_recall']:.3f}"
        )

    if not threshold_rows:
        raise RuntimeError("No expanding-window folds found")
    thresholds = pd.DataFrame(threshold_rows)
    all_flagged = pd.concat(test_flagged_parts, ignore_index=True) if test_flagged_parts else pd.DataFrame()
    all_events = pd.concat(test_event_parts, ignore_index=True) if test_event_parts else pd.DataFrame()
    if not all_flagged.empty:
        by_coin, by_year, by_coin_month = summarize_alarm_events(all_flagged, all_events)
    else:
        by_coin = by_year = by_coin_month = pd.DataFrame()

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.model}_t{args.tier}_h{args.horizon}_{tag}"
    outputs = {
        "thresholds": (thresholds, f"{stem}_rolling_thresholds.csv"),
        "test_alarm_events": (all_events, f"{stem}_test_alarm_events.csv"),
        "test_by_coin": (by_coin, f"{stem}_test_by_coin.csv"),
        "test_by_year": (by_year, f"{stem}_test_by_year.csv"),
        "test_by_coin_month": (by_coin_month, f"{stem}_test_by_coin_month.csv"),
    }
    for label, (frame, filename) in outputs.items():
        path = out_dir / filename
        frame.to_csv(path, index=False)
        print(f"Saved {label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
