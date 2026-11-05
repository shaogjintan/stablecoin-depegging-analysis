"""Train L1-only Logistic Regression with calendar-year CV per outer fold.

For each expanding-window test year, this script chooses C using expanding
calendar-year validation folds contained in that outer fold's training data.
Validation years with no depeg positives are recorded and skipped. The chosen
model is then refit on the complete outer training window and scored on the
untouched outer test year.

Run from the repository root:
    python modelling/train_logreg_yearcv.py
    python modelling/train_logreg_yearcv.py --tier 2 --horizon 24

Artifacts are isolated under results/logreg_l1_yearcv/ and
reports/logreg_l1_yearcv/ so the previous `lr01` results remain intact.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "data_processing"))

from config import EMBARGO_HOURS, PURGE_HOURS, PROCESSED_DATA_DIR  # noqa: E402
from modelling.alarm_events import build_alarm_events  # noqa: E402
from modelling.evaluate import evaluate_run, pooled_metrics, print_report  # noqa: E402
from modelling.models import get_model  # noqa: E402
from modelling.models.base import MedianImputer  # noqa: E402
from modelling.prepare_data import load_folds  # noqa: E402
from modelling.rolling_threshold import (  # noqa: E402
    _choose_threshold, _targetable_episode_ids, _observed_coin_months,
    _false_alarm_stats,
)
from splits import load_master  # noqa: E402

RESULTS = REPO / "results" / "logreg_l1_yearcv"
REPORTS = REPO / "reports" / "logreg_l1_yearcv"
DEFAULT_C_GRID = (0.001, 0.01, 0.1, 1.0)


def _average_precision(y_true, scores) -> float:
    """Average precision, matching the metric used in final_results."""
    return float(average_precision_score(y_true, scores))


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


def _fit_model(X: pd.DataFrame, y: pd.Series, C: float):
    imputer = MedianImputer().fit(X)
    model = get_model("logreg", C=C, penalty="l1")
    model.fit(imputer.transform(X), y)
    return model, imputer


def _choose_c(X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame,
              episodes: pd.DataFrame, c_grid: tuple[float, ...]) -> dict:
    cv_folds, skipped = _calendar_cv_folds(X, y, meta, episodes)
    if not cv_folds:
        raise RuntimeError(
            "No usable calendar-year validation folds in this outer training window. "
            "Need a validation year with depeg positives and an earlier training "
            "period containing both classes."
        )

    candidate_rows = []
    for C in c_grid:
        year_scores = []
        fold_predictions = []
        for fold in cv_folds:
            tr, va = fold["train_idx"], fold["val_idx"]
            model, imputer = _fit_model(X.iloc[tr], y.iloc[tr], C)
            p = model.predict_proba(imputer.transform(X.iloc[va]))[:, 1]
            score = _average_precision(y.iloc[va], p)
            year_scores.append({"validation_year": fold["validation_year"],
                                "average_precision": score})
            fold_predictions.append({"validation_year": fold["validation_year"],
                                     "val_idx": va, "y_score": p})
        candidate_rows.append({
            "C": C,
            "mean_calendar_year_average_precision": float(
                np.mean([r["average_precision"] for r in year_scores])),
            "per_year_scores": year_scores,
            "_fold_predictions": fold_predictions,
        })

    # Stable tie-break: candidates are evaluated in ascending C order, so an
    # exact tie selects the more strongly regularized model.
    winner = max(candidate_rows,
                 key=lambda row: row["mean_calendar_year_average_precision"])
    public_scores = [{k: v for k, v in candidate.items() if not k.startswith("_")}
                     for candidate in candidate_rows]
    return {
        "selected_C": winner["C"],
        "penalty": "l1",
        "selection_metric": "mean_average_precision_across_eligible_calendar_years",
        "validation_years": [f["validation_year"] for f in cv_folds],
        "skipped_years": skipped,
        "candidate_scores": public_scores,
        "cv_fold_predictions": winner["_fold_predictions"],
    }


def _write_reporting(tuning_records: list[dict], coefficient_records: list[dict]) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    summary = []
    for record in tuning_records:
        summary.append({
            "tier": record["tier"], "horizon": record["horizon"],
            "test_year": record["test_year"], "tag": record["tag"],
            "selected_C": record["selected_C"], "penalty": "l1",
            "validation_years": ";".join(map(str, record["validation_years"])),
            "skipped_years": ";".join(
                f"{x['validation_year']}:{x['reason']}" for x in record["skipped_years"]),
            "mean_cv_average_precision": next(
                x["mean_calendar_year_average_precision"] for x in record["candidate_scores"]
                if x["C"] == record["selected_C"]),
        })
    pd.DataFrame(summary).to_csv(REPORTS / "tuning_summary.csv", index=False)
    score_rows = []
    for record in tuning_records:
        for candidate in record["candidate_scores"]:
            for year_score in candidate["per_year_scores"]:
                score_rows.append({
                    "tier": record["tier"], "horizon": record["horizon"],
                    "test_year": record["test_year"], "tag": record["tag"],
                    "C": candidate["C"], "selected_C": record["selected_C"],
                    "is_selected": candidate["C"] == record["selected_C"],
                    "validation_year": year_score["validation_year"],
                    "average_precision": year_score["average_precision"],
                    "candidate_mean_average_precision": candidate[
                        "mean_calendar_year_average_precision"],
                })
    pd.DataFrame(score_rows).to_csv(REPORTS / "candidate_year_scores.csv", index=False)
    pd.DataFrame(coefficient_records).to_csv(REPORTS / "coefficients.csv", index=False)

    # Compact h=24 Tier 2 summary, if that run was included.
    coef = pd.DataFrame(coefficient_records)
    d = coef[(coef["tier"] == 2) & (coef["horizon"] == 24)] if not coef.empty else coef
    slide_rows = []
    for year, group in d.groupby("test_year"):
        market = group[~group["is_onchain"]].nlargest(3, "abs_coef")
        chain = group[group["is_onchain"]]
        if chain.empty:
            top_chain = ""
            nonzero = "0/0"
        else:
            top = chain.loc[chain["abs_coef"].idxmax()]
            top_chain = f"{top['feature']} ({top['coef']:.2f})"
            nonzero = f"{int((chain['coef'] != 0).sum())}/{len(chain)}"
        slide_rows.append({
            "test_year": year, "C": group["C"].iloc[0], "penalty": "l1",
            "top3_market": "; ".join(
                f"{r.feature} ({r.coef:.2f})" for r in market.itertuples()),
            "onchain_nonzero": nonzero, "max_onchain": top_chain,
        })
    if slide_rows:
        pd.DataFrame(slide_rows).to_csv(REPORTS / "coef_slide_h24_t2.csv", index=False)


def _append_final_results(predictions: pd.DataFrame, yearly_rows: list[dict],
                          event_frames: list[pd.DataFrame], tag: str,
                          alarm_budget: float) -> None:
    """Append/replace this tagged LogReg run in the shared final-results tables."""
    out = REPO / "reports" / "final_results"
    out.mkdir(parents=True, exist_ok=True)
    yearly = pd.DataFrame(yearly_rows)
    events = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame()

    summary_rows = []
    for (tier, horizon), group in predictions.groupby(["tier", "horizon"]):
        cls = pooled_metrics(group.y_true, group.y_score)
        yr = yearly[(yearly.tier == tier) & (yearly.horizon_hours == horizon)]
        event_part = (events[(events.tier == tier) & (events.horizon == horizon)]
                      if not events.empty else events)
        caught_events = (event_part.loc[event_part.target_episode_id.notna()]
                         if not event_part.empty else event_part)
        false_count = int(yr.test_false_alarms.sum())
        coin_months = int(yr.test_coin_months.sum())
        alarm_count = int(yr.test_alarm_events.sum())
        true_count = int(yr.test_true_alarms.sum())
        targetable = int(yr.test_targetable_episodes.sum())
        caught = int(yr.test_episodes_caught.sum())
        summary_rows.append({
            "dataset_version": str(group.dataset_version.iloc[0]),
            "model": "logreg", "tier": tier, "tag": tag,
            "model_configuration": tag,
            "feature_set": "market" if tier == 1 else "market + on-chain",
            "horizon_hours": horizon, "test_rows": cls["n"],
            "positive_base_rate": cls["pos_rate"],
            "average_precision": cls["pr_auc"],
            "average_precision_method": "sklearn.average_precision_score",
            "alarm_budget_fa_per_coin_month": alarm_budget,
            "false_alarms": false_count, "coin_months": coin_months,
            "false_alarms_per_coin_month": false_count / coin_months if coin_months else np.nan,
            "episode_recall": caught / targetable if targetable else np.nan,
            "episodes_caught": caught, "targetable_episodes": targetable,
            "alarm_events": alarm_count, "true_alarm_events": true_count,
            "alarm_precision": true_count / alarm_count if alarm_count else np.nan,
            "median_lead_hours": float(caught_events.lead_hours.median())
            if not caught_events.empty else np.nan,
            "max_false_alarms_in_coin_month": int(yr.max_false_alarms_in_coin_month.max()),
            "roc_auc": cls["roc_auc"], "brier": cls["brier"],
            "test_years": int(yr.test_year.nunique()),
            "missed_episodes": targetable - caught,
            "flagged_hours": int(yr.test_flagged_hours.sum()),
        })
    summary = pd.DataFrame(summary_rows)

    # Replace a prior run with the same tag/slices, preserve all other models
    # and tags, and union columns so the existing table schema stays intact.
    for filename, additions, horizon_col in [
        ("model_comparison_by_year.csv", yearly, "horizon_hours"),
        ("model_comparison_summary.csv", summary, "horizon_hours"),
        ("model_alarm_events.csv", events, "horizon"),
    ]:
        path = out / filename
        current = pd.read_csv(path) if path.exists() else pd.DataFrame()
        if not current.empty and "dataset_version" not in current:
            current["dataset_version"] = "legacy"
        if not current.empty and not additions.empty:
            current = current.loc[~(
                current.model.eq("logreg") & current.tag.eq(tag)
                & current.tier.isin(additions.tier.unique())
                & current[horizon_col].isin(additions[horizon_col].unique())
                & (current.test_year.isin(additions.test_year.unique())
                   if "test_year" in current and "test_year" in additions
                   else True)
            )]
        pd.concat([current, additions], ignore_index=True, sort=False).to_csv(path, index=False)

    (out / "logreg_l1_yearcv_metadata.txt").write_text(
        f"LogReg tag: {tag}\nL1-only; C selected by mean calendar-year "
        "cross-validation average precision inside each outer training window. "
        "Calendar folds use UTC year boundaries with configured purge, embargo, "
        "and episode-boundary exclusions. Validation years without depeg "
        "positives are skipped and recorded in reports/logreg_l1_yearcv. "
        f"Alarm budget: {alarm_budget:g} false alarms per observed coin-month.\n",
        encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", type=int, choices=[1, 2], default=None,
                        help="one tier; default runs both tiers")
    parser.add_argument("--horizon", type=int, choices=[1, 6, 24], default=None,
                        help="one horizon; default runs 1, 6, and 24 hours")
    parser.add_argument("--c-grid", type=float, nargs="+", default=list(DEFAULT_C_GRID),
                        help="L1 regularization candidates (default: 0.001 0.01 0.1 1.0)")
    parser.add_argument("--tag", default="l1_yearcv")
    parser.add_argument("--false-alarms-per-coin-month", type=float, default=1.0)
    args = parser.parse_args(argv)
    if any(c <= 0 for c in args.c_grid):
        parser.error("all C values must be positive")
    if args.false_alarms_per_coin_month < 0:
        parser.error("--false-alarms-per-coin-month must be non-negative")

    tiers = [args.tier] if args.tier else [1, 2]
    horizons = [args.horizon] if args.horizon else [1, 6, 24]
    _, episodes = load_master()
    all_test_predictions, tuning_records, coefficient_records = [], [], []
    yearly_rows, event_frames = [], []
    model_dir = RESULTS / "models"
    pred_dir = RESULTS / "predictions"
    tune_dir = REPORTS / "tuning"
    for directory in (model_dir, pred_dir, tune_dir):
        directory.mkdir(parents=True, exist_ok=True)

    for tier in tiers:
        for horizon in horizons:
            folds = load_folds(horizon=horizon, tier=tier)
            for fold in folds:
                year = int(fold["year"])
                X_train = fold["X_train"].reset_index(drop=True)
                y_train = fold["y_train"].reset_index(drop=True)
                train_meta = fold["train_meta"].reset_index(drop=True)
                X_test = fold["X_test"].reset_index(drop=True)
                y_test = fold["y_test"].reset_index(drop=True)
                print(f"T{tier} h{horizon} test {year}: train {len(X_train):,} rows, "
                      f"{int(y_train.sum()):,} positives; test {len(X_test):,} rows")

                chosen = _choose_c(X_train, y_train, train_meta, episodes,
                                   tuple(args.c_grid))
                calibration_parts = []
                for cv_part in chosen["cv_fold_predictions"]:
                    idx = cv_part["val_idx"]
                    part = train_meta.iloc[idx].reset_index(drop=True).copy()
                    part["horizon"] = horizon
                    part["y_true"] = y_train.iloc[idx].reset_index(drop=True)
                    part["y_score"] = cv_part["y_score"]
                    calibration_parts.append(part)
                calibration = pd.concat(calibration_parts, ignore_index=True)
                threshold, val_stats = _choose_threshold(
                    calibration, episodes, horizon,
                    args.false_alarms_per_coin_month)

                model, imputer = _fit_model(X_train, y_train, chosen["selected_C"])
                p_test = model.predict_proba(imputer.transform(X_test))[:, 1]
                test_meta = fold["test_meta"].reset_index(drop=True).copy()
                test_meta["year"] = year
                test_meta["dataset_version"] = fold["dataset_version"]
                test_meta["horizon"] = horizon
                test_meta["tier"] = tier
                test_meta["split"] = "test"
                test_meta["y_true"] = y_test
                test_meta["y_score"] = p_test
                all_test_predictions.append(test_meta)

                flagged, events = build_alarm_events(test_meta, episodes, threshold)
                target_ids = _targetable_episode_ids(test_meta, episodes, horizon)
                caught = (set(events.loc[events.is_true_alarm, "target_episode_id"]
                              .dropna().astype(str)) if not events.empty else set())
                false_count, coin_months, false_rate, max_false = _false_alarm_stats(
                    events, test_meta, _observed_coin_months(test_meta))
                if not events.empty:
                    event_frames.append(events.assign(
                        dataset_version=fold["dataset_version"],
                        test_year=year, model="logreg", tag=args.tag,
                        model_configuration=args.tag, tier=tier,
                    ))
                test_cls = pooled_metrics(y_test, p_test)
                yearly_rows.append({
                    "dataset_version": fold["dataset_version"],
                    "model": "logreg", "tier": tier, "tag": args.tag,
                    "model_configuration": args.tag,
                    "feature_set": "market" if tier == 1 else "market + on-chain",
                    "horizon_hours": horizon,
                    "validation_year": ";".join(map(str, chosen["validation_years"])),
                    "test_year": year, "threshold": threshold,
                    "test_rows": test_cls["n"], "test_positive_hours": test_cls["n_pos"],
                    "test_positive_rate": test_cls["pos_rate"],
                    "test_pr_auc": test_cls["pr_auc"],
                    "test_average_precision": test_cls["pr_auc"],
                    "validation_rows": len(calibration),
                    "validation_positive_hours": int(calibration.y_true.sum()),
                    "validation_positive_base_rate": float(calibration.y_true.mean()),
                    "validation_average_precision": chosen["candidate_scores"][
                        [x["C"] for x in chosen["candidate_scores"]].index(chosen["selected_C"])
                    ]["mean_calendar_year_average_precision"],
                    "test_roc_auc": test_cls["roc_auc"], "test_brier": test_cls["brier"],
                    "validation_false_alarms": val_stats["false_alarms"],
                    "validation_coin_months": val_stats["coin_months"],
                    "validation_false_alarms_per_coin_month": val_stats[
                        "false_alarms_per_coin_month"],
                    "validation_episode_recall": val_stats["episode_recall"],
                    "test_targetable_episodes": len(target_ids),
                    "test_episodes_caught": len(caught & target_ids),
                    "test_episode_recall": len(caught & target_ids) / len(target_ids)
                    if target_ids else np.nan,
                    "test_alarm_events": len(events),
                    "test_true_alarms": int(events.is_true_alarm.sum()) if len(events) else 0,
                    "test_false_alarms": false_count, "test_coin_months": coin_months,
                    "test_false_alarms_per_coin_month": false_rate,
                    "max_false_alarms_in_coin_month": max_false,
                    "test_flagged_hours": len(flagged),
                })

                stem = f"logreg_t{tier}_h{horizon}_{year}_{args.tag}"
                joblib.dump(model, model_dir / f"{stem}.joblib")
                meta = {
                    "dataset_version": fold["dataset_version"],
                    "model": "logreg", "tier": tier, "horizon": horizon,
                    "test_year": year, "tag": args.tag, "tune": True,
                    "model_params": {"C": chosen["selected_C"], "penalty": "l1"},
                    "selection_metric": chosen["selection_metric"],
                    "validation_years": chosen["validation_years"],
                    "skipped_years": chosen["skipped_years"],
                    "candidate_scores": chosen["candidate_scores"],
                    "threshold": threshold,
                    "false_alarm_budget_per_coin_month": args.false_alarms_per_coin_month,
                    "outer_train_rows": int(len(X_train)),
                    "outer_train_positives": int(y_train.sum()),
                    "test_rows": int(len(X_test)),
                    "test_positives": int(y_test.sum()),
                }
                (model_dir / f"{stem}.json").write_text(json.dumps(meta, indent=2))
                (tune_dir / f"{stem}.json").write_text(json.dumps(meta, indent=2))

                tuning_record = {
                    "tier": tier, "horizon": horizon, "test_year": year,
                    "tag": args.tag, **chosen,
                }
                tuning_records.append(tuning_record)
                names = list(model.cols_)
                coefs = np.asarray(model.clf_.coef_).ravel()
                for feature, coef in zip(names, coefs):
                    coefficient_records.append({
                        "dataset_version": fold["dataset_version"],
                        "tier": tier, "horizon": horizon, "test_year": year,
                        "tag": args.tag, "C": chosen["selected_C"],
                        "penalty": "l1", "feature": feature,
                        "is_onchain": "logdiff" in feature,
                        "coef": float(coef), "abs_coef": float(abs(coef)),
                    })
                print(f"  selected C={chosen['selected_C']} (L1); CV years "
                      f"{chosen['validation_years']}; skipped {chosen['skipped_years']}")

    pooled = pd.concat(all_test_predictions, ignore_index=True)
    # Save/evaluate predictions independently for every requested tier/horizon.
    for tier in tiers:
        for horizon in horizons:
            run_pooled = pooled[(pooled["tier"] == tier) &
                                (pooled["horizon"] == horizon)].copy()
            for year, year_part in run_pooled.groupby("year", sort=True):
                path = pred_dir / f"logreg_t{tier}_h{horizon}_{int(year)}_{args.tag}.parquet"
                year_part.to_parquet(path, index=False)
            pooled_path = pred_dir / f"logreg_t{tier}_h{horizon}_{args.tag}_pooled_test.parquet"
            run_pooled.to_parquet(pooled_path, index=False)
            result = evaluate_run(
                run_pooled, tag=f"logreg_t{tier}_h{horizon}_{args.tag}",
                out_dir=RESULTS / "evaluation")
            print_report(result)

    _write_reporting(tuning_records, coefficient_records)
    _append_final_results(pooled, yearly_rows, event_frames, args.tag,
                          args.false_alarms_per_coin_month)
    print(f"\nSaved tuning, coefficients, and slide summaries under {REPORTS}")
    print(f"Saved models, predictions, and evaluation curves under {RESULTS}")
    print(f"Appended LogReg results to {REPO / 'reports' / 'final_results'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
