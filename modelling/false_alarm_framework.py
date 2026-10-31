"""Run fixed baseline and ML models across tiers and horizons; report OOS metrics.

Run from the repository root:
    python3 modelling/false_alarm_framework.py

Each fold model is fitted on inner_train with fixed parameters, then scores
inner_val and test. XGBoost uses exactly the configured tree count without
early stopping; inner_val selects the alarm cutoff under the false-alarm
budget, which is then evaluated on the held-out test year.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "data_processing"))

from config import FORECAST_HORIZONS_HOURS, PROCESSED_DATA_DIR  # noqa: E402
from modelling.alarm_events import build_alarm_events  # noqa: E402
from modelling.evaluate import pooled_metrics  # noqa: E402
from modelling.models import MODEL_REGISTRY, get_model  # noqa: E402
from modelling.models.base import MedianImputer  # noqa: E402
from modelling.prepare_data import load_folds  # noqa: E402
from modelling.rolling_threshold import (  # noqa: E402
    _choose_threshold, _targetable_episode_ids, _observed_coin_months,
    _false_alarm_stats,
)

_FOLDS_CACHE: dict[tuple[int, int], list[dict]] = {}
XGBOOST_FIXED_PARAMS = {"max_depth": 3, "learning_rate": 0.05, "n_estimators": 300}
RANDOM_FOREST_FIXED_PARAMS = {
    "n_estimators": 500, "max_depth": 8, "min_samples_leaf": 50,
    "max_features": "sqrt", "class_weight": "balanced_subsample",
    "random_state": 42,
}
FIXED_TAGS = {
    "baseline": "base_fixed", "persistence": "pers_fixed",
    "xgboost": "xgb_d3_lr005_300", "random_forest": "rf_500_d8_l50",
}


def score_fold(fold: dict, model_name: str,
               model_params: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Fit one fold model and score validation/test with the same fitted model."""
    if model_params is None:
        model_params = {
            "xgboost": XGBOOST_FIXED_PARAMS,
            "random_forest": RANDOM_FOREST_FIXED_PARAMS,
        }.get(model_name, {})
    model = get_model(model_name, **model_params)
    X_train, X_val, X_test = fold["X_inner_train"], fold["X_inner_val"], fold["X_test"]
    if model.needs_imputation:
        imputer = MedianImputer().fit(X_train)
        X_train, X_val, X_test = (
            imputer.transform(X_train), imputer.transform(X_val), imputer.transform(X_test)
        )
    # Validation scores select the alarm cutoff; they do not select tree count.
    model.fit(X_train, fold["y_inner_train"])
    return model.predict_proba(X_val)[:, 1], model.predict_proba(X_test)[:, 1]


def evaluate_configuration(model_name: str, tier: int, horizon: int,
                           episodes: pd.DataFrame, budget: float):
    year_rows, event_frames, score_frames = [], [], []
    cache_key = (tier, horizon)
    if cache_key not in _FOLDS_CACHE:
        _FOLDS_CACHE[cache_key] = load_folds(horizon=horizon, tier=tier)

    for fold in _FOLDS_CACHE[cache_key]:
        year = int(fold["year"])
        val_scores, test_scores = score_fold(fold, model_name)
        val = fold["inner_val_meta"].copy()
        val["horizon"] = horizon
        val["y_true"] = fold["y_inner_val"].reset_index(drop=True)
        val["y_score"] = val_scores
        if model_name == "baseline":
            # The constant score flags every row at its only finite candidate,
            # which cannot satisfy the budget; +inf is the feasible cutoff.
            threshold = float("inf")
            val_stats = {
                "false_alarms": 0,
                "coin_months": _observed_coin_months(val),
                "false_alarms_per_coin_month": 0.0,
                "episode_recall": 0.0,
            }
        else:
            threshold, val_stats = _choose_threshold(val, episodes, horizon, budget)

        test = fold["test_meta"].copy()
        test["hour"] = pd.to_datetime(test["hour"], utc=True)
        test["horizon"] = horizon
        test["split"] = "test"
        test["y_true"] = fold["y_test"].reset_index(drop=True)
        test["y_score"] = test_scores
        flagged, events = build_alarm_events(test, episodes, threshold)
        target_ids = _targetable_episode_ids(test, episodes, horizon)
        caught = (
            set(events.loc[events.is_true_alarm, "target_episode_id"].dropna().astype(str))
            if not events.empty else set()
        )
        false_count, coin_months, false_rate, max_false = _false_alarm_stats(
            events, test, _observed_coin_months(test)
        )
        event_frames.append(events.assign(
            test_year=year, model=model_name, tag=FIXED_TAGS[model_name],
            model_configuration=FIXED_TAGS[model_name], tier=tier, horizon=horizon
        ))
        score_frames.append(pd.DataFrame({"y_true": test["y_true"], "y_score": test_scores}))
        cls = pooled_metrics(test["y_true"], test_scores)
        val_cls = pooled_metrics(fold["y_inner_val"], val_scores)
        year_rows.append({
            "model": model_name, "tier": tier,
            "tag": FIXED_TAGS[model_name],
            "model_configuration": FIXED_TAGS[model_name],
            "feature_set": "market" if tier == 1 else "market + on-chain",
            "horizon_hours": horizon, "validation_year": year - 1,
            "test_year": year, "threshold": threshold,
            "test_rows": cls["n"], "test_positive_hours": cls["n_pos"],
            "test_positive_rate": cls["pos_rate"], "test_pr_auc": cls["pr_auc"],
            "test_average_precision": cls["pr_auc"],
            "validation_rows": val_cls["n"],
            "validation_positive_hours": val_cls["n_pos"],
            "validation_positive_base_rate": val_cls["pos_rate"],
            "validation_average_precision": val_cls["pr_auc"],
            "test_roc_auc": cls["roc_auc"], "test_brier": cls["brier"],
            "validation_false_alarms": val_stats["false_alarms"],
            "validation_coin_months": val_stats["coin_months"],
            "validation_false_alarms_per_coin_month": val_stats["false_alarms_per_coin_month"],
            "validation_episode_recall": val_stats["episode_recall"],
            "test_targetable_episodes": len(target_ids),
            "test_episodes_caught": len(caught & target_ids),
            "test_episode_recall": len(caught & target_ids) / len(target_ids) if target_ids else np.nan,
            "test_alarm_events": len(events),
            "test_true_alarms": int(events.is_true_alarm.sum()) if len(events) else 0,
            "test_false_alarms": false_count, "test_coin_months": coin_months,
            "test_false_alarms_per_coin_month": false_rate,
            "max_false_alarms_in_coin_month": max_false,
            "test_flagged_hours": len(flagged),
        })
    return year_rows, event_frames, score_frames


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "reports/final_results"))
    parser.add_argument("--false-alarms-per-coin-month", type=float, default=1.0)
    parser.add_argument("--models", nargs="+", choices=sorted(MODEL_REGISTRY),
                        default=["baseline", "persistence", "xgboost", "random_forest"])
    parser.add_argument("--tiers", nargs="+", type=int, choices=(1, 2), default=(1, 2))
    parser.add_argument("--horizons", nargs="+", type=int, choices=(1, 6, 24), default=(1, 6, 24))
    args = parser.parse_args(argv)
    if args.false_alarms_per_coin_month < 0:
        parser.error("--false-alarms-per-coin-month must be non-negative")

    episodes = pd.read_parquet(PROCESSED_DATA_DIR / "depeg_episodes.parquet")
    episodes["start"] = pd.to_datetime(episodes["start"], utc=True)
    all_rows, all_events, pooled_scores = [], [], {}
    configs = [(m, t, h) for m in args.models for t in args.tiers
               for h in args.horizons]
    for i, (model, tier, horizon) in enumerate(configs, 1):
        print(f"[{i}/{len(configs)}] {model}, tier {tier}, horizon {horizon}h", flush=True)
        rows, events, score_frames = evaluate_configuration(
            model, tier, horizon, episodes, args.false_alarms_per_coin_month
        )
        all_rows.extend(rows)
        all_events.extend(events)
        pooled_scores[(model, tier, horizon)] = pd.concat(score_frames, ignore_index=True)

    yearly = pd.DataFrame(all_rows)
    keys = ["model", "model_configuration", "tier", "feature_set", "horizon_hours"]
    alarm = yearly.groupby(keys, as_index=False).agg(
        test_years=("test_year", "nunique"),
        episodes_caught=("test_episodes_caught", "sum"),
        targetable_episodes=("test_targetable_episodes", "sum"),
        true_alarm_events=("test_true_alarms", "sum"),
        false_alarms=("test_false_alarms", "sum"),
        alarm_events=("test_alarm_events", "sum"),
        coin_months=("test_coin_months", "sum"),
        flagged_hours=("test_flagged_hours", "sum"),
        max_false_alarms_in_coin_month=("max_false_alarms_in_coin_month", "max"),
    )
    alarm["episode_recall"] = alarm.episodes_caught / alarm.targetable_episodes.replace(0, np.nan)
    alarm["false_alarms_per_coin_month"] = alarm.false_alarms / alarm.coin_months.replace(0, np.nan)
    alarm["alarm_precision"] = alarm.true_alarm_events / alarm.alarm_events.replace(0, np.nan)
    alarm["missed_episodes"] = alarm.targetable_episodes - alarm.episodes_caught

    metric_rows = []
    for (model, tier, horizon), scores in pooled_scores.items():
        metric_rows.append({
            "model": model, "tier": tier,
            "tag": FIXED_TAGS[model],
            "model_configuration": FIXED_TAGS[model],
            "feature_set": "market" if tier == 1 else "market + on-chain",
            "horizon_hours": horizon, **pooled_metrics(scores.y_true, scores.y_score),
        })
    metrics = pd.DataFrame(metric_rows).rename(columns={
        "n": "test_rows", "n_pos": "test_positive_hours",
        "pos_rate": "positive_base_rate", "pr_auc": "average_precision",
        "roc_auc": "roc_auc", "brier": "brier",
    })
    metrics["average_precision_method"] = "sklearn.average_precision_score"
    metrics["alarm_budget_fa_per_coin_month"] = args.false_alarms_per_coin_month
    summary = metrics.merge(alarm, on=keys, how="left")

    events = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    if not events.empty:
        caught = events.loc[events.target_episode_id.notna()].copy()
        caught["horizon_hours"] = caught.horizon
        leads = caught.groupby(["model", "tier", "horizon_hours"], as_index=False).lead_hours.median()
        summary = summary.merge(leads.rename(columns={"lead_hours": "median_lead_hours"}),
                                on=["model", "tier", "horizon_hours"], how="left")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary.sort_values(keys).to_csv(out / "model_comparison_summary.csv", index=False)
    yearly.sort_values(keys + ["test_year"]).to_csv(out / "model_comparison_by_year.csv", index=False)
    events.to_csv(out / "model_alarm_events.csv", index=False)
    (out / "run_metadata.txt").write_text(
        "Expanding-window evaluation; standard downward depeg definition: price "
        "below $0.99 for at least 2 consecutive hours; episodes merged using "
        "the configured 24-hour gap. Each test year is scored by a model fit "
        "on inner_train; inner_val selects the alarm threshold; the held-out "
        "test year provides reported metrics. XGBoost uses exactly 300 trees, "
        "without early stopping. Average precision uses sklearn.average_precision_score.\n\n"
        f"False-alarm budget: {args.false_alarms_per_coin_month:g} per observed coin-month.\n"
        f"Models: {', '.join(args.models)}\nTiers: {', '.join(map(str, args.tiers))}\n"
        f"Horizons (hours): {', '.join(map(str, args.horizons))}\n"
        f"Fixed XGBoost parameters: {XGBOOST_FIXED_PARAMS}\n"
        f"Fixed random-forest parameters: {RANDOM_FOREST_FIXED_PARAMS}\n",
        encoding="utf-8",
    )
    print(f"Saved {out / 'model_comparison_summary.csv'}")
    print(f"Saved {out / 'model_comparison_by_year.csv'}")
    print(f"Saved {out / 'model_alarm_events.csv'}")
    print(f"Saved {out / 'run_metadata.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
