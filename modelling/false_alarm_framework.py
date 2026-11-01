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
from config import ACTIVE_EPISODES_PATH  # versioned modelling inputs
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
            dataset_version=fold["dataset_version"],
            test_year=year, model=model_name, tag=FIXED_TAGS[model_name],
            model_configuration=FIXED_TAGS[model_name], tier=tier, horizon=horizon
        ))
        score_frames.append(pd.DataFrame({"y_true": test["y_true"], "y_score": test_scores}))
        cls = pooled_metrics(test["y_true"], test_scores)
        val_cls = pooled_metrics(fold["y_inner_val"], val_scores)
        year_rows.append({
            "dataset_version": fold["dataset_version"],
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


def evaluate_tier1_on_common_sample(
    model_name: str, horizon: int, episodes: pd.DataFrame, budget: float,
) -> dict:
    """Refit Tier 1 on Tier 2 eligible rows and return pooled OOS metrics."""
    tier1_folds = load_folds(horizon=horizon, tier=1)
    tier2_folds = load_folds(horizon=horizon, tier=2)
    tier2_by_year = {int(fold["year"]): fold for fold in tier2_folds}
    years, event_frames, score_frames = [], [], []

    for original in tier1_folds:
        year = int(original["year"])
        if year not in tier2_by_year:
            continue
        fold = _common_sample_fold(original, tier2_by_year[year])
        val_scores, test_scores = score_fold(fold, model_name)

        val = fold["inner_val_meta"].copy()
        val["horizon"] = horizon
        val["y_true"] = fold["y_inner_val"].reset_index(drop=True)
        val["y_score"] = val_scores
        if model_name == "baseline":
            threshold = float("inf")
            val_stats = {"false_alarms_per_coin_month": 0.0, "episode_recall": 0.0}
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
        years.append({
            "test_year": year,
            "test_rows": len(test),
            "test_positive_hours": int(test["y_true"].sum()),
            "positive_base_rate": float(test["y_true"].mean()) if len(test) else np.nan,
            "validation_false_alarms_per_coin_month": val_stats["false_alarms_per_coin_month"],
            "validation_episode_recall": val_stats["episode_recall"],
            "episodes_caught": len(caught & target_ids),
            "targetable_episodes": len(target_ids),
            "alarm_events": len(events),
            "true_alarm_events": int(events.is_true_alarm.sum()) if not events.empty else 0,
            "false_alarms": false_count,
            "coin_months": coin_months,
            "false_alarms_per_coin_month": false_rate,
            "max_false_alarms_in_coin_month": max_false,
            "flagged_hours": len(flagged),
        })
        if not events.empty:
            event_frames.append(events)
        score_frames.append(pd.DataFrame({
            "y_true": test["y_true"].to_numpy(), "y_score": test_scores,
        }))

    if not score_frames:
        raise RuntimeError(f"No common-sample folds available for {model_name}, h={horizon}")
    yearly = pd.DataFrame(years)
    scores = pd.concat(score_frames, ignore_index=True)
    cls = pooled_metrics(scores["y_true"], scores["y_score"])
    events = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame()
    caught_events = events.loc[events.target_episode_id.notna()] if not events.empty else events
    false_total = int(yearly.false_alarms.sum())
    coin_month_total = int(yearly.coin_months.sum())
    alarm_total = int(yearly.alarm_events.sum())
    true_total = int(yearly.true_alarm_events.sum())
    target_total = int(yearly.targetable_episodes.sum())
    caught_total = int(yearly.episodes_caught.sum())
    return {
        "model": model_name,
        "tag": FIXED_TAGS[model_name],
        "horizon_hours": horizon,
        "test_rows": int(cls["n"]),
        "positive_base_rate": float(cls["pos_rate"]),
        "average_precision": float(cls["pr_auc"]),
        "episodes_caught": caught_total,
        "targetable_episodes": target_total,
        "episode_recall": caught_total / target_total if target_total else np.nan,
        "alarm_budget_fa_per_coin_month": budget,
        "false_alarms": false_total,
        "coin_months": coin_month_total,
        "false_alarms_per_coin_month": false_total / coin_month_total if coin_month_total else np.nan,
        "alarm_events": alarm_total,
        "true_alarm_events": true_total,
        "alarm_precision": true_total / alarm_total if alarm_total else np.nan,
        "median_lead_hours": (
            float(caught_events["lead_hours"].median())
            if not caught_events.empty else np.nan
        ),
        "max_false_alarms_in_coin_month": int(yearly.max_false_alarms_in_coin_month.max()),
        "test_years": int(yearly.test_year.nunique()),
    }


def write_tier_comparison(
    summary: pd.DataFrame, common_tier1: pd.DataFrame, out_path: Path,
) -> None:
    """Write a wide, paired Tier 1 vs Tier 2 table on the common sample."""
    metrics = [
        "test_rows", "positive_base_rate", "average_precision",
        "episodes_caught", "targetable_episodes", "episode_recall",
        "false_alarms", "coin_months", "false_alarms_per_coin_month",
        "alarm_budget_fa_per_coin_month", "alarm_events", "true_alarm_events",
        "alarm_precision", "median_lead_hours", "max_false_alarms_in_coin_month",
    ]
    tier1 = common_tier1[["model", "tag", "horizon_hours", *metrics]].copy()
    tier2 = summary.loc[summary.tier.eq(2), [
        "model", "tag", "horizon_hours", *metrics,
    ]].copy()
    tier1 = tier1.rename(columns={c: f"tier1_{c}" for c in metrics})
    tier2 = tier2.rename(columns={c: f"tier2_{c}" for c in metrics})
    paired = tier1.merge(tier2, on=["model", "tag", "horizon_hours"],
                          how="inner", validate="one_to_one")
    paired["common_test_rows"] = paired["tier1_test_rows"]
    paired["common_targetable_episodes"] = paired["tier1_targetable_episodes"]
    for metric in (
        "positive_base_rate", "average_precision", "episodes_caught",
        "episode_recall", "false_alarms_per_coin_month", "alarm_precision",
        "median_lead_hours",
    ):
        paired[f"delta_{metric}_tier2_minus_tier1"] = (
            paired[f"tier2_{metric}"] - paired[f"tier1_{metric}"]
        )
    paired["common_sample"] = "Tier 2 eligible coin-hours; Tier 1 refit on same rows"
    paired["dataset_version"] = "corrected_v1"
    paired.sort_values(["model", "horizon_hours"]).to_csv(out_path, index=False)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "reports/final_results"))
    parser.add_argument("--false-alarms-per-coin-month", type=float, default=1.0)
    parser.add_argument("--models", nargs="+", choices=sorted(MODEL_REGISTRY),
                        default=["baseline", "persistence", "xgboost", "random_forest"])
    parser.add_argument("--tiers", nargs="+", type=int, choices=(1, 2), default=(1, 2))
    parser.add_argument("--horizons", nargs="+", type=int, choices=(1, 6, 24), default=(1, 6, 24))
    parser.add_argument(
        "--common-tier-comparison-only", action="store_true",
        help="create model_comparison_by_tier.csv from existing final results without rerunning all models",
    )
    args = parser.parse_args(argv)
    if args.false_alarms_per_coin_month < 0:
        parser.error("--false-alarms-per-coin-month must be non-negative")

    out = Path(args.out_dir)
    if args.common_tier_comparison_only:
        if 2 not in args.tiers:
            parser.error("--common-tier-comparison-only requires tier 2 in --tiers")
        summary_path = out / "model_comparison_summary.csv"
        if not summary_path.exists():
            raise FileNotFoundError(
                f"{summary_path} is missing; run the full evaluation first"
            )
        summary = pd.read_csv(summary_path)
        episodes = pd.read_parquet(ACTIVE_EPISODES_PATH)
        episodes["start"] = pd.to_datetime(episodes["start"], utc=True)
        common_rows = [
            evaluate_tier1_on_common_sample(
                model, horizon, episodes, args.false_alarms_per_coin_month,
            )
            for model in args.models for horizon in args.horizons
        ]
        write_tier_comparison(summary, pd.DataFrame(common_rows),
                              out / "model_comparison_by_tier.csv")
        print(f"Saved {out / 'model_comparison_by_tier.csv'}")
        return 0

    episodes = pd.read_parquet(ACTIVE_EPISODES_PATH)
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
            "dataset_version": "corrected_v1",
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

    out.mkdir(parents=True, exist_ok=True)
    if 2 in args.tiers:
        common_tier1_rows = []
        for model in args.models:
            for horizon in args.horizons:
                print(
                    f"Common-sample Tier 1 refit: {model}, horizon {horizon}h",
                    flush=True,
                )
                common_tier1_rows.append(evaluate_tier1_on_common_sample(
                    model, horizon, episodes, args.false_alarms_per_coin_month,
                ))
        write_tier_comparison(
            summary, pd.DataFrame(common_tier1_rows),
            out / "model_comparison_by_tier.csv",
        )
    summary.sort_values(keys).to_csv(out / "model_comparison_summary.csv", index=False)
    yearly.sort_values(keys + ["test_year"]).to_csv(out / "model_comparison_by_year.csv", index=False)
    events.to_csv(out / "model_alarm_events.csv", index=False)
    (out / "run_metadata.txt").write_text(
        "Dataset: corrected_v1; master_corrected_v1.parquet and "
        "depeg_episodes_corrected_v1.parquet; bar-close timestamps, "
        "new merged episode targets, causal recovery exclusion.\n"
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
        f"Fixed random-forest parameters: {RANDOM_FOREST_FIXED_PARAMS}\n"
        + ("model_comparison_by_tier.csv refits Tier 1 on Tier 2 eligible "
           "coin-hours within each fold so both tiers share train, validation, "
           "and test observations.\n" if 2 in args.tiers else ""),
        encoding="utf-8",
    )
    print(f"Saved {out / 'model_comparison_summary.csv'}")
    print(f"Saved {out / 'model_comparison_by_year.csv'}")
    print(f"Saved {out / 'model_alarm_events.csv'}")
    if 2 in args.tiers:
        print(f"Saved {out / 'model_comparison_by_tier.csv'}")
    print(f"Saved {out / 'run_metadata.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
