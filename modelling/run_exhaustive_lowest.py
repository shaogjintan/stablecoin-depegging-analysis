"""Calibrate corrected RF/XGB with exhaustive lowest-feasible thresholds.

python -m modelling.run_exhaustive_lowest --xgb-run runs/XGB-NEW

XGB scores are frozen. RF fits the same inner_train rows with the shared
fixed configuration. Validation only selects cutoffs. The maximum-recall
policy is evaluated alongside as a reference using identical test scores.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import joblib
import pandas as pd
from modelling.run_summary import aggregate
from modelling.alarm_events import build_alarm_events
from modelling.corrected_lr_rf import RF_PARAMS
from modelling.models import get_model
from modelling.models.base import MedianImputer
from modelling.rolling_threshold import _choose_threshold, _targetable_episode_ids, _false_alarm_stats, _observed_coin_months

RF_PARAMS = {**RF_PARAMS, "n_jobs": -1}  # Worker count only; same statistical configuration.


def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n")




def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xgb-run", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/EXHAUSTIVE-LOWEST-NEW"))
    parser.add_argument("--horizon", type=int, action="append", choices=[1, 6, 24])
    parser.add_argument("--configs", nargs="+",
                        choices=["market_original", "market_matched", "market_onchain"],
                        default=["market_matched", "market_onchain"],
                        help="default compares both tiers on identical eligible samples")
    parser.add_argument("--resume", action="store_true", help="reuse completed fits in this output directory")
    args = parser.parse_args()
    source = json.loads((args.xgb_run / "run.json").read_text())
    if source.get("status") != "complete":
        raise ValueError("XGB source run must be complete")
    args.out_dir.mkdir(parents=True, exist_ok=args.resume)
    existing = (pd.read_csv(args.out_dir / "by_year.csv")
                if args.resume and (args.out_dir / "by_year.csv").exists()
                else pd.DataFrame())
    if args.resume:
        previous = json.loads((args.out_dir / "run.json").read_text())
        if previous["source_run_id"] != source["run_id"]:
            raise ValueError("Resume source run does not match")
    shutil.copy2(args.xgb_run / "episodes.parquet", args.out_dir / "episodes.parquet")
    episodes = pd.read_parquet(args.out_dir / "episodes.parquet")
    metadata = dict(
        status="running", started_utc=datetime.now(timezone.utc).isoformat(),
        source_xgb_run=str(args.xgb_run), source_run_id=source["run_id"],
        source_manifest_sha256=hashlib.sha256((args.xgb_run / "run.json").read_bytes()).hexdigest(),
        dataset_version=source.get("dataset_version"), python=sys.version,
        configs=args.configs, horizons=args.horizon or source.get("horizons", [1, 6, 24]),
        packages={p: importlib.metadata.version(p) for p in
                  ["numpy", "pandas", "scikit-learn", "xgboost", "numba"]},
        policy="exhaustive_lowest", reference_policy="exact_max_recall",
        budget_fa_per_coin_month=1., rf_params=RF_PARAMS,
        calibration="inner_val only; no refit after calibration",
        xgboost="reuse saved scores; fixed 300 trees",
        random_forest="fit saved inner_train rows; training-only imputation",
    )
    save_json(args.out_dir / "run.json", metadata)
    rows, event_frames, scores = [], [], {}
    folders = sorted((args.xgb_run / "fits").iterdir())
    for number, folder in enumerate(folders, 1):
        info = json.loads((folder / "fit.json").read_text())
        h, config, year = info["horizon"], info["config"], info["year"]
        if config not in args.configs:
            continue
        if args.horizon and h not in args.horizon:
            continue
        val_raw = pd.read_parquet(folder / "inner_val.parquet")
        test_raw = pd.read_parquet(folder / "test.parquet")
        train = None
        for name in ["xgboost", "random_forest"]:
            fit_dir = args.out_dir / "fits" / f"{name}_{folder.name}"
            fit_dir.mkdir(parents=True, exist_ok=args.resume)
            done = (existing.loc[
                existing.model.eq(name) & existing.config.eq(config)
                & existing.horizon_hours.eq(h) & existing.test_year.eq(year)]
                if not existing.empty else existing)
            if len(done) == 2:
                rows.extend(done.to_dict("records"))
                for policy in done.policy:
                    event_frames.append(pd.read_csv(fit_dir / f"alarms_{policy}.csv",
                                                    parse_dates=["alarm_hour"]))
                key = (name, config, h)
                scores[key] = pd.concat([scores.get(key, pd.DataFrame()),
                    pd.read_parquet(fit_dir / "test.parquet")[["y_true", "y_score"]]],
                    ignore_index=True)
                print(f"[{number}/{len(folders)}] {name} {folder.name}: reused", flush=True)
                continue
            val = val_raw[["coin", "hour", "episode_id", "horizon", "y_true", "y_score"]].copy()
            test = test_raw[["coin", "hour", "episode_id", "horizon", "y_true", "y_score"]].copy()
            if name == "random_forest":
                train = pd.read_parquet(folder / "inner_train.parquet")
                features = info["features"]
                if args.resume and (fit_dir / "pipeline.joblib").exists():
                    fitted = joblib.load(fit_dir / "pipeline.joblib")
                    model, imputer = fitted["model"], fitted["imputer"]
                else:
                    imputer = MedianImputer().fit(train[features])
                    model = get_model(name, **RF_PARAMS).fit(
                        imputer.transform(train[features]), train.y_true)
                val["y_score"] = model.predict_proba(imputer.transform(val_raw[features]))[:, 1]
                test["y_score"] = model.predict_proba(imputer.transform(test_raw[features]))[:, 1]
                joblib.dump(dict(model=model, imputer=imputer), fit_dir / "pipeline.joblib")
            val.to_parquet(fit_dir / "inner_val.parquet", index=False)
            test.to_parquet(fit_dir / "test.parquet", index=False)
            key = (name, config, h)
            scores[key] = pd.concat([scores.get(key, pd.DataFrame()),
                                     test[["y_true", "y_score"]]], ignore_index=True)
            months = _observed_coin_months(val)
            choices = {
                policy: _choose_threshold(val, episodes, h, 1., policy=policy)
                for policy in ("exhaustive_lowest", "exact_max_recall")
            }
            targets = _targetable_episode_ids(test, episodes, h)
            validation_targets = _targetable_episode_ids(val, episodes, h)
            thresholds = {}
            for policy, (threshold, validation_stats) in choices.items():
                _, ve = build_alarm_events(val, episodes, threshold)
                vf, _, vr, _ = _false_alarm_stats(ve, val)
                vc = len(set(ve.target_episode_id.dropna().astype(str)) & validation_targets)
                assert vf == validation_stats["false_alarms"] and vc == validation_stats["episodes_warned"] and vr <= 1.
                _, events = build_alarm_events(test, episodes, threshold)
                false, coin_months, rate, _ = _false_alarm_stats(events, test)
                caught = len(set(events.target_episode_id.dropna().astype(str)) & targets)
                events = events.assign(model=name, config=config, horizon_hours=h,
                                       policy=policy, test_year=year)
                event_frames.append(events)
                events.to_csv(fit_dir / f"alarms_{policy}.csv", index=False)
                rows.append(dict(
                    model=name, config=config, horizon_hours=h, test_year=year,
                    policy=policy, threshold=threshold, false_alarms=false,
                    coin_months=coin_months, false_alarms_per_coin_month=rate,
                    episodes_caught=caught, targetable_episodes=len(targets),
                    alarm_events=len(events), true_alarms=int(events.is_true_alarm.sum()),
                    test_rows=len(test), validation_false_alarms=vf,
                    validation_coin_months=months, validation_episodes_caught=vc,
                    validation_targetable_episodes=len(validation_targets)))
                thresholds[policy] = dict(threshold=threshold, validation_false_alarms=vf,
                                          validation_episodes_caught=vc)
            save_json(fit_dir / "fit.json", dict(
                model=name, source_fit=str(folder), features=info["features"],
                config=config, horizon=h, year=year, fit_partition="inner_train",
                calibration_partition="inner_val", thresholds=thresholds))
            pd.DataFrame(rows).to_csv(args.out_dir / "by_year.csv", index=False)
            print(f"[{number}/{len(folders)}] {name} {folder.name}: complete", flush=True)
    yearly, summary, events = aggregate(rows, event_frames, scores)
    yearly.to_csv(args.out_dir / "by_year.csv", index=False)
    summary.to_csv(args.out_dir / "summary_all_policies.csv", index=False)
    selected = summary.loc[summary.policy.eq("exhaustive_lowest")]
    selected.to_csv(args.out_dir / "summary.csv", index=False)
    if set(args.configs) == {"market_matched", "market_onchain"}:
        comparison = selected.pivot(
            index=["model", "horizon_hours"], columns="config",
            values=["episodes_caught", "targetable_episodes", "false_alarms",
                    "false_alarms_per_coin_month", "episode_recall",
                    "average_precision", "median_lead_hours"])
        comparison.columns = [
            ("tier1_" if config == "market_matched" else "tier2_") + metric
            for metric, config in comparison.columns]
        comparison.reset_index().to_csv(args.out_dir / "comparison_by_tier.csv", index=False)
    events.to_csv(args.out_dir / "alarm_events.csv", index=False)
    metadata.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat(),
                    fit_count=len(rows) // 2)
    save_json(args.out_dir / "run.json", metadata)
    print(summary.loc[summary.policy.eq("exhaustive_lowest")].to_string(index=False))


if __name__ == "__main__":
    main()
