"""Select a cutoff with a practice model, then refit before the outer test.

python -m modelling.training.train_models --out-dir \
artifacts/runs/primary/ALL-MODELS-REPRODUCED

Uses corrected folds, matched tiers, and the existing purge/embargo rules.
Previous runs are never reused as final-model scores or overwritten.
"""
import argparse
from datetime import datetime, timezone
import importlib.metadata
from pathlib import Path
import shutil
import uuid

import joblib
import numpy as np
import pandas as pd

from data_processing.config import MODELLING_DATA_VERSION, PURGE_HOURS, EMBARGO_HOURS
from data_processing.splits import load_master, active_input_paths
from modelling.evaluation.alarm_events import build_alarm_events
from data_processing.sample_preparation import corrected_folds
from modelling.training.logistic_tuning import fit_lr, tune_lr
from modelling.evaluation.metrics import metrics
from modelling.training.parameters import XGBOOST_FIXED_PARAMS, RANDOM_FOREST_FIXED_PARAMS
from modelling.utils.io import sha256, save_json
from data_processing.matched_sample import _common_sample_fold
from modelling.models import get_model
from modelling.models.base import MedianImputer
from modelling.evaluation.threshold_selection import _choose_threshold, _targetable_episode_ids
from modelling.evaluation.run_summary import aggregate


def fit(name, X, y, C=None):
    """Fit the selected model and its training-derived preprocessing."""
    if name == "logreg":
        return fit_lr(X, y, C)
    params = XGBOOST_FIXED_PARAMS if name == "xgboost" else RANDOM_FOREST_FIXED_PARAMS if name == "random_forest" else {}
    model = get_model(name, **params)
    imputer = MedianImputer().fit(X) if model.needs_imputation else None
    model.fit(imputer.transform(X) if imputer is not None else X, y)
    return model, imputer


def predict(bundle, X):
    """Predict positive-class scores using a fitted model and its imputer."""
    model, imputer = bundle
    return model.predict_proba(imputer.transform(X) if imputer is not None else X)[:, 1]


def frame(fold, part):
    """Build a partition's prediction metadata and horizon-specific labels."""
    result = fold[f"{part}_meta"].copy()
    result["y_true"] = fold[f"y_{part}"]
    result["horizon"] = fold["horizon"]
    return result


def main():
    """Run practice fitting, calibration, final refitting, and scoring."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["random_forest", "xgboost", "logreg"],
                        choices=["random_forest", "xgboost", "logreg", "baseline", "persistence"],
                        help="models to fit (default: random_forest xgboost logreg)")
    parser.add_argument("--horizon", type=int, action="append", choices=[1, 6, 24],
                        help="forecast horizon; repeat to select multiple (default: 1, 6, 24)")
    parser.add_argument("--year", type=int, action="append", choices=range(2021, 2026))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    run_id = str(uuid.uuid4())
    info = dict(run_id=run_id, status="running", started_utc=datetime.now(timezone.utc).isoformat(),
                dataset_version=MODELLING_DATA_VERSION, models=args.models,
                horizons=args.horizon or [1, 6, 24], years=args.year or list(range(2021, 2026)),
                configs=["market_matched", "market_onchain"], policy="exhaustive_lowest",
                refit_after_calibration=True, budget_fa_per_coin_month=1.,
                purge_hours=PURGE_HOURS, embargo_hours=EMBARGO_HOURS,
                protocol="practice inner_train -> calibrate inner_val -> refit train -> test; cutoff unchanged",
                logreg_selection="calendar-year CV inside inner_train; keep selected C for refit",
                rf_prediction_summation="serial tree order; fitting uses fixed n_jobs",
                packages={p: importlib.metadata.version(p) for p in
                          ["numpy", "pandas", "scikit-learn", "xgboost"]},
                input_sha256={str(p): sha256(p) for p in active_input_paths()})
    save_json(args.out_dir / "run.json", info)
    source = args.out_dir / "source"
    source.mkdir()
    root = Path(__file__).resolve().parents[2]
    for name in ["modelling", "data_processing"]:
        shutil.copytree(root / name, source / name, ignore=shutil.ignore_patterns("__pycache__"))
    rows, events_all, scores = [], [], {}
    try:
        master, episodes = load_master()
        episodes.to_parquet(args.out_dir / "episodes.parquet", index=False)
        for h in info["horizons"]:
            tier1, tier2 = corrected_folds(master, episodes, h, 1), corrected_folds(master, episodes, h, 2)
            for a, b in zip(tier1, tier2):
                for config, fold in [("market_matched", _common_sample_fold(a, b)), ("market_onchain", b)]:
                    year, features = int(fold["year"]), fold["features"]
                    if year not in info["years"]:
                        continue
                    for name in args.models:
                        folder = args.out_dir / "fits" / f"{name}_{config}_h{h}_{year}"
                        folder.mkdir(parents=True)
                        train = pd.concat([frame(fold, "inner_train"), fold["X_inner_train"]], axis=1)
                        C = tune_lr(train, features, episodes, folder, h) if name == "logreg" else None
                        practice = fit(name, fold["X_inner_train"], fold["y_inner_train"], C)
                        val = frame(fold, "inner_val")
                        val["y_score"] = predict(practice, fold["X_inner_val"])
                        cutoff, validation = _choose_threshold(val, episodes, h, 1., policy="exhaustive_lowest")
                        joblib.dump(dict(model=practice[0], imputer=practice[1]), folder / "practice_pipeline.joblib")
                        # Full outer train is provided by the splitter, rather than
                        # concatenating inner partitions with their artificial gap.
                        final = fit(name, fold["X_train"], fold["y_train"], C)
                        joblib.dump(dict(model=final[0], imputer=final[1]), folder / "pipeline.joblib")
                        test = frame(fold, "test")
                        test["y_score"] = predict(final, fold["X_test"])
                        if not np.isfinite(test.y_score).all():
                            raise ValueError(f"Nonfinite scores: {folder.name}")
                        test["threshold"] = cutoff
                        val["threshold"] = cutoff
                        for part in ["inner_train", "inner_val", "train", "test"]:
                            persisted = frame(fold, part)
                            if part == "inner_val":
                                persisted = val
                            elif part == "test":
                                persisted = test
                            pd.concat([persisted, fold[f"X_{part}"]], axis=1).to_parquet(
                                folder / f"{part}.parquet", index=False)
                        _, alarms = build_alarm_events(test, episodes, cutoff)
                        alarms = alarms.assign(model=name, config=config, horizon_hours=h,
                                               policy="exhaustive_lowest", test_year=year)
                        alarms.to_csv(folder / "alarms.csv", index=False)
                        targets = _targetable_episode_ids(test, episodes, h)
                        result = metrics(test, alarms, targets)
                        row = dict(model=name, config=config, horizon_hours=h, test_year=year,
                                   policy="exhaustive_lowest", threshold=cutoff, selected_C=C,
                                   **result, coin_months=result["observed_coin_months"],
                                   test_rows=result["rows"], validation=validation)
                        rows.append(row)
                        events_all.append(alarms)
                        key = (name, config, h)
                        scores[key] = pd.concat([scores.get(key, pd.DataFrame()), test], ignore_index=True)
                        save_json(folder / "fit.json", dict(
                            run_id=run_id, model=name, config=config, horizon=h, year=year,
                            features=features, selected_C=C, practice_fit_partition="inner_train",
                            fit_partition="train", calibration_partition="inner_val",
                            refit_after_calibration=True, threshold_policy="exhaustive_lowest",
                            threshold=cutoff, validation=validation,
                            params=final[0].get_params(), practice_params=practice[0].get_params(),
                            final_train_rows=len(fold["y_train"]),
                            final_train_last_hour=str(fold["train_meta"].hour.max()),
                            practice_sha256=sha256(folder / "practice_pipeline.joblib"),
                            pipeline_sha256=sha256(folder / "pipeline.joblib")))
                        pd.DataFrame(rows).to_csv(args.out_dir / "by_year.csv", index=False)
                        print(f"{folder.name}: caught={result['episodes_caught']}/{len(targets)} "
                              f"FA={result['false_alarms']} cutoff={cutoff}", flush=True)
        yearly, summary, alarms = aggregate(rows, events_all, scores)
        positive = yearly.groupby(["model", "config", "horizon_hours", "policy"], as_index=False).agg(
            positive_hours=("positive_hours", "sum"))
        summary = summary.merge(positive, on=["model", "config", "horizon_hours", "policy"])
        summary["positive_rate"] = summary.positive_hours / summary.test_rows
        summary.to_csv(args.out_dir / "summary.csv", index=False)
        alarms.to_csv(args.out_dir / "alarm_events.csv", index=False)
        info.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as exc:
        info.update(status="failed", error=repr(exc))
        raise
    finally:
        save_json(args.out_dir / "run.json", info)


if __name__ == "__main__":
    main()
