"""Professor-style final refit for the corrected model matrix.

For test year Y the cutoff is the verified lowest-feasible cutoff that was chosen
on year Y-1 with the practice model fitted on years before Y-1 (saved-score
recalibration run). The final model is refit with the same procedure on the
`train` partition (all eligible hours before Jan 1 of Y minus the purge) and
scores Y with that cutoff. Cutoffs are copied, never reselected.

Persistence has no learned parameters, so its scores are unchanged. The constant
prior is carried over unrefit: with a fixed cutoff a refit constant would only
switch every alarm on or off.

python -m modelling.refit_final --xgb-source <run> --lr-rf-source <run> \
    --cutoff-source <lowest-feasible recalibration run> --out-dir <new dir>
Add --resume to continue an interrupted run in the same directory.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import traceback
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

from data_processing.config import PURGE_HOURS, EMBARGO_HOURS
from data_processing.splits import load_master
from modelling.alarm_events import build_alarm_events
from modelling.corrected_data import corrected_folds
from modelling.corrected_lr_rf import RF_PARAMS, fit_lr, metrics, tune_lr
from modelling.corrected_xgb import PARAMS as XGB_PARAMS, paired_uncertainty
from modelling.false_alarm_framework import _common_sample_fold
from modelling.models import get_model
from modelling.models.base import MedianImputer
from modelling.models.xgboost_model import XGBModel
from modelling.rolling_threshold import _targetable_episode_ids

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ("market_original", "market_matched", "market_onchain")
HORIZONS = (1, 6, 24)
YEARS = tuple(range(2021, 2026))
MODELS = ("logreg", "random_forest", "xgboost", "persistence", "baseline")
REFIT = ("logreg", "random_forest", "xgboost")
SOURCE_FILES = ("refit_final.py", "verify_refit_run.py", "corrected_lr_rf.py", "corrected_xgb.py",
                "corrected_data.py", "rolling_threshold.py", "alarm_events.py",
                "train_logreg_yearcv.py", "false_alarm_framework.py")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def save_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def _keys(frame):
    return set(zip(frame.coin, pd.to_datetime(frame.hour, utc=True)))


def check_refit_partition(train, inner_train, inner_val, test, year):
    """Final training set must hold practice train+validation and nothing from the test year."""
    boundary = pd.Timestamp(f"{int(year)}-01-01", tz="UTC")
    hours, test_hours = pd.to_datetime(train.hour, utc=True), pd.to_datetime(test.hour, utc=True)
    if hours.max() >= boundary - pd.Timedelta(hours=PURGE_HOURS):
        raise ValueError(f"{year}: final training rows reach the test purge window")
    if test_hours.min() < boundary + pd.Timedelta(hours=EMBARGO_HOURS):
        raise ValueError(f"{year}: test rows start inside the embargo")
    train_keys = _keys(train)
    if train_keys & _keys(test):
        raise ValueError(f"{year}: test rows found in final training set")
    if not _keys(inner_train) <= train_keys or not _keys(inner_val) <= train_keys:
        raise ValueError(f"{year}: final training set does not contain practice train and validation")
    if len(train_keys) != len(train):
        raise ValueError(f"{year}: duplicate coin-hours in final training set")
    return dict(train_rows=len(train), train_positive_hours=int(train.y_true.sum()),
                practice_train_rows=len(inner_train), validation_rows_added=len(inner_val),
                gap_rows_added=len(train) - len(inner_train) - len(inner_val),
                train_last_origin=str(hours.max()), test_first_origin=str(test_hours.min()))


def _frame(fold, part):
    return pd.concat([fold[f"{part}_meta"], fold[f"X_{part}"], fold[f"y_{part}"].rename("y_true")], axis=1)


def _check_sources(xgb_source, lrrf_source, cutoff_source):
    for source in (xgb_source, lrrf_source):
        info = json.loads((source / "run.json").read_text())
        verified = json.loads((source / "verification.json").read_text())
        if info["status"] != "complete" or verified["status"] != "passed":
            raise ValueError(f"Source is not complete and verified: {source}")
    info = json.loads((cutoff_source / "run.json").read_text())
    verified = json.loads((cutoff_source / "verification.json").read_text())
    if info["status"] != "complete" or verified["status"] != "passed":
        raise ValueError(f"Cutoff source is not complete and verified: {cutoff_source}")
    if info["threshold_policy"] != "lowest_feasible":
        raise ValueError("Cutoff source must use the lowest_feasible policy")
    used = {Path(r["path"]).resolve() for r in info["source_runs"]}
    if used != {xgb_source.resolve(), lrrf_source.resolve()}:
        raise ValueError("Cutoff source was not derived from these model sources")
    for record in info["source_runs"]:
        if digest(Path(record["path"]) / "run.json") != record["manifest_sha256"]:
            raise ValueError(f"Source manifest changed since cutoff selection: {record['path']}")


def _prepare(out, xgb_source, lrrf_source, scope):
    master, episodes = load_master()
    for source in (xgb_source, lrrf_source):
        pd.testing.assert_frame_equal(master, pd.read_parquet(source / "corrected_master.parquet"), check_exact=True)
        pd.testing.assert_frame_equal(episodes, pd.read_parquet(source / "episodes.parquet"), check_exact=True)
    if not (out / "episodes.parquet").exists():
        episodes.to_parquet(out / "episodes.parquet", index=False)
    cells = {}
    for h in sorted({h for _, h, _ in scope}):
        a, b = corrected_folds(master, episodes, h, 1), corrected_folds(master, episodes, h, 2)
        folds = dict(market_original=a, market_onchain=b,
                     market_matched=[_common_sample_fold(x, y) for x, y in zip(a, b)])
        for config, year_folds in folds.items():
            for fold in year_folds:
                year = int(fold["year"])
                if (config, h, year) not in scope:
                    continue
                cell = f"{config}_h{h}_{year}"
                d = out / "partitions" / cell
                cells[cell] = dict(config=config, horizon=h, year=year, features=fold["features"])
                if (d / "partition.json").exists():
                    continue
                parts = {p: _frame(fold, p) for p in ("train", "inner_train", "inner_val", "test")}
                for part in ("inner_train", "inner_val", "test"):
                    reference = pd.read_parquet(lrrf_source / "partitions" / cell / f"{part}.parquet")
                    pd.testing.assert_frame_equal(parts[part], reference[parts[part].columns], check_exact=True)
                xgb_test = pd.read_parquet(xgb_source / "fits" / cell / "test.parquet")
                columns = ["coin", "hour", "y_true", *fold["features"]]
                pd.testing.assert_frame_equal(parts["test"][columns], xgb_test[columns], check_exact=True)
                stats = check_refit_partition(parts["train"], parts["inner_train"], parts["inner_val"],
                                              parts["test"], year)
                d.mkdir(parents=True, exist_ok=True)
                parts["train"].to_parquet(d / "train.parquet", index=False)
                parts["test"].to_parquet(d / "test.parquet", index=False)
                save_json(d / "partition.json", dict(cells[cell], **stats,
                    train_sha256=digest(d / "train.parquet"), test_sha256=digest(d / "test.parquet"),
                    practice_partitions_match_source=True))
                print(f"partition {cell}: final train {stats['train_rows']} rows "
                      f"(practice {stats['practice_train_rows']} + validation {stats['validation_rows_added']} "
                      f"+ gap {stats['gap_rows_added']})", flush=True)
    return episodes, cells


def fit_cell(out, model, cell, spec, sources, run_id):
    """Fit one final model, apply the copied cutoff and save the cell atomically."""
    start = time.perf_counter()
    name = f"{model}_{cell}"
    final = out / "fits" / name
    if (final / "result.json").exists():
        return json.loads((final / "result.json").read_text())
    temporary = out / "fits" / f".tmp_{name}"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    config, h, year, features = spec["config"], spec["horizon"], spec["year"], spec["features"]
    episodes = pd.read_parquet(out / "episodes.parquet")
    partition = json.loads((out / "partitions" / cell / "partition.json").read_text())
    train = pd.read_parquet(out / "partitions" / cell / "train.parquet")
    test = pd.read_parquet(out / "partitions" / cell / "test.parquet")
    assert digest(out / "partitions" / cell / "train.parquet") == partition["train_sha256"]
    assert digest(out / "partitions" / cell / "test.parquet") == partition["test_sha256"]
    evaluation_path = sources["cutoff"] / "cells" / f"{model}_{cell}" / "evaluation.json"
    evaluation = json.loads(evaluation_path.read_text())
    if evaluation["threshold_policy"] != "lowest_feasible":
        raise ValueError(f"{name}: cutoff source is not lowest_feasible")
    threshold = float(evaluation["threshold"])
    practice_dir = (sources["xgb"] / "fits" / cell if model == "xgboost"
                    else sources["lrrf"] / "fits" / name)
    practice = pd.read_parquet(practice_dir / "test.parquet")
    pd.testing.assert_frame_equal(practice[["coin", "hour", "y_true"]], test[["coin", "hour", "y_true"]],
                                  check_exact=True)
    imputer, extra = None, {}
    X = train[features]
    if model == "xgboost":
        fitted = XGBModel(**XGB_PARAMS).fit(X, train.y_true)
        assert fitted.clf_.get_booster().num_boosted_rounds() == 300
        extra.update(params=XGB_PARAMS, scale_pos_weight_used=fitted.scale_pos_weight_used_)
    elif model == "logreg":
        with warnings.catch_warnings(), threadpool_limits(limits=1):
            C = tune_lr(train, features, episodes, temporary, h)
            fitted, imputer = fit_lr(X, train.y_true, C)
        extra.update(selected_C=C, n_iter=fitted.clf_.n_iter_.tolist())
    elif model in ("random_forest", "persistence"):
        fitted = get_model(model, **(RF_PARAMS if model == "random_forest" else {}))
        if fitted.needs_imputation:
            imputer = MedianImputer().fit(X)
            X = imputer.transform(X)
        fitted.fit(X, train.y_true)
    else:
        fitted = None
    if fitted is None:
        scores = practice.y_score.to_numpy()
        refit, note = False, "constant prior carried over from practice fit; not refit"
    else:
        Xt = test[features] if imputer is None else imputer.transform(test[features])
        scores = fitted.predict_proba(Xt)[:, 1]
        refit = model in REFIT
        note = ("refit on all pre-test-year data" if refit
                else "no learned parameters; scores identical to practice fit")
        if model == "persistence":
            np.testing.assert_array_equal(scores, practice.y_score.to_numpy())
        artifact = temporary / ("model.joblib" if model == "xgboost" else "pipeline.joblib")
        joblib.dump(fitted if model == "xgboost" else {"model": fitted, "imputer": imputer}, artifact)
        reloaded = joblib.load(artifact)
        Xr = test[features] if imputer is None else reloaded["imputer"].transform(test[features])
        again = (reloaded if model == "xgboost" else reloaded["model"]).predict_proba(Xr)[:, 1]
        np.testing.assert_allclose(again, scores, atol=1e-8, rtol=0)
    assert np.isfinite(scores).all()
    pred = test[["coin", "hour", "episode_id", "y_true"]
                + (["dataset_version"] if "dataset_version" in test else [])].copy()
    pred["y_score"], pred["threshold"] = scores, threshold
    pred["horizon"], pred["model"], pred["config"], pred["test_year"] = h, model, config, year
    pred["run_id"], pred["fit_id"] = run_id, name
    pd.concat([pred, test[features]], axis=1).to_parquet(temporary / "test.parquet", index=False)
    _, events = build_alarm_events(pred, episodes, threshold)
    targets = _targetable_episode_ids(pred, episodes, h)
    events["fit_id"], events["run_id"], events["model"] = name, run_id, model
    events.to_csv(temporary / "alarms.csv", index=False)
    importance = []
    if model == "logreg":
        importance = [dict(feature=f, value=float(v), type="standardized L1 coefficient")
                      for f, v in zip(fitted.cols_, fitted.clf_.coef_[0])]
    elif model == "random_forest":
        importance = [dict(feature=f, value=float(v), type="normalized mean decrease in impurity")
                      for f, v in zip(features, fitted.feature_importances_)]
    for r in importance:
        r.update(model=model, config=config, horizon_hours=h, test_year=year, fit_id=name,
                 run_id=run_id, is_onchain="logdiff" in r["feature"])
    if importance:
        pd.DataFrame(importance).to_csv(temporary / "importance.csv", index=False)
    if fitted is not None:
        extra["model_sha256"] = digest(artifact)
    result = dict(model=model, config=config, horizon_hours=h, test_year=year, run_id=run_id,
                  final_refit=refit, threshold=threshold, threshold_policy="lowest_feasible",
                  threshold_source="practice-model validation year (copied, not reselected)",
                  **metrics(pred, events, targets),
                  practice_average_precision=float(average_precision_score(practice.y_true, practice.y_score)),
                  validation_false_alarms=evaluation["validation_false_alarms"],
                  validation_coin_months=evaluation["validation_coin_months"],
                  train_rows=partition["train_rows"], train_positive_hours=partition["train_positive_hours"],
                  practice_train_rows=partition["practice_train_rows"],
                  selected_C=extra.get("selected_C", np.nan),
                  scale_pos_weight_used=extra.get("scale_pos_weight_used", np.nan),
                  elapsed_seconds=time.perf_counter() - start)
    save_json(temporary / "fit.json", dict(
        spec, fit_id=name, model=model, cell=cell, run_id=run_id, protocol=note,
        fit_partition="train (all eligible hours before test year minus purge)",
        practice_fit=str(practice_dir / "fit.json"), practice_fit_sha256=digest(practice_dir / "fit.json"),
        cutoff_evaluation=str(evaluation_path), cutoff_evaluation_sha256=digest(evaluation_path),
        threshold=threshold, threshold_policy="lowest_feasible", partition=partition, **extra))
    save_json(temporary / "result.json", result)
    temporary.replace(final)
    print(f"{name}: {result['elapsed_seconds']:.0f}s AP={result['average_precision']:.6f} "
          f"caught={result['episodes_caught']}/{result['targetable_episodes']} FA={result['false_alarms']}",
          flush=True)
    return result


def _aggregate(out, episodes, sources, run_id):
    rows = [json.loads((f / "result.json").read_text())
            for f in sorted((out / "fits").iterdir()) if not f.name.startswith(".")]
    yearly = pd.DataFrame(rows).sort_values(["model", "config", "horizon_hours", "test_year"])
    yearly.to_csv(out / "per_year.csv", index=False)
    pooled, identities, predictions, importance = [], [], {}, []
    for (model, config, h), group in yearly.groupby(["model", "config", "horizon_hours"]):
        folders = [out / "fits" / f"{model}_{config}_h{h}_{y}" for y in group.test_year]
        pred = pd.concat([pd.read_parquet(f / "test.parquet", columns=[
            "coin", "hour", "episode_id", "y_true", "y_score", "threshold", "horizon"]) for f in folders],
            ignore_index=True)
        events = pd.concat([pd.read_csv(f / "alarms.csv") for f in folders], ignore_index=True)
        events["is_true_alarm"] = events.is_true_alarm.astype(bool)
        targets = _targetable_episode_ids(pred, episodes, h)
        pooled.append(dict(model=model, config=config, horizon_hours=h, protocol="final_refit",
                           threshold_policy="lowest_feasible", **metrics(pred, events, targets)))
        predictions[model, config, h] = pred
        caught = set(events.target_episode_id.dropna().astype(str))
        for ep in sorted(targets):
            e = episodes.loc[episodes.episode_id.eq(ep)].iloc[0]
            identities.append(dict(model=model, config=config, horizon_hours=h, episode_id=ep,
                                   coin=e.coin, episode_start=e.start, caught=ep in caught,
                                   earliest_alarm_lead_hours=events.loc[
                                       events.target_episode_id.eq(ep), "lead_hours"].max()))
        importance.extend(pd.read_csv(f / "importance.csv") for f in folders if (f / "importance.csv").exists())
    pooled = pd.DataFrame(pooled)
    pooled.to_csv(out / "pooled.csv", index=False)
    identities = pd.DataFrame(identities)
    identities.to_csv(out / "caught_identities.csv", index=False)
    if importance:
        pd.concat(importance, ignore_index=True).to_csv(out / "importance_per_fit.csv", index=False)
    yearly.loc[yearly.model.eq("logreg"), ["config", "horizon_hours", "test_year", "selected_C"]].to_csv(
        out / "lr_selected_c.csv", index=False)
    uncertainty = []
    for model in REFIT:
        for h in sorted(yearly.horizon_hours.unique()):
            if (model, "market_matched", h) in predictions and (model, "market_onchain", h) in predictions:
                ids = identities.loc[identities.model.eq(model)]
                uncertainty.append(dict(model=model, **paired_uncertainty(
                    predictions[model, "market_matched", h], predictions[model, "market_onchain", h], ids, h)))
    pd.DataFrame(uncertainty).to_csv(out / "matched_uncertainty.csv", index=False)
    comparison = pooled[["model", "config", "horizon_hours", "average_precision", "episodes_caught",
                         "targetable_episodes", "false_alarms", "observed_coin_months"]].copy()
    no_refit = pd.read_csv(sources["cutoff"] / "primary_pooled.csv", float_precision="round_trip")
    max_recall = pd.read_csv(sources["xgb"].parent / "ALL-MODELS-POOLED.csv", float_precision="round_trip")
    keys = ["model", "config", "horizon_hours"]
    for label, frame in (("no_refit_lowest", no_refit), ("max_recall", max_recall)):
        comparison = comparison.merge(frame[keys + ["average_precision", "episodes_caught", "false_alarms"]]
                                      .rename(columns={c: f"{label}_{c}" for c in
                                                       ("average_precision", "episodes_caught", "false_alarms")}),
                                      on=keys, how="left", validate="one_to_one")
    comparison.to_csv(out / "protocol_comparison.csv", index=False)
    return yearly, pooled


def run(xgb_source, lrrf_source, cutoff_source, out, workers=6, configs=CONFIGS, horizons=HORIZONS,
        years=YEARS, models=MODELS, shap=True, resume=False):
    scope = {(c, int(h), int(y)) for c in configs for h in horizons for y in years}
    sources = dict(xgb=xgb_source, lrrf=lrrf_source, cutoff=cutoff_source)
    requested = dict(configs=sorted(configs), horizons=sorted(int(h) for h in horizons),
                     test_years=sorted(int(y) for y in years), models=sorted(models),
                     sources={k: str(v) for k, v in sources.items()})
    if out.exists():
        if not resume:
            raise FileExistsError(f"Use a new output directory or --resume: {out}")
        manifest = json.loads((out / "run.json").read_text())
        if manifest["status"] == "complete" or manifest["scope"] != requested:
            raise ValueError("Resume needs an unfinished run with the same scope and sources")
    else:
        if resume:
            raise FileNotFoundError(out)
        out.mkdir(parents=True)
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-refit"
        manifest = dict(run_id=run_id, status="running", protocol="final_refit", scope=requested,
                        threshold_policy="lowest_feasible",
                        budget="practice-model validation: total grouped FA / observed eligible coin-months <= 1",
                        horizons=requested["horizons"], test_years=requested["test_years"],
                        shap_requested=bool(shap and "xgboost" in models),
                        purge_hours=PURGE_HOURS, embargo_hours=EMBARGO_HOURS,
                        source_commit=subprocess.check_output(
                            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
                        source_code_sha256={n: digest(ROOT / "modelling" / n) for n in SOURCE_FILES},
                        limitations=[
                            "Cutoff chosen on practice-model scores and applied to a different final model.",
                            "Constant prior is not refit; persistence has no parameters.",
                            "Retrospective protocol change after inspecting earlier test outcomes.",
                            "Validation false-alarm budget is not a test-period guarantee."])
        snapshot = out / "source"
        snapshot.mkdir()
        for n in SOURCE_FILES:
            shutil.copy2(ROOT / "modelling" / n, snapshot / n)
        (out / "source_diff.patch").write_bytes(
            subprocess.check_output(["git", "-C", str(ROOT), "diff", "--binary"]))
    run_id = manifest["run_id"]
    manifest.update(status="running", resumed_at=str(datetime.now(timezone.utc)) if resume else None)
    save_json(out / "run.json", manifest)
    try:
        _check_sources(xgb_source, lrrf_source, cutoff_source)
        episodes, cells = _prepare(out, xgb_source, lrrf_source, scope)
        tasks = [(m, c) for m in MODELS if m in models for c in sorted(cells)]
        failures = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(fit_cell, out, m, c, cells[c], sources, run_id): (m, c) for m, c in tasks}
            for done, future in enumerate(as_completed(futures), 1):
                m, c = futures[future]
                try:
                    future.result()
                except BrokenProcessPool:
                    raise
                except Exception:
                    failures.append(f"{m}_{c}")
                    print(f"FAILED {m}_{c}\n{traceback.format_exc()}", flush=True)
                manifest.update(completed_cells=done - len(failures), failed_cells=failures,
                                updated_at=str(datetime.now(timezone.utc)))
                save_json(out / "run.json", manifest)
        if failures:
            raise RuntimeError(f"{len(failures)} cells failed: {failures}")
        yearly, pooled = _aggregate(out, episodes, sources, run_id)
        if manifest["shap_requested"]:
            from modelling.corrected_xgb import run_shap
            run_shap(out, fit_prefix="xgboost_")
        manifest.update(status="complete", cells=len(yearly), pooled_rows=len(pooled),
                        finished_at=str(datetime.now(timezone.utc)))
        save_json(out / "run.json", manifest)
    except BaseException as error:
        manifest.update(status="incomplete", error=f"{type(error).__name__}: {error}")
        save_json(out / "run.json", manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--xgb-source", type=Path, required=True)
    parser.add_argument("--lr-rf-source", type=Path, required=True)
    parser.add_argument("--cutoff-source", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--config", action="append", choices=CONFIGS, dest="configs")
    parser.add_argument("--horizon", action="append", type=int, choices=HORIZONS, dest="horizons")
    parser.add_argument("--years", type=int, nargs="+", choices=YEARS)
    parser.add_argument("--model", action="append", choices=MODELS, dest="models")
    parser.add_argument("--no-shap", action="store_true")
    parser.add_argument("--resume", action="store_true")
    a = parser.parse_args()
    run(a.xgb_source.resolve(), a.lr_rf_source.resolve(), a.cutoff_source.resolve(), a.out_dir.resolve(),
        workers=a.workers, configs=a.configs or CONFIGS, horizons=a.horizons or HORIZONS,
        years=a.years or YEARS, models=a.models or MODELS, shap=not a.no_shap, resume=a.resume)


if __name__ == "__main__":
    main()
