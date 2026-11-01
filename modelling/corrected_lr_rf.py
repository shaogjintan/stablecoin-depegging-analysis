"""Comparable LR/RF and baseline runs; never writes to frozen XGB evidence."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

from modelling.alarm_events import build_alarm_events
from modelling.corrected_data import prepare_corrected, corrected_folds
from modelling.corrected_xgb import sha256, save_json, paired_uncertainty
from modelling.false_alarm_framework import _common_sample_fold, RANDOM_FOREST_FIXED_PARAMS
from modelling.models import get_model
from modelling.models.base import MedianImputer
from modelling.rolling_threshold import _choose_threshold, _targetable_episode_ids, _false_alarm_stats
from modelling.train_logreg_yearcv import DEFAULT_C_GRID, _calendar_cv_folds, _fit_model

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ("market_original", "market_matched", "market_onchain")
MODELS = ("logreg", "random_forest", "baseline", "persistence")
PARTS = ("inner_train", "inner_val", "test")
RF_PARAMS = {**RANDOM_FOREST_FIXED_PARAMS, "n_jobs": 4}


def fit_lr(X, y, C):
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        return _fit_model(X, y, C)


def select_c(candidates):
    return max(candidates, key=lambda r: (r["mean_average_precision"], -r["C"]))["C"]


def tune_lr(train, features, episodes, folder, h):
    X, y = train[features], train.y_true
    folds, skipped = _calendar_cv_folds(X, y, train, episodes)
    save_json(folder / "cv_skipped.json", skipped)
    for row in skipped:
        print(f"{folder.name}: CV {row['validation_year']} skipped: {row['reason']}", flush=True)
    if not folds:
        raise RuntimeError(f"{folder.name}: no usable inner-training CV folds; no fallback C")
    candidates = []
    for f in folds:
        tr, va = f["train_idx"], f["val_idx"]
        a, b = train.iloc[tr], train.iloc[va]
        assert a.hour.max() + pd.Timedelta(hours=h + 1) < b.hour.min()
        d = folder / "cv" / str(f["validation_year"])
        d.mkdir(parents=True)
        a.to_parquet(d / "train.parquet", index=False)
        b.to_parquet(d / "validation.parquet", index=False)
        save_json(d / "split.json", {k: v for k, v in f.items() if not k.endswith("_idx")})
    for C in DEFAULT_C_GRID:
        per_year = []
        for f in folds:
            tr, va = f["train_idx"], f["val_idx"]
            model, imputer = fit_lr(X.iloc[tr], y.iloc[tr], C)
            scores = model.predict_proba(imputer.transform(X.iloc[va]))[:, 1]
            d = folder / "cv" / str(f["validation_year"])
            joblib.dump({"model": model, "imputer": imputer}, d / f"C{C}.joblib")
            pred = train.iloc[va][["coin", "hour", "y_true"]].copy()
            pred["y_score"] = scores
            pred.to_parquet(d / f"C{C}.parquet", index=False)
            per_year.append(dict(validation_year=f["validation_year"],
                                 average_precision=float(average_precision_score(y.iloc[va], scores)),
                                 n_iter=model.clf_.n_iter_.tolist()))
        candidates.append(dict(C=C, mean_average_precision=float(np.mean(
            [r["average_precision"] for r in per_year])), per_year=per_year))
    selected = select_c(candidates)
    tuning = dict(selected_C=selected, grid=list(DEFAULT_C_GRID), penalty="l1",
                  selection="unweighted mean AP across usable inner-training calendar years; ties smaller C",
                  candidates=candidates, skipped=skipped,
                  calibration_used_for_selection=False)
    save_json(folder / "tuning.json", tuning)
    return selected


def prepare_partitions(out, frozen):
    original = ROOT / "data" / "processed" / "master.parquet"
    master, episodes = prepare_corrected(pd.read_parquet(original))
    pd.testing.assert_frame_equal(master, pd.read_parquet(frozen / "corrected_master.parquet"), check_exact=True)
    episodes = episodes.reset_index(drop=True)
    pd.testing.assert_frame_equal(episodes, pd.read_parquet(frozen / "episodes.parquet"), check_exact=True)
    episodes.to_parquet(out / "episodes.parquet", index=False)
    cells = {}
    for h in (6, 24):
        a, b = corrected_folds(master, episodes, h, 1), corrected_folds(master, episodes, h, 2)
        configurations = dict(market_original=a, market_onchain=b,
                              market_matched=[_common_sample_fold(x, y) for x, y in zip(a, b)])
        for config in CONFIGS:
            for fold in configurations[config]:
                year = int(fold["year"])
                cell = f"{config}_h{h}_{year}"
                d = out / "partitions" / cell
                d.mkdir(parents=True)
                features = fold["features"]
                for part in PARTS:
                    p = pd.concat([fold[f"{part}_meta"], fold[f"X_{part}"],
                                   fold[f"y_{part}"].rename("y_true")], axis=1)
                    reference = pd.read_parquet(frozen / "fits" / cell / f"{part}.parquet")
                    pd.testing.assert_frame_equal(p, reference[p.columns], check_exact=True)
                    p.to_parquet(d / f"{part}.parquet", index=False)
                cells[cell] = dict(config=config, horizon=h, year=year, features=features)
    save_json(out / "partitions.json", cells)
    return episodes, cells


def metrics(pred, events, targets):
    caught = set(events.target_episode_id.dropna().astype(str)) & targets
    fa, months, rate, maximum = _false_alarm_stats(events, pred)
    return dict(rows=len(pred), positive_hours=int(pred.y_true.sum()),
                positive_rate=float(pred.y_true.mean()),
                average_precision=float(average_precision_score(pred.y_true, pred.y_score)),
                targetable_episodes=len(targets), episodes_caught=len(caught),
                episode_recall=len(caught) / len(targets) if targets else np.nan,
                raw_flagged_hours=int(pred.y_score.ge(pred.threshold).sum()),
                alarm_events=len(events), true_alarms=int(events.is_true_alarm.sum()),
                false_alarms=fa, observed_coin_months=months,
                false_alarms_per_coin_month=rate, max_false_alarms_in_coin_month=maximum,
                median_lead_hours=float(events.loc[events.is_true_alarm, "lead_hours"].median()))


def score_cell(out, cell, spec, name, episodes, run_id):
    start = time.perf_counter()
    folder = out / "fits" / f"{name}_{cell}"
    folder.mkdir(parents=True, exist_ok=False)
    features, h = spec["features"], spec["horizon"]
    parts = {p: pd.read_parquet(out / "partitions" / cell / f"{p}.parquet") for p in PARTS}
    train = parts["inner_train"]
    assert set(train.y_true) == {0, 1}
    imputer = None
    if name == "logreg":
        C = tune_lr(train, features, episodes, folder, h)
        model, imputer = fit_lr(train[features], train.y_true, C)
    else:
        model = get_model(name, **(RF_PARAMS if name == "random_forest" else {}))
        X = train[features]
        if model.needs_imputation:
            imputer = MedianImputer().fit(X)
            X = imputer.transform(X)
        model.fit(X, train.y_true)
    joblib.dump({"model": model, "imputer": imputer}, folder / "pipeline.joblib")
    frames = {}
    for part in ("inner_val", "test"):
        X = parts[part][features]
        if imputer is not None:
            X = imputer.transform(X)
        pred = parts[part][["coin", "hour", "episode_id", "y_true"]].copy()
        pred["y_score"] = model.predict_proba(X)[:, 1]
        assert np.isfinite(pred.y_score).all()
        pred["horizon"], pred["model"], pred["config"] = h, name, spec["config"]
        pred["run_id"], pred["fit_id"], pred["test_year"] = run_id, folder.name, spec["year"]
        frames[part] = pred
    threshold, val_stats = _choose_threshold(frames["inner_val"], episodes, h, 1.)
    for part, pred in frames.items():
        pred["threshold"] = threshold
        pred.to_parquet(folder / f"{part}.parquet", index=False)
    pred = frames["test"]
    _, events = build_alarm_events(pred, episodes, threshold)
    targets = _targetable_episode_ids(pred, episodes, h)
    events["fit_id"], events["run_id"], events["model"] = folder.name, run_id, name
    events.to_csv(folder / "alarms.csv", index=False)
    preprocess = {"median": imputer.medians_.to_dict()} if imputer is not None else {"type": "none"}
    importance = []
    if name == "logreg":
        preprocess.update(low=model.lo_.to_dict(), high=model.hi_.to_dict(),
                          scaler_mean=model.scaler_.mean_.tolist(), scaler_scale=model.scaler_.scale_.tolist(),
                          feature_order=model.cols_, n_iter=model.clf_.n_iter_.tolist(),
                          intercept=model.clf_.intercept_.tolist())
        importance = [dict(feature=f, value=float(v), type="standardized L1 coefficient")
                      for f, v in zip(model.cols_, model.clf_.coef_[0])]
    elif name == "random_forest":
        importance = [dict(feature=f, value=float(v), type="normalized mean decrease in impurity")
                      for f, v in zip(features, model.feature_importances_)]
    for r in importance:
        r.update(model=name, config=spec["config"], horizon_hours=h, test_year=spec["year"],
                 fit_id=folder.name, run_id=run_id, is_onchain="logdiff" in r["feature"])
    pd.DataFrame(importance, columns=["feature", "value", "type", "model", "config", "horizon_hours",
                                     "test_year", "fit_id", "run_id", "is_onchain"]).to_csv(
                                         folder / "importance.csv", index=False)
    save_json(folder / "fit.json", dict(
        **spec, model=name, fit_id=folder.name, run_id=run_id, cell=cell,
        fit_partition="inner_train", calibration_partition="inner_val", refit_after_calibration=False,
        params=model.get_params(), estimator_params=model.clf_.get_params() if name in MODELS[:2] else {},
        preprocessing=preprocess, pipeline_sha256=sha256(folder / "pipeline.joblib"),
        threshold=threshold, validation=val_stats, elapsed_seconds=time.perf_counter() - start,
        partition_paths={p: str(Path("partitions") / cell / f"{p}.parquet") for p in PARTS}))
    row = dict(model=name, config=spec["config"], horizon_hours=h, test_year=spec["year"],
               run_id=run_id, threshold=threshold, **metrics(pred, events, targets),
               elapsed_seconds=time.perf_counter() - start,
               validation_average_precision=float(average_precision_score(
                   frames["inner_val"].y_true, frames["inner_val"].y_score)),
               validation_false_alarms=val_stats["false_alarms"],
               validation_coin_months=val_stats["coin_months"],
               validation_episode_recall=val_stats["episode_recall"],
               selected_C=model.C if name == "logreg" else np.nan)
    print(f"{folder.name}: {row['elapsed_seconds']:.2f}s AP={row['average_precision']:.8f} "
          f"caught={row['episodes_caught']}/{row['targetable_episodes']} FA={row['false_alarms']}", flush=True)
    return row


def aggregate(out, episodes, run_id):
    years = pd.read_csv(out / "per_year.csv")
    rows, identities, importance, pooled = [], [], [], {}
    for name in MODELS:
        for config in CONFIGS:
            for h in (6, 24):
                folders = [out / "fits" / f"{name}_{config}_h{h}_{y}" for y in range(2021, 2026)]
                pred = pd.concat([pd.read_parquet(f / "test.parquet") for f in folders], ignore_index=True)
                events = pd.concat([pd.read_csv(f / "alarms.csv") for f in folders], ignore_index=True)
                events["is_true_alarm"] = events.is_true_alarm.astype(bool)
                targets = _targetable_episode_ids(pred, episodes, h)
                rows.append(dict(model=name, config=config, horizon_hours=h, run_id=run_id,
                                 **metrics(pred, events, targets)))
                pred.to_parquet(out / f"predictions_{name}_{config}_h{h}.parquet", index=False)
                events.to_csv(out / f"alarms_{name}_{config}_h{h}.csv", index=False)
                caught = set(events.target_episode_id.dropna().astype(str)) & targets
                for ep in sorted(targets):
                    e = episodes.loc[episodes.episode_id.eq(ep)].iloc[0]
                    identities.append(dict(model=name, config=config, horizon_hours=h,
                                           run_id=run_id, episode_id=ep, coin=e.coin, episode_start=e.start,
                                           caught=ep in caught, earliest_alarm_lead_hours=events.loc[
                                               events.target_episode_id.eq(ep), "lead_hours"].max()))
                pooled[name, config, h] = pred
                if name in MODELS[:2]:
                    importance.extend(pd.read_csv(f / "importance.csv") for f in folders)
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary.csv", index=False)
    ids = pd.DataFrame(identities)
    ids.to_csv(out / "caught_identities.csv", index=False)
    uncertainty = []
    for name in MODELS[:2]:
        for h in (6, 24):
            r = paired_uncertainty(pooled[name, "market_matched", h], pooled[name, "market_onchain", h],
                                   ids[ids.model.eq(name)], h)
            uncertainty.append(dict(model=name, run_id=run_id, **r))
    pd.DataFrame(uncertainty).to_csv(out / "matched_uncertainty.csv", index=False)
    pd.concat(importance, ignore_index=True).to_csv(out / "importance_per_fit.csv", index=False)
    years[years.model.eq("logreg")].to_csv(out / "lr_selected_c.csv", index=False)


def run(out, frozen, benchmark=False):
    out.mkdir(parents=True, exist_ok=False)
    frozen_info = json.loads((frozen / "run.json").read_text())
    assert frozen_info["status"] == "complete"
    assert sha256(ROOT / "data" / "processed" / "master.parquet") == frozen_info["source_master_sha256"]
    for rel, expected in frozen_info["source_code_sha256"].items():
        assert sha256(ROOT / rel) == expected, f"Frozen protocol source differs: {rel}"
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    sources = [ROOT / "modelling" / n for n in (
        "corrected_lr_rf.py", "verify_lr_rf_run.py", "corrected_data.py", "corrected_xgb.py",
        "exact_threshold.py", "rolling_threshold.py", "alarm_events.py", "train_logreg_yearcv.py",
        "false_alarm_framework.py", "prepare_data.py", "models\\base.py", "models\\logreg.py",
        "models\\random_forest_model.py", "models\\baseline.py")]
    sources += [ROOT / "data_processing" / n for n in ("splits.py", "config.py", "merge_and_label.py")]
    manifest = dict(run_id=run_id, status="running", benchmark=benchmark, frozen_run_id=frozen_info["run_id"],
                    frozen_path=str(frozen), command=[sys.executable, *sys.argv],
                    source_commit=subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                                          text=True).strip(),
                    source_master_sha256=frozen_info["source_master_sha256"],
                    thread_limits=4, rf_params=RF_PARAMS, lr_grid=list(DEFAULT_C_GRID),
                    packages={p: importlib.metadata.version(p) for p in (
                        "numpy", "pandas", "scikit-learn", "joblib", "numba", "threadpoolctl")},
                    limitations=frozen_info["limitations"][:-1] + [
                        "No h1, sensitivity, smoothing, ensembles, upstream network rebuild or live validation.",
                        "LR coefficients and RF impurity importance are scoring-fit methodology, not causal benefit.",
                        "Fold-specific constant training priors can have pooled AP different from pooled prevalence."],
                    source_code_sha256={str(p.relative_to(ROOT)): sha256(p) for p in sources})
    save_json(out / "run.json", manifest)
    for p in sources:
        target = out / "source" / p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
    (out / "source_diff.patch").write_bytes(subprocess.check_output(["git", "-C", str(ROOT), "diff", "--binary"]))
    frozen_hashes = {str(p.relative_to(frozen)): sha256(p) for p in frozen.rglob("*") if p.is_file()}
    save_json(out / "frozen_evidence_sha256.json", frozen_hashes)
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as recorded, threadpool_limits(limits=4):
        warnings.simplefilter("default")
        episodes, cells = prepare_partitions(out, frozen)
        rows = []
        for cell, spec in cells.items():
            if benchmark and cell != "market_original_h6_2021":
                continue
            for name in (MODELS[:2] if benchmark else MODELS):
                rows.append(score_cell(out, cell, spec, name, episodes, run_id))
                pd.DataFrame(rows).to_csv(out / "per_year.csv", index=False)
        if not benchmark:
            aggregate(out, episodes, run_id)
    manifest["warnings"] = [dict(category=w.category.__name__, message=str(w.message),
                                 filename=w.filename, lineno=w.lineno) for w in recorded]
    assert all(sha256(frozen / rel) == digest for rel, digest in frozen_hashes.items())
    manifest.update(status="benchmark_complete" if benchmark else "scoring_complete",
                    elapsed_seconds=time.perf_counter() - started, fits=len(rows))
    save_json(out / "run.json", manifest)
    print(f"Saved {len(rows)} fits; {len(recorded)} recorded warnings; {manifest['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--frozen-xgb", required=True, type=Path)
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    out = args.out_dir.resolve()
    existed = out.exists()
    try:
        run(out, args.frozen_xgb.resolve(), args.benchmark)
    finally:
        if not existed and (out / "run.json").exists():
            state = json.loads((out / "run.json").read_text())
            if state["status"] == "running":
                state.update(status="incomplete", failure="Execution did not reach scoring completion; see command stderr log.")
                save_json(out / "run.json", state)
