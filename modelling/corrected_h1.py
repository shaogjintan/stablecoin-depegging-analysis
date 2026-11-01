"""Separate h1 LR/RF/baseline run, using the unchanged corrected protocol."""
import argparse
import json
from pathlib import Path
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from modelling.corrected_data import corrected_folds
from modelling.corrected_lr_rf import CONFIGS, MODELS, PARTS, ROOT, score_cell, metrics
from modelling.corrected_lr_rf import run as run_portable
from modelling.corrected_xgb import sha256, save_json, paired_uncertainty
from modelling.false_alarm_framework import _common_sample_fold
from modelling.rolling_threshold import _targetable_episode_ids
from modelling.verify_lr_rf_run import (
    reference_events, reference_metrics, assert_values, check_preprocessing, verify_cv,
)


def run(out, parent=None):
    if parent is None:
        run_portable(out, horizons=(1,))
        from modelling.verify_lr_rf_run import verify
        verify(out)
        return
    info = json.loads((parent / "run.json").read_text())
    assert info["status"] == "complete", "Wait for the h6/h24 run's independent verification first."
    assert json.loads((parent / "verification.json").read_text())["status"] == "passed"
    frozen = Path(info["frozen_path"])
    for rel, digest in info["source_code_sha256"].items():
        if rel != "modelling\\verify_lr_rf_run.py":
            assert sha256(ROOT / rel) == digest, f"Scoring source changed: {rel}"
    assert sha256(ROOT / "data" / "processed" / "master.parquet") == info["source_master_sha256"]
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    run_id = info["run_id"] + "-h1"
    manifest = dict(run_id=run_id, status="running", parent_run_id=info["run_id"],
                    parent_run=str(parent), horizons=[1], models=list(MODELS),
                    source_master_sha256=info["source_master_sha256"],
                    source_code_sha256=info["source_code_sha256"], thread_limits=4,
                    packages=info["packages"], limitations=info["limitations"],
                    h1_source_sha256=sha256(Path(__file__)),
                    comparison="h1 is not frozen-V3-XGBoost-comparable; V3 contains only h6/h24")
    save_json(out / "run.json", manifest)
    master = pd.read_parquet(frozen / "corrected_master.parquet")
    episodes = pd.read_parquet(frozen / "episodes.parquet")
    episodes.to_parquet(out / "episodes.parquet", index=False)
    a, b = corrected_folds(master, episodes, 1, 1), corrected_folds(master, episodes, 1, 2)
    configs = dict(market_original=a, market_onchain=b,
                   market_matched=[_common_sample_fold(x, y) for x, y in zip(a, b)])
    rows, pooled_rows, identities, importance, pooled, checked, cv_count = [], [], [], [], {}, 0, 0
    with warnings.catch_warnings(record=True) as recorded, threadpool_limits(limits=4):
        warnings.simplefilter("default")
        for config in CONFIGS:
            for fold in configs[config]:
                cell = f"{config}_h1_{fold['year']}"
                spec = dict(config=config, horizon=1, year=int(fold["year"]), features=fold["features"])
                d = out / "partitions" / cell
                d.mkdir(parents=True)
                for part in PARTS:
                    p = pd.concat([fold[f"{part}_meta"], fold[f"X_{part}"],
                                   fold[f"y_{part}"].rename("y_true")], axis=1)
                    p.to_parquet(d / f"{part}.parquet", index=False)
                for name in MODELS:
                    row = score_cell(out, cell, spec, name, episodes, run_id)
                    rows.append(row)
                    pd.DataFrame(rows).to_csv(out / "per_year.csv", index=False)
                    folder = out / "fits" / f"{name}_{cell}"
                    fit = json.loads((folder / "fit.json").read_text())
                    bundle = joblib.load(folder / "pipeline.joblib")
                    train = pd.read_parquet(d / "inner_train.parquet")
                    check_preprocessing(bundle, train, spec["features"])
                    if name == "logreg":
                        cv_count += verify_cv(folder, train, fit)
                    for part in ("inner_val", "test"):
                        p = pd.read_parquet(folder / f"{part}.parquet")
                        source = pd.read_parquet(d / f"{part}.parquet")
                        pd.testing.assert_frame_equal(p[["coin", "hour", "y_true"]],
                                                      source[["coin", "hour", "y_true"]])
                        X = source[spec["features"]]
                        if bundle["imputer"] is not None:
                            X = bundle["imputer"].transform(X)
                        np.testing.assert_allclose(bundle["model"].predict_proba(X)[:, 1],
                                                   p.y_score, atol=1e-8, rtol=0)
                        if part == "test":
                            ev = reference_events(p, episodes, fit["threshold"], 1)
                            assert_values(row, reference_metrics(p, ev, episodes, 1))
                    checked += 1
                    print(f"Independent h1 check {checked}/60 passed", flush=True)
        for name in MODELS:
            for config in CONFIGS:
                folders = [out / "fits" / f"{name}_{config}_h1_{y}" for y in range(2021, 2026)]
                p = pd.concat([pd.read_parquet(f / "test.parquet") for f in folders], ignore_index=True)
                ev = pd.concat([pd.read_csv(f / "alarms.csv") for f in folders], ignore_index=True)
                ev["is_true_alarm"] = ev.is_true_alarm.astype(bool)
                targets = _targetable_episode_ids(p, episodes, 1)
                row = dict(model=name, config=config, horizon_hours=1, run_id=run_id,
                           **metrics(p, ev, targets))
                independent = pd.concat([reference_events(g, episodes, g.threshold.iloc[0], 1)
                                         for _, g in p.groupby("test_year")], ignore_index=True)
                assert_values(row, reference_metrics(p, independent, episodes, 1))
                pooled_rows.append(row)
                p.to_parquet(out / f"predictions_{name}_{config}_h1.parquet", index=False)
                ev.to_csv(out / f"alarms_{name}_{config}_h1.csv", index=False)
                pooled[name, config] = p
                caught = set(ev.target_episode_id.dropna()) & targets
                for ep in sorted(targets):
                    e = episodes.loc[episodes.episode_id.eq(ep)].iloc[0]
                    identities.append(dict(model=name, config=config, horizon_hours=1,
                                           episode_id=ep, episode_start=e.start, caught=ep in caught))
                if name in MODELS[:2]:
                    importance.extend(pd.read_csv(f / "importance.csv") for f in folders)
        ids = pd.DataFrame(identities)
        ids.to_csv(out / "caught_identities.csv", index=False)
        uncertainty = []
        for name in MODELS[:2]:
            uncertainty.append(dict(model=name, run_id=run_id, **paired_uncertainty(
                pooled[name, "market_matched"], pooled[name, "market_onchain"],
                ids.loc[ids.model.eq(name)], 1)))
        pd.DataFrame(uncertainty).to_csv(out / "matched_uncertainty.csv", index=False)
    assert checked == 60
    pd.DataFrame(pooled_rows).to_csv(out / "summary.csv", index=False)
    pd.concat(importance, ignore_index=True).to_csv(out / "importance_per_fit.csv", index=False)
    manifest.update(status="complete", elapsed_seconds=time.perf_counter() - started,
                    warnings=[dict(category=w.category.__name__, message=str(w.message)) for w in recorded])
    save_json(out / "run.json", manifest)
    save_json(out / "verification.json", dict(status="passed", fits_checked=checked,
                                             cv_candidate_fits_checked=cv_count, independent_metrics=True))
    print(pd.DataFrame(pooled_rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-run", type=Path,
                        help="historical exact-reference mode; omit for a portable new run")
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    run(args.out_dir.resolve(), args.parent_run.resolve() if args.parent_run else None)
