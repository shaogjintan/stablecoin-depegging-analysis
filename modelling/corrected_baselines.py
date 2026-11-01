"""Run corrected baselines without touching legacy outputs or other worktrees."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import uuid

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from modelling.alarm_events import build_alarm_events
from modelling.corrected_data import corrected_folds
from data_processing.config import MODELLING_DATA_VERSION, PURGE_HOURS, EMBARGO_HOURS
from data_processing.splits import load_master, active_input_paths
from modelling.corrected_xgb import sha256, save_json
from modelling.false_alarm_framework import _common_sample_fold
from modelling.models.baseline import ConstantPriorModel, PersistenceModel
from modelling.rolling_threshold import _choose_threshold, _false_alarm_stats, _targetable_episode_ids

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_FILES = (
    "modelling\\corrected_data.py", "modelling\\rolling_threshold.py",
    "modelling\\exact_threshold.py", "modelling\\alarm_events.py",
    "modelling\\prepare_data.py", "data_processing\\splits.py",
    "data_processing\\merge_and_label.py", "data_processing\\config.py",
)


def metrics(pred, events, episodes, h):
    targets = _targetable_episode_ids(pred, episodes, h)
    caught = set(events.target_episode_id.dropna().astype(str)) & targets
    false, months, rate, maximum = _false_alarm_stats(events, pred)
    return dict(
        rows=len(pred), positive_hours=int(pred.y_true.sum()),
        positive_rate=float(pred.y_true.mean()),
        average_precision=float(average_precision_score(pred.y_true, pred.y_score)),
        raw_flagged_hours=int(pred.y_score.ge(pred.threshold).sum()),
        targetable_episodes=len(targets), episodes_caught=len(caught),
        episode_recall=len(caught) / len(targets) if targets else np.nan,
        alarm_events=len(events), true_alarms=int(events.is_true_alarm.sum()),
        false_alarms=false, observed_coin_months=months,
        false_alarms_per_coin_month=rate, max_false_alarms_in_coin_month=maximum,
        median_lead_hours=float(events.loc[events.is_true_alarm, "lead_hours"].median()),
    )


def _run(out, horizons=(1, 6, 24), years=range(2021, 2026)):
    if (not horizons or not set(horizons).issubset({1, 6, 24})
            or not years or not set(years).issubset(range(2021, 2026))):
        raise ValueError("Invalid horizon or year scope")
    source, episode_source = active_input_paths()
    master, episodes = load_master()
    out.mkdir(parents=True, exist_ok=False)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    paths = list(PROTOCOL_FILES) + [
        "modelling\\models\\baseline.py", "modelling\\false_alarm_framework.py",
        "modelling\\corrected_baselines.py", "modelling\\verify_baselines.py",
        "modelling\\checks\\test_corrected_baselines.py",
    ]
    info = dict(
        run_id=run_id, status="running",
        source_commit=subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        source_master_sha256=sha256(source),
        source_episodes_sha256=sha256(episode_source), dataset_version=MODELLING_DATA_VERSION,
        purge_hours=PURGE_HOURS, embargo_hours=EMBARGO_HOURS,
        source_code_sha256={rel: sha256(ROOT / rel) for rel in paths},
        packages={p: importlib.metadata.version(p) for p in
                  ("numpy", "pandas", "scikit-learn", "numba", "pyarrow")},
        horizons=list(horizons), test_years=list(years),
        fit_partition="inner_train", calibration_partition="inner_val",
        prior="mean of config/horizon/year inner_train labels only; no refit",
        persistence="min(max(1-price,0),0.01), price at completed source bar",
        clock="origin and onset = source bar start + 1 hour",
        threshold="all distinct validation scores and +inf; max targetable recall at FA/observed coin-month <=1; ties fewer FA then higher cutoff",
        limitations=[
            "Conditional corrected backtest, not upstream source certification.",
            "Unknown exchange/on-chain publication delays; idealized completed-bar availability.",
            "Outcome availability is an offline evaluation filter, not a live predictor.",
            "Observed coin-months count any eligible row, including partial months.",
            "Persistence has no on-chain inputs; matched eligibility is not on-chain information gain.",
            "Validation budget is not a guaranteed test budget; pooled prior AP is not generally base rate.",
        ],
    )
    save_json(out / "run.json", info)
    for rel in paths:
        target = out / "source" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, target)
    (out / "source_diff.patch").write_bytes(
        subprocess.check_output(["git", "-C", str(ROOT), "diff", "--binary"]))
    print(f"RUN {run_id}: using committed corrected master", flush=True)
    master.to_parquet(out / "corrected_master.parquet", index=False)
    episodes.to_parquet(out / "episodes.parquet", index=False)
    episodes.to_csv(out / "episodes.csv", index=False)
    info["corrected_master_sha256"] = sha256(out / "corrected_master.parquet")
    info["episodes_sha256"] = sha256(out / "episodes.parquet")
    save_json(out / "run.json", info)
    rows, summaries, identities, candidates = [], [], [], []
    for h in horizons:
        f1, f2 = corrected_folds(master, episodes, h, 1), corrected_folds(master, episodes, h, 2)
        configs = {
            "market_original": f1,
            "market_matched": [_common_sample_fold(a, b) for a, b in zip(f1, f2)],
            "market_onchain": f2,
        }
        for config, folds in configs.items():
            pooled = {name: [[], []] for name in ("constant_prior", "persistence")}
            for fold in folds:
                year = int(fold["year"])
                if year not in years:
                    continue
                sample = out / "samples" / f"{config}_h{h}_{year}"
                sample.mkdir(parents=True, exist_ok=False)
                parts = {}
                for part in ("inner_train", "inner_val", "test"):
                    frame = fold[f"{part}_meta"].copy()
                    frame["y_true"] = fold[f"y_{part}"]
                    frame["peg_deviation"] = fold[f"X_{part}"].peg_deviation
                    frame["price"] = frame.peg_deviation + 1
                    parts[part] = frame
                    frame.to_parquet(sample / f"{part}.parquet", index=False)
                for name, cls in (("constant_prior", ConstantPriorModel),
                                  ("persistence", PersistenceModel)):
                    fit_id = f"{name}_{config}_h{h}_{year}"
                    folder = out / "fits" / fit_id
                    folder.mkdir(parents=True, exist_ok=False)
                    model = cls().fit(fold["X_inner_train"], fold["y_inner_train"])
                    scores = {}
                    for part in ("inner_val", "test"):
                        frame = parts[part].copy()
                        frame["horizon"], frame["y_score"] = h, model.predict_proba(fold[f"X_{part}"])[:, 1]
                        frame["run_id"], frame["fit_id"], frame["config"], frame["model"] = run_id, fit_id, config, name
                        frame["test_year"] = year
                        scores[part] = frame
                    val, test = scores["inner_val"], scores["test"]
                    threshold, validation = _choose_threshold(val, episodes, h, 1.)
                    for frame in scores.values():
                        frame["threshold"] = threshold
                    _, events = build_alarm_events(test, episodes, threshold)
                    events = events.assign(run_id=run_id, fit_id=fit_id, config=config, model=name, test_year=year)
                    val.to_parquet(folder / "inner_val.parquet", index=False)
                    test.to_parquet(folder / "test.parquet", index=False)
                    events.to_csv(folder / "alarms.csv", index=False)
                    fit = dict(
                        run_id=run_id, fit_id=fit_id, model=name, config=config, horizon=h, year=year,
                        sample=str(sample.relative_to(out)), features=fold["features"],
                        score_inputs=[] if name == "constant_prior" else ["peg_deviation"],
                        fit_partition="inner_train", calibration_partition="inner_val",
                        train_rows=len(parts["inner_train"]), train_positive_hours=int(parts["inner_train"].y_true.sum()),
                        model_params=model.get_params(), threshold=threshold, validation=validation,
                    )
                    save_json(folder / "fit.json", fit)
                    result = dict(run_id=run_id, fit_id=fit_id, model=name, config=config,
                                  horizon_hours=h, test_year=year, threshold=threshold,
                                  training_prior=float(parts["inner_train"].y_true.mean()),
                                  train_rows=fit["train_rows"], train_positive_hours=fit["train_positive_hours"],
                                  validation_rows=len(val), validation_positive_hours=int(val.y_true.sum()),
                                  validation_positive_rate=float(val.y_true.mean()),
                                  validation_average_precision=float(average_precision_score(val.y_true, val.y_score)),
                                  **{f"validation_{k}": v for k, v in validation.items()},
                                  **metrics(test, events, episodes, h))
                    rows.append(result)
                    pooled[name][0].append(test)
                    pooled[name][1].append(events)
                    # These are the complete constant-prior candidates, including the all-row alarm case.
                    if name == "constant_prior":
                        for cutoff in (float(val.y_score.iloc[0]), np.inf):
                            candidate = val.assign(threshold=cutoff)
                            _, ev = build_alarm_events(candidate, episodes, cutoff)
                            candidates.append(dict(run_id=run_id, fit_id=fit_id, config=config,
                                                   horizon_hours=h, test_year=year, threshold=cutoff,
                                                   selected=cutoff == threshold, **metrics(candidate, ev, episodes, h)))
                    print(f"{fit_id}: AP={result['average_precision']:.9g}; "
                          f"caught={result['episodes_caught']}/{result['targetable_episodes']}; "
                          f"FA={result['false_alarms']}/{result['observed_coin_months']}; cutoff={threshold}", flush=True)
            for name, (preds, alarms) in pooled.items():
                pred, events = pd.concat(preds, ignore_index=True), pd.concat(alarms, ignore_index=True)
                pred.to_parquet(out / f"predictions_{name}_{config}_h{h}.parquet", index=False)
                events.to_csv(out / f"alarms_{name}_{config}_h{h}.csv", index=False)
                summaries.append(dict(run_id=run_id, model=name, config=config, horizon_hours=h,
                                      **metrics(pred, events, episodes, h)))
                targets = _targetable_episode_ids(pred, episodes, h)
                for ep in sorted(targets):
                    record = episodes.loc[episodes.episode_id.eq(ep)].iloc[0]
                    lead = events.loc[events.target_episode_id.eq(ep), "lead_hours"]
                    identities.append(dict(run_id=run_id, model=name, config=config, horizon_hours=h,
                                           episode_id=ep, coin=record.coin, episode_start=record.start,
                                           caught=not lead.empty, earliest_alarm_lead_hours=lead.max()))
    pd.DataFrame(rows).to_csv(out / "per_year.csv", index=False)
    pd.DataFrame(summaries).to_csv(out / "summary.csv", index=False)
    pd.DataFrame(identities).to_csv(out / "caught_identities.csv", index=False)
    pd.DataFrame(candidates).to_csv(out / "constant_prior_candidates.csv", index=False)
    info["status"] = "complete"
    save_json(out / "run.json", info)
    print(pd.DataFrame(summaries).to_string(index=False), flush=True)


def run(out, horizons=(1, 6, 24), years=range(2021, 2026)):
    if out.exists():
        raise FileExistsError(f"Use a new output directory: {out}")
    try:
        _run(out, horizons, years)
    except BaseException as error:
        manifest = out / "run.json"
        if manifest.exists():
            info = json.loads(manifest.read_text())
            info.update(status="incomplete", error=f"{type(error).__name__}: {error}")
            save_json(manifest, info)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--horizon", type=int, action="append", choices=(1, 6, 24))
    parser.add_argument("--years", type=int, nargs="+", choices=range(2021, 2026),
                        default=list(range(2021, 2026)))
    args = parser.parse_args()
    run(args.out_dir.resolve(), tuple(dict.fromkeys(args.horizon or (1, 6, 24))),
        tuple(dict.fromkeys(args.years)))
