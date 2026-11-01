"""Independent saved-prediction oracle and LR/RF scoring-fit verification."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

from modelling.corrected_xgb import sha256, save_json
from modelling.corrected_lr_rf import CONFIGS, MODELS, PARTS, RF_PARAMS

HOUR = 3600_000_000_000
KEYS = ["coin", "hour", "y_true"]


def reference_ap(y, scores):
    y, scores = np.asarray(y, dtype=int), np.asarray(scores, dtype=float)
    assert len(y) and np.isfinite(scores).all()
    if y.sum() == 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    labels, ranks = y[order], scores[order]
    ends = np.r_[np.flatnonzero(ranks[1:] != ranks[:-1]), len(ranks) - 1]
    positive = labels.cumsum()[ends]
    gained = np.diff(np.r_[0, positive])
    return float(np.sum(gained * positive / (ends + 1)) / labels.sum())


def reference_events(pred, episodes, cutoff, h):
    rows = []
    cooldown = h * HOUR
    for coin, g in pred.groupby("coin", sort=True):
        e = episodes.loc[episodes.coin.eq(coin)].sort_values("start")
        starts = e.start.astype("int64").to_numpy()
        ids = e.episode_id.to_numpy()
        flagged = g.loc[g.y_score.ge(cutoff)].sort_values("hour")
        last = None
        for hour, score in flagged[["hour", "y_score"]].itertuples(index=False, name=None):
            t = hour.value
            if last is not None and t - last <= cooldown:
                continue
            last = t
            pos = np.searchsorted(starts, t, side="right")
            caught = pos < len(starts) and starts[pos] - t <= cooldown
            rows.append(dict(coin=coin, alarm_hour=hour, y_score=score, is_true_alarm=bool(caught),
                             target_episode_id=ids[pos] if caught else None,
                             lead_hours=(starts[pos] - t) / HOUR if caught else np.nan))
    return pd.DataFrame(rows, columns=["coin", "alarm_hour", "y_score", "is_true_alarm",
                                      "target_episode_id", "lead_hours"])


def reference_targets(pred, episodes, h):
    targets = set()
    for coin, g in pred.groupby("coin", sort=False):
        times = np.sort(g.hour.astype("int64").to_numpy())
        for ep, start in episodes.loc[episodes.coin.eq(coin), ["episode_id", "start"]].itertuples(
                index=False, name=None):
            pos = np.searchsorted(times, start.value, side="left") - 1
            if pos >= 0 and start.value - times[pos] <= h * HOUR:
                targets.add(ep)
    return targets


def reference_metrics(pred, events, episodes, h):
    targets = reference_targets(pred, episodes, h)
    caught = set(events.target_episode_id.dropna()) & targets
    true = events.is_true_alarm.astype(bool)
    false = events.loc[~true]
    months = len(pred[["coin", "hour"]].assign(month=pred.hour.dt.strftime("%Y-%m"))[
        ["coin", "month"]].drop_duplicates())
    costs = false.assign(month=false.alarm_hour.map(lambda t: t.strftime("%Y-%m"))).groupby(
        ["coin", "month"]).size()
    return dict(rows=len(pred), positive_hours=int(pred.y_true.sum()),
                positive_rate=float(pred.y_true.mean()), average_precision=reference_ap(pred.y_true, pred.y_score),
                targetable_episodes=len(targets), episodes_caught=len(caught),
                episode_recall=len(caught) / len(targets) if targets else np.nan,
                raw_flagged_hours=int(pred.y_score.ge(pred.threshold).sum()),
                alarm_events=len(events), true_alarms=int(true.sum()), false_alarms=len(false),
                observed_coin_months=months, false_alarms_per_coin_month=len(false) / months,
                max_false_alarms_in_coin_month=int(costs.max()) if len(costs) else 0,
                median_lead_hours=float(events.lead_hours.median()))


def assert_values(actual, expected):
    for k, v in expected.items():
        np.testing.assert_allclose(actual[k], v, atol=1e-12, rtol=0, equal_nan=True, err_msg=k)


def check_preprocessing(bundle, train, features):
    model, imputer = bundle["model"], bundle["imputer"]
    X = train[features]
    if imputer is not None:
        pd.testing.assert_series_equal(imputer.medians_, X.median())
        X = imputer.transform(X)
    if model.name == "logreg":
        assert model.penalty == "l1" and model.class_weight == "balanced" and model.winsor == .01
        X = X.copy()
        X["abs_peg_dev"] = X.peg_deviation.abs()
        X["dev_x_vol6"] = X.peg_deviation * X.realised_vol_6h
        assert list(X.columns) == model.cols_
        pd.testing.assert_series_equal(model.lo_, X.quantile(.01))
        pd.testing.assert_series_equal(model.hi_, X.quantile(.99))
        a = X.clip(model.lo_, model.hi_, axis=1).to_numpy()
        np.testing.assert_allclose(model.scaler_.mean_, a.mean(axis=0), atol=1e-12, rtol=1e-12)
        variance = a.var(axis=0)
        np.testing.assert_allclose(model.scaler_.var_, variance, atol=1e-12, rtol=1e-12)
        assert model.clf_.solver == "liblinear" and model.clf_.random_state == 42
        assert max(model.clf_.n_iter_) < model.clf_.max_iter
        np.testing.assert_array_equal(model.clf_.classes_, [0, 1])
    elif model.name == "random_forest":
        assert_values(model.clf_.get_params(), {k: v for k, v in RF_PARAMS.items() if isinstance(v, int)})
        for k, v in RF_PARAMS.items():
            assert model.clf_.get_params()[k] == v
        assert model.feature_names_ == features and len(model.clf_.estimators_) == 500
        np.testing.assert_array_equal(model.clf_.classes_, [0, 1])
    elif model.name == "baseline":
        assert model.positive_rate_ == train.y_true.mean()


def verify_cv(folder, train, fit, purge_hours=24, embargo_hours=24):
    tuning = json.loads((folder / "tuning.json").read_text())
    assert tuning["grid"] == [.001, .01, .1, 1.]
    assert not tuning["calibration_used_for_selection"]
    candidates = []
    count = 0
    for candidate in tuning["candidates"]:
        scores = []
        for record in candidate["per_year"]:
            d = folder / "cv" / str(record["validation_year"])
            a, b = pd.read_parquet(d / "train.parquet"), pd.read_parquet(d / "validation.parquet")
            assert a.hour.max() + pd.Timedelta(hours=fit["horizon"] + 1) < b.hour.min()
            assert b.hour.dt.year.eq(record["validation_year"]).all()
            assert a.hour.dt.year.lt(record["validation_year"]).all()
            assert a.hour.max() < pd.Timestamp(f"{record['validation_year']}-01-01", tz="UTC") - pd.Timedelta(hours=purge_hours)
            assert b.hour.min() >= pd.Timestamp(f"{record['validation_year']}-01-01", tz="UTC") + pd.Timedelta(hours=embargo_hours)
            for p in (a, b):
                selected = p[["coin", "hour"]].merge(train, on=["coin", "hour"], validate="one_to_one")
                pd.testing.assert_frame_equal(p.reset_index(drop=True), selected[p.columns])
            assert b.hour.max() <= train.hour.max()
            bundle = joblib.load(d / f"C{candidate['C']}.joblib")
            check_preprocessing(bundle, a, fit["features"])
            assert bundle["model"].C == candidate["C"]
            p = pd.read_parquet(d / f"C{candidate['C']}.parquet")
            pd.testing.assert_frame_equal(p[KEYS], b[KEYS].reset_index(drop=True))
            scores_actual = bundle["model"].predict_proba(bundle["imputer"].transform(b[fit["features"]]))[:, 1]
            np.testing.assert_allclose(scores_actual, p.y_score, atol=1e-8, rtol=0)
            ap = reference_ap(p.y_true, p.y_score)
            np.testing.assert_allclose(ap, record["average_precision"], atol=1e-12, rtol=0)
            scores.append(ap)
            count += 1
        mean = float(np.mean(scores))
        np.testing.assert_allclose(mean, candidate["mean_average_precision"], atol=1e-12, rtol=0)
        candidates.append((mean, -candidate["C"]))
    assert tuning["selected_C"] == -max(candidates)[1] == fit["params"]["C"]
    return count


def verify(out):
    started = time.perf_counter()
    info = json.loads((out / "run.json").read_text())
    assert info["status"] in ("scoring_complete", "complete")
    frozen = Path(info["frozen_path"]) if info.get("frozen_path") else None
    if frozen:
        frozen_hashes = json.loads((out / "frozen_evidence_sha256.json").read_text())
        assert set(frozen_hashes) == {str(p.relative_to(frozen)) for p in frozen.rglob("*") if p.is_file()}
        assert all(sha256(frozen / p) == digest for p, digest in frozen_hashes.items())
    horizons = tuple(info.get("horizons", (6, 24)))
    test_years = tuple(info.get("test_years", range(2021, 2026)))
    configs = tuple(info.get("configs", CONFIGS))
    models = tuple(info.get("models", MODELS))
    purge_hours = info.get("purge_hours", 24)
    embargo_hours = info.get("embargo_hours", 24)
    for rel, digest in info["source_code_sha256"].items():
        assert sha256(out / "source" / rel) == digest
    episodes = pd.read_parquet(out / "episodes.parquet")
    if frozen:
        pd.testing.assert_frame_equal(episodes, pd.read_parquet(frozen / "episodes.parquet"))
    else:
        assert sha256(out / "episodes.parquet") == info["episodes_sha256"]
        assert sha256(out / "corrected_master.parquet") == info["corrected_master_sha256"]
        master = pd.read_parquet(out / "corrected_master.parquet")
    cells = json.loads((out / "partitions.json").read_text())
    assert set(cells) == {f"{c}_h{h}_{y}" for c in configs for h in horizons for y in test_years}
    expected_fits = {f"{name}_{cell}" for name in models for cell in cells}
    assert {p.name for p in (out / "fits").iterdir()} == expected_fits
    summary = pd.read_csv(out / "summary.csv")
    years = pd.read_csv(out / "per_year.csv")
    importance = (pd.read_csv(out / "importance_per_fit.csv")
                  if (out / "importance_per_fit.csv").exists() else pd.DataFrame())
    assert len(summary) == len(models) * len(configs) * len(horizons)
    assert len(years) == len(expected_fits)
    records, cv_count = [], 0
    checkpoint = out / "verification_checkpoint.json"
    checked = json.loads(checkpoint.read_text()) if checkpoint.exists() and checkpoint.stat().st_size else {}
    with threadpool_limits(limits=4):
        for cell, spec in cells.items():
            parts = {p: pd.read_parquet(out / "partitions" / cell / f"{p}.parquet") for p in PARTS}
            for part, p in parts.items():
                assert not p.duplicated(["coin", "hour"]).any()
                assert np.isfinite(p[spec["features"]]).all().all()
                if frozen:
                    frozen_part = pd.read_parquet(frozen / "fits" / cell / f"{part}.parquet")
                    pd.testing.assert_frame_equal(p, frozen_part[p.columns])
                else:
                    name = "market" if spec["config"] == "market_original" else "market_onchain"
                    base = master.loc[master[f"prediction_eligible_{name}_h{spec['horizon']}"]]
                    val = pd.Timestamp(f"{spec['year']-1}-01-01", tz="UTC")
                    test = pd.Timestamp(f"{spec['year']}-01-01", tz="UTC")
                    end = pd.Timestamp(f"{spec['year']+1}-01-01", tz="UTC")
                    purge, embargo = pd.Timedelta(hours=purge_hours), pd.Timedelta(hours=embargo_hours)
                    dropped = set()
                    for boundary in (val, test):
                        dropped.update(episodes.loc[
                            (episodes.end >= boundary-purge) & (episodes.start <= boundary+embargo),
                            "episode_id"])
                    masks = {
                        "inner_train": base.hour < val-purge,
                        "inner_val": (base.hour >= val+embargo) & (base.hour < test-purge),
                        "test": (base.hour >= test+embargo) & (base.hour < end),
                    }
                    expected = base.loc[masks[part] & ~base.episode_id.isin(dropped)].sort_values(
                        ["coin", "hour"])
                    expected = expected[["coin", "hour", "episode_id"] + spec["features"]
                                        + [f"label_h{spec['horizon']}"]].rename(
                        columns={f"label_h{spec['horizon']}": "y_true"}).reset_index(drop=True)
                    expected["y_true"] = expected.y_true.astype(int)
                    expected[spec["features"]] = expected[spec["features"]].astype(float)
                    if "dataset_version" in p:
                        expected["dataset_version"] = info["dataset_version"]
                    pd.testing.assert_frame_equal(p, expected[p.columns])
                truth = np.zeros(len(p), dtype=int)
                for coin, group in p.groupby("coin", sort=False):
                    starts = np.sort(episodes.loc[episodes.coin.eq(coin), "start"].astype("int64").to_numpy())
                    hours = group.hour.astype("int64").to_numpy()
                    truth[group.index] = (np.searchsorted(starts, hours + spec["horizon"] * HOUR, side="right") >
                                          np.searchsorted(starts, hours, side="right"))
                np.testing.assert_array_equal(truth, p.y_true)
            for earlier, later in (("inner_train", "inner_val"), ("inner_val", "test")):
                gap = parts[later].hour.min() - parts[earlier].hour.max()
                assert gap >= pd.Timedelta(hours=purge_hours + embargo_hours + 1)
                assert parts[earlier].hour.max() + pd.Timedelta(hours=spec["horizon"] + 1) < parts[later].hour.min()
            for name in models:
                folder = out / "fits" / f"{name}_{cell}"
                if folder.name in checked:
                    saved = checked[folder.name]
                    assert all(sha256(out / rel) == digest for rel, digest in saved["hashes"].items())
                    records.append(saved["record"])
                    cv_count += saved["cv_count"]
                    continue
                fit = json.loads((folder / "fit.json").read_text())
                assert fit["run_id"] == info["run_id"] and not fit["refit_after_calibration"]
                assert fit["fit_partition"] == "inner_train" and fit["calibration_partition"] == "inner_val"
                assert sha256(folder / "pipeline.joblib") == fit["pipeline_sha256"]
                bundle = joblib.load(folder / "pipeline.joblib")
                check_preprocessing(bundle, parts["inner_train"], spec["features"])
                fit_cv_count = verify_cv(folder, parts["inner_train"], fit, purge_hours, embargo_hours) if name == "logreg" else 0
                cv_count += fit_cv_count
                frames = {}
                for part in ("inner_val", "test"):
                    pred = pd.read_parquet(folder / f"{part}.parquet")
                    pd.testing.assert_frame_equal(pred[KEYS], parts[part][KEYS])
                    X = parts[part][spec["features"]]
                    if bundle["imputer"] is not None:
                        X = bundle["imputer"].transform(X)
                    proba = bundle["model"].predict_proba(X)
                    assert proba.shape == (len(X), 2) and np.isfinite(proba).all()
                    np.testing.assert_allclose(proba[:, 1], pred.y_score, atol=1e-8, rtol=0)
                    frames[part] = pred
                h = spec["horizon"]
                val_events = reference_events(frames["inner_val"], episodes, fit["threshold"], h)
                val_metrics = reference_metrics(frames["inner_val"], val_events, episodes, h)
                assert val_metrics["false_alarms"] == fit["validation"]["false_alarms"]
                assert val_metrics["observed_coin_months"] == fit["validation"]["coin_months"]
                assert val_metrics["false_alarms_per_coin_month"] <= 1
                assert fit["validation"]["candidate_count"] == frames["inner_val"].y_score.nunique() + 1
                assert_values({"episode_recall": fit["validation"]["episode_recall"]},
                              {"episode_recall": val_metrics["episode_recall"]
                               if val_metrics["targetable_episodes"] else 0.})
                actual = pd.read_csv(folder / "alarms.csv")
                actual["alarm_hour"] = pd.to_datetime(actual.alarm_hour, utc=True)
                expected = reference_events(frames["test"], episodes, fit["threshold"], h)
                pd.testing.assert_frame_equal(actual[expected.columns], expected, check_dtype=False,
                                              atol=1e-12, rtol=0)
                row = years.loc[years.model.eq(name) & years.config.eq(spec["config"]) &
                                years.horizon_hours.eq(h) & years.test_year.eq(spec["year"])]
                assert len(row) == 1
                assert_values(row.iloc[0], reference_metrics(frames["test"], expected, episodes, h))
                if name in MODELS[:2]:
                    imp = importance.loc[importance.fit_id.eq(folder.name)]
                    values = (bundle["model"].clf_.coef_[0] if name == "logreg"
                              else bundle["model"].feature_importances_)
                    names = bundle["model"].cols_ if name == "logreg" else spec["features"]
                    assert imp.feature.tolist() == names
                    np.testing.assert_allclose(imp.value, values, atol=1e-12, rtol=0)
                records.append(dict(fit_id=folder.name, score_reload=True, independent_alarm_match=True,
                                    frozen_partition_match=bool(frozen),
                                    saved_partition_match=True, validation_budget_pass=True))
                checked[folder.name] = dict(record=records[-1], cv_count=fit_cv_count,
                                            hashes={str(p.relative_to(out)): sha256(p)
                                                    for p in folder.rglob("*") if p.is_file()})
                temporary = checkpoint.with_suffix(".tmp")
                save_json(temporary, checked)
                temporary.replace(checkpoint)
                print(f"Verified {len(records)}/{len(expected_fits)}: {folder.name}", flush=True)
    for r in summary.to_dict("records"):
        name, config, h = r["model"], r["config"], r["horizon_hours"]
        pred = pd.read_parquet(out / f"predictions_{name}_{config}_h{h}.parquet")
        assert not pred.duplicated(["coin", "hour"]).any()
        if frozen:
            frozen_pred = pd.read_parquet(frozen / f"predictions_{config}_h{h}.parquet")
            pd.testing.assert_frame_equal(pred[KEYS], frozen_pred[KEYS])
        ev = pd.concat([reference_events(p, episodes, p.threshold.iloc[0], h)
                        for _, p in pred.groupby("test_year", sort=True)], ignore_index=True)
        assert_values(r, reference_metrics(pred, ev, episodes, h))
        np.testing.assert_allclose(reference_ap(pred.y_true, pred.y_score),
                                   average_precision_score(pred.y_true, pred.y_score), atol=1e-12, rtol=0)
        targets = reference_targets(pred, episodes, h)
        if frozen:
            frozen_ids = pd.read_csv(frozen / "caught_identities.csv")
            assert targets == set(frozen_ids.loc[frozen_ids.config.eq(config) &
                                                frozen_ids.horizon_hours.eq(h), "episode_id"])
    matched_checks = 0
    for name in models:
        if not {"market_matched", "market_onchain"}.issubset(configs):
            continue
        for h in horizons:
            a, b = [pd.read_parquet(out / f"predictions_{name}_{c}_h{h}.parquet")
                    for c in ("market_matched", "market_onchain")]
            pd.testing.assert_frame_equal(a[KEYS], b[KEYS])
            matched_checks += 1
    if frozen:
        assert all(sha256(frozen / p) == digest for p, digest in frozen_hashes.items())
    pd.DataFrame(records).to_csv(out / "verification_fits.csv", index=False)
    info.update(status="complete", independent_verification="passed")
    save_json(out / "run.json", info)
    save_json(out / "verification.json", dict(
        status="passed", run_id=info["run_id"], fits_checked=len(records), cv_candidate_fits_checked=cv_count,
        pooled_rows_checked=len(summary),
        frozen_partition_checks=len(cells)*len(PARTS) if frozen else 0,
        saved_partition_checks=len(cells)*len(PARTS), matched_pooled_checks=matched_checks,
        independent_ap=True, independent_alarm_events=True, independent_targets_and_cost=True,
        frozen_xgb_shap_unchanged=bool(frozen), elapsed_seconds=time.perf_counter() - started,
        limitation="Every cutoff regression-tested on synthetic reference; real selected thresholds checked for counts/budget, not independently re-searched."))
    save_json(out / "artifact_sha256.json", {str(p.relative_to(out)): sha256(p)
                                            for p in out.rglob("*") if p.is_file() and
                                            p.name != "artifact_sha256.json"})
    print(f"PASS {len(records)} fits, {cv_count} CV candidate fits, {len(summary)} pooled rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    verify(parser.parse_args().run_dir.resolve())
