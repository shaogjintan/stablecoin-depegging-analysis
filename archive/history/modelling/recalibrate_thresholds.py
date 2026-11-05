"""Re-evaluate verified saved scores under an explicit threshold policy; never fit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

from modelling.alarm_events import build_alarm_events
from modelling.corrected_lr_rf import metrics
from modelling.corrected_xgb import paired_uncertainty
from modelling.rolling_threshold import _choose_threshold, _targetable_episode_ids, normalize_policy


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


@njit
def independent_candidate_counts(scores, times, ends, targets, cutoffs, delta, max_false):
    """Regenerate each cutoff; stop only after its FA budget is already exceeded."""
    size = 1
    while size < len(scores):
        size *= 2
    tree = np.full(2 * size, -np.inf)
    tree[size:size + len(scores)] = scores
    for node in range(size - 1, 0, -1):
        tree[node] = max(tree[node * 2], tree[node * 2 + 1])
    false = np.zeros(len(cutoffs), dtype=np.int64)
    caught = np.zeros(len(cutoffs), dtype=np.int64)
    for k in range(len(cutoffs)):
        seen = np.zeros(max(len(scores), targets.max() + 1) + 1, dtype=np.bool_)
        begin = 0
        for end in ends:
            cursor = begin
            while cursor < end:
                # First qualifying score at/after cursor, via a static range-max tree.
                node = cursor + size
                while True:
                    while node % 2 == 0:
                        node //= 2
                    if tree[node] >= cutoffs[k]:
                        while node < size:
                            node *= 2
                            if tree[node] < cutoffs[k]:
                                node += 1
                        break
                    node += 1
                    if node & -node == node:
                        node = 2 * size
                        break
                pos = node - size
                if pos >= end:
                    break
                target = targets[pos]
                if target < 0:
                    false[k] += 1
                    if false[k] > max_false:
                        break
                elif not seen[target]:
                    seen[target] = True
                    caught[k] += 1
                cursor = np.searchsorted(times[begin:end], times[pos] + delta, side="right") + begin
            begin = end
            if false[k] > max_false:
                break
    return caught, false


def oracle_choice(pred, episodes, h, budget, policy):
    policy = normalize_policy(policy)
    frame = pred.sort_values(["coin", "hour"]).reset_index(drop=True)
    times = pd.to_datetime(frame.hour, utc=True).astype("datetime64[ns, UTC]").astype("int64").to_numpy()
    ids = {str(ep): i for i, ep in enumerate(episodes.episode_id)}
    targets = np.full(len(frame), -1, dtype=np.int64)
    delta = h * 3600_000_000_000
    for coin, group in frame.groupby("coin", sort=False):
        eps = episodes.loc[episodes.coin.eq(coin)].sort_values("start")
        starts = pd.to_datetime(eps.start, utc=True).astype("datetime64[ns, UTC]").astype("int64").to_numpy()
        for pos in group.index:
            j = np.searchsorted(starts, times[pos], side="right")
            if j < len(starts) and starts[j] <= times[pos] + delta:
                targets[pos] = ids[str(eps.iloc[j].episode_id)]
    cutoffs = np.r_[np.unique(frame.y_score), np.inf]
    ends = frame.groupby("coin", sort=False).size().cumsum().to_numpy(dtype=np.int64)
    months = frame[["coin"]].assign(month=frame.hour.dt.strftime("%Y-%m")).drop_duplicates().shape[0]
    caught, false = independent_candidate_counts(frame.y_score.to_numpy(dtype=float), times,
                                                  ends, targets, cutoffs, delta, budget * months)
    feasible = np.flatnonzero(false <= budget * months)
    idx = (feasible[0] if policy == "lowest_feasible" else
           max(feasible, key=lambda i: (caught[i], -false[i], cutoffs[i])))
    return float(cutoffs[idx]), int(caught[idx]), int(false[idx]), len(cutoffs)


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def run(sources, out, policy="lowest_feasible"):
    out.mkdir(parents=True, exist_ok=False)
    manifest = dict(status="in_progress", threshold_policy=policy,
                    budget="total validation grouped FA / observed eligible coin-months <= 1",
                    fits_reused=True, source_runs=[], limitation=(
                        "Retrospective policy comparison after inspecting test outcomes; "
                        "conditional saved fits; not an untouched confirmatory experiment."))
    save_json(out / "run.json", manifest)
    try:
        snapshot = out / "source"
        snapshot.mkdir()
        for name in ("recalibrate_thresholds.py", "rolling_threshold.py", "exact_threshold.py",
                     "alarm_events.py", "corrected_lr_rf.py", "corrected_xgb.py"):
            (snapshot / name).write_bytes(Path(__file__).with_name(name).read_bytes())
        manifest["evaluation_source_sha256"] = {
            p.name: digest(p) for p in snapshot.iterdir()}
        _evaluate(sources, out, policy, manifest)
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        save_json(out / "run.json", manifest)
        raise


def _evaluate(sources, out, policy, manifest):
    rows, old_rows, identities, provenance = [], [], [], []
    predictions, alarms = {}, {}
    source_episodes = None
    unique_cells = set()
    for source in sources:
        info = json.loads((source / "run.json").read_text())
        verified = json.loads((source / "verification.json").read_text())
        if info["status"] != "complete" or verified["status"] != "passed":
            raise ValueError(f"Source is not complete and verified: {source}")
        manifest["source_runs"].append(dict(path=str(source), run_id=info["run_id"],
                                            manifest_sha256=digest(source / "run.json")))
        episodes = pd.read_parquet(source / "episodes.parquet")
        if source_episodes is not None:
            pd.testing.assert_frame_equal(episodes, source_episodes, check_exact=True)
        source_episodes = episodes
        for folder in sorted((source / "fits").iterdir()):
            fit_path = folder / "fit.json"
            fit = json.loads(fit_path.read_text())
            model = fit.get("model", "xgboost")
            config, h, year = fit["config"], int(fit["horizon"]), int(fit["year"])
            key = model, config, h
            cell = (*key, year)
            if cell in unique_cells:
                raise ValueError(f"Duplicate source cell: {cell}")
            unique_cells.add(cell)
            if fit["fit_partition"] != "inner_train" or fit["calibration_partition"] != "inner_val":
                raise ValueError(f"Incompatible source partitions: {folder}")
            artifact = folder / ("model.joblib" if model == "xgboost" else "pipeline.joblib")
            expected = fit.get("model_sha256", fit.get("pipeline_sha256"))
            if digest(artifact) != expected:
                raise ValueError(f"Source fitted model hash mismatch: {artifact}")
            frames = {part: pd.read_parquet(folder / f"{part}.parquet")
                      for part in ("inner_val", "test")}
            val, test = frames["inner_val"], frames["test"]
            threshold, stats = _choose_threshold(val, episodes, h, 1., policy=policy)
            oracle, nc, nf, candidates = oracle_choice(val, episodes, h, 1., policy)
            if (threshold != oracle or stats["episodes_warned"] != nc
                    or stats["false_alarms"] != nf or stats["candidate_count"] != candidates):
                raise AssertionError(f"Independent threshold oracle disagreement: {cell}")
            old_policy = fit.get("threshold_policy", "max_validation_recall")
            old_threshold = float(fit["threshold"])
            if old_policy != "max_validation_recall":
                raise ValueError(f"Expected max-validation-recall sensitivity source: {folder}")
            if threshold > old_threshold:
                raise AssertionError(f"Primary cutoff exceeds feasible sensitivity cutoff: {cell}")
            targets = _targetable_episode_ids(test, episodes, h)
            test = test.copy()
            test["threshold"] = threshold
            _, events = build_alarm_events(test, episodes, threshold)
            old_test = test.assign(threshold=old_threshold)
            _, old_events = build_alarm_events(old_test, episodes, old_threshold)
            common = dict(model=model, config=config, horizon_hours=h, test_year=year)
            result = dict(common, **metrics(test, events, targets), threshold=threshold,
                          source_threshold=old_threshold, threshold_policy=policy,
                          validation_false_alarms=nf, validation_coin_months=stats["coin_months"],
                          validation_episode_recall=stats["episode_recall"],
                          threshold_candidates=candidates, independent_oracle_passed=True)
            old_result = dict(common, **metrics(old_test, old_events, targets), threshold=old_threshold,
                              threshold_policy=old_policy)
            if result["raw_flagged_hours"] < old_result["raw_flagged_hours"]:
                raise AssertionError(f"Raw flags decreased at lower threshold: {cell}")
            rows.append(result)
            old_rows.append(old_result)
            predictions.setdefault(key, []).append(test)
            alarms.setdefault(key, []).append(events)
            caught = set(events.target_episode_id.dropna().astype(str))
            for ep in sorted(targets):
                identities.append(dict(common, episode_id=ep, caught=ep in caught,
                    episode_start=episodes.set_index("episode_id").loc[ep, "start"],
                    earliest_alarm_lead_hours=events.loc[events.target_episode_id.eq(ep), "lead_hours"].max()))
            dest = out / "cells" / f"{model}_{config}_h{h}_{year}"
            dest.mkdir(parents=True)
            events.to_csv(dest / "alarms.csv", index=False)
            save_json(dest / "evaluation.json", dict(result, source_fit=str(fit_path),
                                                      validation=stats))
            provenance.append(dict(common, fit=str(fit_path), fit_sha256=digest(fit_path),
                model_sha256=expected, validation_sha256=digest(folder / "inner_val.parquet"),
                test_sha256=digest(folder / "test.parquet")))
            print(f"{model} {config} h{h} {year}: cutoff {old_threshold:g} -> {threshold:g}; "
                  f"caught {old_result['episodes_caught']} -> {result['episodes_caught']}; "
                  f"FA {old_result['false_alarms']} -> {result['false_alarms']}", flush=True)
    expected_cells = {(m, c, h, y) for m in ("xgboost", "logreg", "random_forest", "baseline", "persistence")
                      for c in ("market_original", "market_matched", "market_onchain")
                      for h in (1, 6, 24) for y in range(2021, 2026)}
    if unique_cells != expected_cells:
        raise ValueError(f"Full matrix required; missing={expected_cells - unique_cells}; "
                         f"extra={unique_cells - expected_cells}")
    yearly, sensitivity = pd.DataFrame(rows), pd.DataFrame(old_rows)
    yearly.to_csv(out / "primary_per_year.csv", index=False)
    sensitivity.to_csv(out / "sensitivity_per_year.csv", index=False)
    pd.DataFrame(provenance).to_csv(out / "source_provenance.csv", index=False)
    identities = pd.DataFrame(identities)
    identities.to_csv(out / "caught_identities.csv", index=False)
    pooled, uncertainty = [], []
    for key, parts in predictions.items():
        model, config, h = key
        pred, events = pd.concat(parts, ignore_index=True), pd.concat(alarms[key], ignore_index=True)
        targets = set(identities.loc[identities.model.eq(model) & identities.config.eq(config)
                                     & identities.horizon_hours.eq(h), "episode_id"])
        pooled.append(dict(model=model, config=config, horizon_hours=h,
                           threshold_policy=policy, **metrics(pred, events, targets)))
    for model in ("xgboost", "logreg", "random_forest", "baseline", "persistence"):
        for h in (1, 6, 24):
            a = pd.concat(predictions[(model, "market_matched", h)], ignore_index=True)
            b = pd.concat(predictions[(model, "market_onchain", h)], ignore_index=True)
            if model in ("baseline", "persistence"):
                pd.testing.assert_frame_equal(a[["coin", "hour", "y_true", "y_score", "threshold"]],
                                              b[["coin", "hour", "y_true", "y_score", "threshold"]])
            # AP uncertainty depends only on unchanged scores; use the existing helper
            # to retain its exact resampling definitions and seeds.
            if model not in ("baseline", "persistence"):
                uncertainty.append(dict(model=model, **paired_uncertainty(
                    a, b, identities.loc[identities.model.eq(model)], h)))
    pd.DataFrame(pooled).to_csv(out / "primary_pooled.csv", index=False)
    pd.DataFrame(uncertainty).to_csv(out / "matched_uncertainty.csv", index=False)
    for source in sources:
        recorded = next(x for x in manifest["source_runs"] if x["path"] == str(source))
        if digest(source / "run.json") != recorded["manifest_sha256"]:
            raise AssertionError(f"Source manifest changed during evaluation: {source}")
    for item in provenance:
        fit_path = Path(item["fit"])
        if digest(fit_path) != item["fit_sha256"]:
            raise AssertionError(f"Source fit changed: {fit_path}")
        for part in ("inner_val", "test"):
            if digest(fit_path.parent / f"{part}.parquet") != item[
                    "validation_sha256" if part == "inner_val" else "test_sha256"]:
                raise AssertionError(f"Source scores changed: {fit_path}")
    manifest.update(status="complete", cells=len(yearly), pooled_rows=len(pooled))
    save_json(out / "run.json", manifest)
    save_json(out / "verification.json", dict(status="passed", cells=len(yearly),
        independent_full_candidate_oracle=True, source_scores_unchanged=True,
        models_reused=True, primary_cutoffs_not_higher=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    run([p.resolve() for p in args.source_run], args.out_dir.resolve())
