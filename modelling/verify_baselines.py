"""Independent tie-aware AP, greedy alarms and full-scan threshold verification."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

HOUR = 3600_000_000_000
KEYS = ["coin", "hour", "y_true"]
EVENT_KEYS = ["coin", "alarm_hour", "target_episode_id", "lead_hours", "is_true_alarm"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def independent_ap(y, scores):
    y, scores = np.asarray(y), np.asarray(scores)
    assert len(y) and np.isfinite(scores).all() and np.isin(y, [0, 1]).all()
    if not y.sum():
        return 0.
    order = np.argsort(-scores, kind="stable")
    ranked = y[order]
    ends = np.r_[np.flatnonzero(np.diff(scores[order]) != 0) + 1, len(y)]
    tp = np.cumsum(ranked)[ends - 1]
    return float(np.sum(np.diff(np.r_[0, tp]) * tp / ends) / y.sum())


def target_codes(pred, episodes, h):
    ordered = pred.sort_values(["coin", "hour"]).reset_index(drop=True)
    codes = pd.factorize(ordered.coin, sort=False)[0].astype(np.int64)
    targets = np.full(len(ordered), -1, dtype=np.int64)
    times = ordered.hour.astype("int64").to_numpy()
    eps = episodes.sort_values(["coin", "start"]).reset_index(drop=True)
    for coin, g in ordered.groupby("coin", sort=False):
        e = eps.loc[eps.coin.eq(coin)]
        starts = e.start.astype("int64").to_numpy()
        idx = g.index.to_numpy()
        nxt = np.searchsorted(starts, times[idx], side="right")
        valid = nxt < len(starts)
        rows = idx[valid]
        hits = nxt[valid]
        within = starts[hits] <= times[rows] + h * HOUR
        targets[rows[within]] = e.index.to_numpy()[hits[within]]
    np.testing.assert_array_equal(targets >= 0, ordered.y_true.eq(1))
    return ordered, times, codes, targets, eps


@njit
def full_scan_candidates(scores, times, coins, targets, cutoffs, horizon_ns, n_episodes):
    """Independent O(rows * candidates) oracle, not the production Fenwick search."""
    counts = np.zeros((len(cutoffs), 3), dtype=np.int64)
    for j in range(len(cutoffs)):
        caught = np.zeros(n_episodes, dtype=np.bool_)
        last_coin, last_hour = -1, 0
        for i in range(len(scores)):
            if scores[i] < cutoffs[j]:
                continue
            if coins[i] == last_coin and times[i] <= last_hour + horizon_ns:
                continue
            last_coin, last_hour = coins[i], times[i]
            counts[j, 2] += 1
            if targets[i] < 0:
                counts[j, 1] += 1
            elif not caught[targets[i]]:
                caught[targets[i]] = True
                counts[j, 0] += 1
    return counts


def independent_events(pred, episodes, h):
    p, times, coins, targets, eps = target_codes(pred, episodes, h)
    rows = []
    last_coin, last_hour = None, None
    for i in np.flatnonzero(p.y_score.ge(p.threshold).to_numpy()):
        if last_coin == coins[i] and times[i] <= last_hour + h * HOUR:
            continue
        last_coin, last_hour = coins[i], times[i]
        target = int(targets[i])
        start = eps.iloc[target].start if target >= 0 else pd.NaT
        rows.append(dict(coin=p.iloc[i].coin, alarm_hour=p.iloc[i].hour,
                         target_episode_id=str(eps.iloc[target].episode_id) if target >= 0 else None,
                         lead_hours=(start.value - times[i]) / HOUR if target >= 0 else np.nan,
                         is_true_alarm=target >= 0))
    events = pd.DataFrame(rows, columns=EVENT_KEYS)
    events["is_true_alarm"] = events.is_true_alarm.astype(bool)
    target_ids = set(eps.loc[np.unique(targets[targets >= 0]), "episode_id"].astype(str))
    return events, target_ids


def independent_metrics(pred, events, targets):
    caught = set(events.target_episode_id.dropna().astype(str)) & targets
    false = events.loc[~events.is_true_alarm]
    months = set(zip(pred.coin, pred.hour.dt.strftime("%Y-%m")))
    false_months = false.assign(month=pd.to_datetime(false.alarm_hour, utc=True).dt.strftime("%Y-%m"))
    monthly = false_months.groupby(["coin", "month"]).size()
    return dict(
        rows=len(pred), positive_hours=int(pred.y_true.sum()), positive_rate=float(pred.y_true.mean()),
        average_precision=independent_ap(pred.y_true, pred.y_score),
        raw_flagged_hours=int(pred.y_score.ge(pred.threshold).sum()),
        targetable_episodes=len(targets), episodes_caught=len(caught),
        episode_recall=len(caught) / len(targets) if targets else np.nan,
        alarm_events=len(events), true_alarms=int(events.is_true_alarm.sum()),
        false_alarms=len(false), observed_coin_months=len(months),
        false_alarms_per_coin_month=len(false) / len(months),
        max_false_alarms_in_coin_month=int(monthly.max()) if len(monthly) else 0,
        median_lead_hours=float(events.loc[events.is_true_alarm, "lead_hours"].median()),
    )


def assert_metrics(actual, expected):
    for key, value in actual.items():
        np.testing.assert_allclose(value, expected[key], atol=1e-12, rtol=0, equal_nan=True,
                                   err_msg=key)


def assert_events(actual, path):
    saved = pd.read_csv(path)
    saved["alarm_hour"] = pd.to_datetime(saved.alarm_hour, utc=True)
    actual = actual.copy()
    actual["alarm_hour"] = pd.to_datetime(actual.alarm_hour, utc=True)
    for d in (actual, saved):
        d["target_episode_id"] = d.target_episode_id.fillna("").astype(str)
    pd.testing.assert_frame_equal(
        actual[EVENT_KEYS].sort_values(["coin", "alarm_hour"]).reset_index(drop=True),
        saved[EVENT_KEYS].sort_values(["coin", "alarm_hour"]).reset_index(drop=True),
        check_dtype=False, check_exact=False, atol=1e-12, rtol=0,
    )


def check_samples(out, master, episodes, purge_hours=24, embargo_hours=24,
                  horizons=(1, 6, 24), years=range(2021, 2026)):
    checked = 0
    for sample in sorted((out / "samples").iterdir()):
        config, horizon, year = sample.name.rsplit("_", 2)
        h, y = int(horizon[1:]), int(year)
        eligibility = "market" if config == "market_original" else "market_onchain"
        base = master.loc[master[f"prediction_eligible_{eligibility}_h{h}"]]
        bval, btest = pd.Timestamp(f"{y-1}-01-01", tz="UTC"), pd.Timestamp(f"{y}-01-01", tz="UTC")
        end = pd.Timestamp(f"{y+1}-01-01", tz="UTC")
        purge = pd.Timedelta(hours=purge_hours)
        embargo = pd.Timedelta(hours=embargo_hours)
        drop = set()
        for boundary in (bval, btest):
            drop.update(episodes.loc[(episodes.end >= boundary - purge) &
                                     (episodes.start <= boundary + embargo), "episode_id"])
        masks = {
            "inner_train": base.hour < bval - purge,
            "inner_val": (base.hour >= bval + embargo) & (base.hour < btest - purge),
            "test": (base.hour >= btest + embargo) & (base.hour < end),
        }
        parts = {}
        for part, mask in masks.items():
            saved = pd.read_parquet(sample / f"{part}.parquet")
            expected = base.loc[mask & ~base.episode_id.isin(drop)].sort_values(["coin", "hour"])
            expected = expected[["coin", "hour", f"label_h{h}"]].rename(columns={f"label_h{h}": "y_true"})
            expected.y_true = expected.y_true.astype(int)
            pd.testing.assert_frame_equal(saved[KEYS], expected.reset_index(drop=True))
            target_codes(saved, episodes, h)
            source = saved[["coin", "hour"]].merge(master[["coin", "hour", "price", "bar_start", "peg_deviation"]],
                                                 on=["coin", "hour"], validate="one_to_one")
            np.testing.assert_allclose(saved.price, source.price, atol=0, rtol=0)
            np.testing.assert_allclose(saved.peg_deviation, source.peg_deviation, atol=0, rtol=0)
            assert (source.hour - source.bar_start).eq(pd.Timedelta(hours=1)).all()
            assert saved.price.ge(.99).all()
            parts[part] = saved
            checked += 1
        gap = pd.Timedelta(hours=purge_hours + embargo_hours + 1)
        assert parts["inner_val"].hour.min() - parts["inner_train"].hour.max() >= gap
        assert parts["test"].hour.min() - parts["inner_val"].hour.max() >= gap
    for h in horizons:
        for year in years:
            for part in ("inner_train", "inner_val", "test"):
                a, b = [pd.read_parquet(out / "samples" / f"{c}_h{h}_{year}" / f"{part}.parquet")
                        for c in ("market_matched", "market_onchain")]
                pd.testing.assert_frame_equal(a, b)
    return checked


def compatibility(out, references, info, save=True):
    rows, comparisons = [], []
    summary = pd.read_csv(out / "summary.csv")
    comparisons.extend(summary.assign(source_run_id=info["run_id"], source_path=str(out),
                                       compatibility="verified baseline").to_dict("records"))
    for reference in references:
        other = json.loads((reference / "run.json").read_text())
        hashes = other["source_code_sha256"]
        protocol = {k: v for k, v in info["source_code_sha256"].items()
                    if k in hashes and k not in (
                        "modelling\\false_alarm_framework.py", "modelling\\models\\baseline.py",
                        "modelling\\corrected_baselines.py", "modelling\\verify_baselines.py",
                        "modelling\\checks\\test_corrected_baselines.py")}
        assert len(protocol) >= 4
        for rel, expected in protocol.items():
            assert hashes[rel] == expected, f"Shared protocol differs: {reference}: {rel}"
            snapshot = reference / "source" / rel
            if not snapshot.exists():
                snapshot = reference / "source" / Path(rel).name
            assert digest(snapshot) == expected
        assert other["source_master_sha256"] == info["source_master_sha256"]
        if (reference / "corrected_master.parquet").exists():
            assert digest(reference / "corrected_master.parquet") == info["corrected_master_sha256"]
        verified = (reference / "verification.json").exists()
        if verified:
            verified = json.loads((reference / "verification.json").read_text())["status"] == "passed"
        complete = other["status"] == "complete" and verified and not other.get("benchmark", False)
        partitions = 0
        for folder in sorted((reference / "fits").iterdir()):
            fit = json.loads((folder / "fit.json").read_text())
            h, year, config = int(fit["horizon"]), int(fit["year"]), fit["config"]
            sample = out / "samples" / f"{config}_h{h}_{year}"
            assert fit["fit_partition"] == "inner_train" and fit["calibration_partition"] == "inner_val"
            for part in ("inner_train", "inner_val", "test"):
                path = (reference / fit["partition_paths"][part]
                        if "partition_paths" in fit else folder / f"{part}.parquet")
                if not path.exists():
                    assert not complete
                    continue
                a, b = pd.read_parquet(sample / f"{part}.parquet"), pd.read_parquet(path)
                pd.testing.assert_frame_equal(a[KEYS], b[KEYS])
                partitions += 1
        rows.append(dict(run_id=info["run_id"], reference_run_id=other["run_id"],
                         reference_path=str(reference), status=other["status"],
                         shared_protocol_hashes_checked=len(protocol), partitions_checked=partitions,
                         eligible_for_comparison=complete,
                         caveat="Different runner/model/documentation versions; shared protocol and saved partitions checked"))
        if complete:
            data = pd.read_csv(reference / "summary.csv")
            if "model" not in data:
                data["model"] = "xgboost"
            # Do not duplicate the other agent's baselines in the model comparison.
            data = data.loc[~data.model.isin(["baseline", "constant_prior", "persistence"])]
            data["source_run_id"], data["source_path"] = other["run_id"], str(reference)
            data["compatibility"] = "shared protocol and saved partitions exact"
            comparisons.extend(data.to_dict("records"))
    if save:
        pd.DataFrame(rows).to_csv(out / "compatibility.csv", index=False)
        pd.DataFrame(comparisons).to_csv(out / "final_model_comparisons.csv", index=False)
    return rows


def verify(out, references, check_only=False):
    if not check_only and (out / "verification.json").exists():
        raise FileExistsError("Verification evidence already exists; use --check-only to reverify without overwriting")
    info = json.loads((out / "run.json").read_text())
    assert info["status"] == "complete"
    print(f"VERIFY RUN {info['run_id']}; checker SHA256 {digest(Path(__file__))}; check_only={check_only}", flush=True)
    assert digest(out / "corrected_master.parquet") == info["corrected_master_sha256"]
    assert digest(out / "episodes.parquet") == info["episodes_sha256"]
    for rel, expected in info["source_code_sha256"].items():
        assert digest(out / "source" / rel) == expected
    master, episodes = pd.read_parquet(out / "corrected_master.parquet"), pd.read_parquet(out / "episodes.parquet")
    assert not master.duplicated(["coin", "hour"]).any()
    horizons = tuple(info.get("horizons", (1, 6, 24)))
    test_years = tuple(info.get("test_years", range(2021, 2026)))
    sample_ids = {f"{c}_h{h}_{y}" for c in ("market_original", "market_matched", "market_onchain")
                  for h in horizons for y in test_years}
    assert {p.name for p in (out / "samples").iterdir()} == sample_ids
    assert {p.name for p in (out / "fits").iterdir()} == {
        f"{model}_{sample}" for model in ("constant_prior", "persistence") for sample in sample_ids}
    sample_count = check_samples(out, master, episodes, info.get("purge_hours", 24),
                                info.get("embargo_hours", 24), horizons, test_years)
    years, summary = pd.read_csv(out / "per_year.csv"), pd.read_csv(out / "summary.csv")
    assert len(years) == 6 * len(horizons) * len(test_years)
    assert len(summary) == 6 * len(horizons)
    records, pooled_records, oracle_rows = [], [], []
    for folder in sorted((out / "fits").iterdir()):
        fit = json.loads((folder / "fit.json").read_text())
        assert fit["run_id"] == info["run_id"]
        train = pd.read_parquet(out / fit["sample"] / "inner_train.parquet")
        assert len(train) == fit["train_rows"] and int(train.y_true.sum()) == fit["train_positive_hours"]
        prior = float(train.y_true.mean())
        if fit["model"] == "constant_prior":
            assert fit["model_params"]["positive_rate"] == prior and fit["score_inputs"] == []
        else:
            assert fit["score_inputs"] == ["peg_deviation"]
        parts = {}
        for part in ("inner_val", "test"):
            pred = pd.read_parquet(folder / f"{part}.parquet")
            columns = KEYS + ["price", "peg_deviation"]
            pd.testing.assert_frame_equal(
                pred[columns], pd.read_parquet(out / fit["sample"] / f"{part}.parquet")[columns])
            expected = np.full(len(pred), prior) if fit["model"] == "constant_prior" else np.minimum(
                np.maximum(1 - pred.price.to_numpy(), 0), .01)
            np.testing.assert_allclose(pred.y_score, expected, atol=0, rtol=0)
            assert pred.threshold.eq(fit["threshold"]).all()
            assert pred.run_id.eq(info["run_id"]).all() and pred.fit_id.eq(fit["fit_id"]).all()
            parts[part] = pred
        val, test = parts["inner_val"], parts["test"]
        ordered, times, coins, targets, eps = target_codes(val, episodes, fit["horizon"])
        cutoffs = np.r_[np.unique(ordered.y_score), np.inf]
        counts = full_scan_candidates(ordered.y_score.to_numpy(), times, coins, targets, cutoffs,
                                      fit["horizon"] * HOUR, len(eps))
        exposure = len(set(zip(val.coin, val.hour.dt.strftime("%Y-%m"))))
        eligible = np.flatnonzero(counts[:, 1] <= exposure)
        best = max(eligible, key=lambda i: (counts[i, 0], -counts[i, 1], cutoffs[i]))
        assert cutoffs[best] == fit["threshold"]
        assert len(cutoffs) == fit["validation"]["candidate_count"]
        ve, vt = independent_events(val, episodes, fit["horizon"])
        vm = independent_metrics(val, ve, vt)
        assert vm["false_alarms"] == fit["validation"]["false_alarms"]
        assert vm["observed_coin_months"] == fit["validation"]["coin_months"]
        assert vm["false_alarms_per_coin_month"] <= 1
        assert vm["episodes_caught"] == fit["validation"]["episodes_warned"]
        assert len(vt) == fit["validation"]["episodes_targetable"]
        assert vm["episode_recall"] == fit["validation"]["episode_recall"]
        if fit["model"] == "constant_prior":
            np.testing.assert_allclose(vm["average_precision"], val.y_true.mean(), atol=1e-12, rtol=0)
        for i, cutoff in enumerate(cutoffs):
            oracle_rows.append(dict(run_id=info["run_id"], fit_id=fit["fit_id"], threshold=cutoff,
                                    caught=int(counts[i, 0]), false_alarms=int(counts[i, 1]),
                                    alarm_events=int(counts[i, 2]), observed_coin_months=exposure,
                                    feasible=bool(counts[i, 1] <= exposure), selected=i == best))
        events, ids = independent_events(test, episodes, fit["horizon"])
        assert_events(events, folder / "alarms.csv")
        actual = independent_metrics(test, events, ids)
        expected_row = years.loc[years.fit_id.eq(fit["fit_id"])]
        assert len(expected_row) == 1
        assert_metrics(actual, expected_row.iloc[0])
        np.testing.assert_allclose(vm["average_precision"], expected_row.iloc[0].validation_average_precision,
                                   atol=1e-12, rtol=0)
        records.append(dict(run_id=info["run_id"], fit_id=fit["fit_id"], candidates_checked=len(cutoffs),
                            scores_exact=True, partitions_exact=True, threshold_exact=True, metrics_exact=True))
        print(f"PASS {fit['fit_id']}: {len(cutoffs)} full-scan candidates", flush=True)
    for r in summary.to_dict("records"):
        tag = f"{r['model']}_{r['config']}_h{r['horizon_hours']}"
        pred = pd.read_parquet(out / f"predictions_{tag}.parquet")
        parts = [pd.read_parquet(out / "fits" / f"{tag}_{y}" / "test.parquet") for y in test_years]
        pd.testing.assert_frame_equal(pred, pd.concat(parts, ignore_index=True))
        events, ids = independent_events(pred, episodes, r["horizon_hours"])
        assert_events(events, out / f"alarms_{tag}.csv")
        actual = independent_metrics(pred, events, ids)
        assert_metrics(actual, r)
        saved_ids = pd.read_csv(out / "caught_identities.csv")
        saved_ids = saved_ids.loc[saved_ids.model.eq(r["model"]) & saved_ids.config.eq(r["config"]) &
                                  saved_ids.horizon_hours.eq(r["horizon_hours"])]
        assert set(saved_ids.episode_id) == ids
        for identity in saved_ids.itertuples():
            leads = events.loc[events.target_episode_id.eq(identity.episode_id), "lead_hours"]
            assert identity.caught == (not leads.empty)
            np.testing.assert_allclose(identity.earliest_alarm_lead_hours, leads.max(),
                                       atol=1e-12, rtol=0, equal_nan=True)
        pooled_records.append(dict(run_id=info["run_id"], model=r["model"], config=r["config"],
                                   horizon_hours=r["horizon_hours"], **actual))
    # Same inputs, labels and thresholds must give identical matched/on-chain baseline predictions.
    for name in ("constant_prior", "persistence"):
        for h in horizons:
            a, b = [pd.read_parquet(out / f"predictions_{name}_{c}_h{h}.parquet")
                    for c in ("market_matched", "market_onchain")]
            pd.testing.assert_frame_equal(a[KEYS + ["y_score", "threshold"]], b[KEYS + ["y_score", "threshold"]])
    comparisons = compatibility(out, references, info, save=not check_only)
    if check_only:
        print(f"PASS READ-ONLY: {len(records)} fits; {sample_count} partitions; "
              f"{len(oracle_rows)} full-scan candidates; {len(summary)} pooled rows; compatibility inspected", flush=True)
        return
    pd.DataFrame(records).to_csv(out / "verification_fits.csv", index=False)
    pd.DataFrame(pooled_records).to_csv(out / "verification_pooled.csv", index=False)
    pd.DataFrame(oracle_rows).to_csv(out / "verification_threshold_candidates.csv", index=False)
    slides = summary.loc[summary.config.isin(["market_original", "market_onchain"])].copy()
    slides["slide"] = np.where(slides.config.eq("market_original"), 19, 23)
    slides["tier"] = np.where(slides.config.eq("market_original"), 1, 2)
    slides.to_csv(out / "replacement_slides_19_23.csv", index=False)
    verifier = out / "verification_source" / Path(__file__).name
    verifier.parent.mkdir(exist_ok=False)
    verifier.write_bytes(Path(__file__).read_bytes())
    write_json(out / "verification.json", dict(
        run_id=info["run_id"], status="passed", fits_checked=len(records),
        sample_partitions_checked=sample_count, pooled_rows_checked=len(pooled_records),
        validation_candidates_checked=len(oracle_rows), independent_ap=True,
        independent_grouped_alarms=True, independent_full_scan_threshold_selection=True,
        executed_verifier_sha256=digest(verifier), executed_verifier_path=str(verifier.relative_to(out)),
        compatibility=comparisons,
    ))
    artifacts = [p for p in out.rglob("*") if p.is_file() and p.name != "artifact_sha256.json"]
    write_json(out / "artifact_sha256.json", {str(p.relative_to(out)): digest(p) for p in artifacts})
    print(f"PASS: {len(records)} fits; {sample_count} partitions; "
          f"{len(oracle_rows)} full-scan candidates; {len(summary)} pooled rows; compatibility inspected", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, action="append", default=[])
    parser.add_argument("--check-only", action="store_true",
                        help="repeat every check without writing or replacing evidence")
    args = parser.parse_args()
    verify(args.run_dir.resolve(), [p.resolve() for p in args.reference], check_only=args.check_only)
