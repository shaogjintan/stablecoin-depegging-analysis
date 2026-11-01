"""Corrected fixed-round XGB run. From repo root:

python -m modelling.corrected_xgb --out-dir <new immutable directory>
python -m modelling.corrected_xgb --shap-only --out-dir <completed directory>

Legacy reports/results remain untouched. All clocks denote bar closes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import shutil
import uuid

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from modelling.alarm_events import build_alarm_events
from modelling.corrected_data import prepare_corrected, corrected_folds
from modelling.false_alarm_framework import _common_sample_fold
from modelling.models.xgboost_model import XGBModel
from modelling.prepare_data import MARKET_FEATURES, ONCHAIN_FEATURES
from modelling.rolling_threshold import (
    _choose_threshold, _targetable_episode_ids, _observed_coin_months, _false_alarm_stats,
)

ROOT = Path(__file__).resolve().parents[1]
PARAMS = dict(max_depth=3, learning_rate=.05, n_estimators=300, n_jobs=4,
              min_child_weight=1, scale_pos_weight="auto", random_state=42)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def score_and_save(fold, config, out, run_id, episodes):
    year, h = int(fold["year"]), int(fold["horizon"])
    fit_id = f"{config}_h{h}_{year}"
    folder = out / "fits" / fit_id
    folder.mkdir(parents=True, exist_ok=False)
    model = XGBModel(**PARAMS)
    model.fit(fold["X_inner_train"], fold["y_inner_train"])
    assert model.clf_.get_booster().num_boosted_rounds() == 300
    joblib.dump(model, folder / "model.joblib")
    model.clf_.save_model(folder / "model.ubj")
    train = pd.concat([fold["inner_train_meta"], fold["X_inner_train"],
                       fold["y_inner_train"].rename("y_true")], axis=1)
    train.to_parquet(folder / "inner_train.parquet", index=False)
    frames = {}
    for part in ("inner_val", "test"):
        X = fold[f"X_{part}"]
        scores = model.predict_proba(X)[:, 1]
        restored = joblib.load(folder / "model.joblib")
        np.testing.assert_allclose(restored.predict_proba(X)[:, 1], scores, atol=1e-8, rtol=0)
        frame = fold[f"{part}_meta"].copy()
        frame["horizon"], frame["y_true"], frame["y_score"] = h, fold[f"y_{part}"], scores
        frame["run_id"], frame["fit_id"] = run_id, fit_id
        persisted = pd.concat([frame, X], axis=1)
        persisted.to_parquet(folder / f"{part}.parquet", index=False)
        frames[part] = frame
    val, test = frames["inner_val"], frames["test"]
    threshold, vs = _choose_threshold(val, episodes, h, 1.)
    _, events = build_alarm_events(test, episodes, threshold)
    targets = _targetable_episode_ids(test, episodes, h)
    caught = set(events.target_episode_id.dropna().astype(str)) & targets
    false, months, rate, maximum = _false_alarm_stats(events, test)
    events["fit_id"], events["run_id"], events["config"] = fit_id, run_id, config
    events["test_year"] = year
    events.to_csv(folder / "alarms.csv", index=False)
    save_json(folder / "fit.json", {
        "run_id": run_id, "fit_id": fit_id, "config": config, "horizon": h, "year": year,
        "features": fold["features"], "preprocessing": "identity; corrected calendar features; finite complete rows",
        "fit_partition": "inner_train", "calibration_partition": "inner_val",
        "params": PARAMS, "scale_pos_weight_used": model.scale_pos_weight_used_,
        "boosted_rounds": 300, "early_stopping": False, "threshold": threshold,
        "validation": vs, "train_rows": len(train), "train_positive_hours": int(train.y_true.sum()),
        "train_last_origin": str(train.hour.max()), "validation_first_origin": str(val.hour.min()),
        "validation_last_origin": str(val.hour.max()), "test_first_origin": str(test.hour.min()),
        "model_sha256": sha256(folder / "model.joblib"), "score_reload_max_tolerance": 1e-8,
    })
    row = dict(config=config, horizon_hours=h, test_year=year, threshold=threshold,
               validation_rows=len(val), validation_positive_hours=int(val.y_true.sum()),
               validation_average_precision=average_precision_score(val.y_true, val.y_score),
               validation_false_alarms=vs["false_alarms"], validation_coin_months=vs["coin_months"],
               validation_recall=vs["episode_recall"], threshold_candidates=vs["candidate_count"],
               test_rows=len(test), positive_hours=int(test.y_true.sum()),
               positive_rate=float(test.y_true.mean()),
               average_precision=average_precision_score(test.y_true, test.y_score),
               targetable_episodes=len(targets), episodes_caught=len(caught),
               episode_recall=len(caught) / len(targets) if targets else np.nan,
               alarm_events=len(events), true_alarms=int(events.is_true_alarm.sum()),
               false_alarms=false, observed_coin_months=months, false_alarms_per_coin_month=rate,
               max_false_alarms_in_coin_month=maximum,
               median_lead_hours=float(events.loc[events.is_true_alarm, "lead_hours"].median()),
               run_id=run_id)
    print(f"{fit_id}: AP={row['average_precision']:.6f}, catches={len(caught)}/{len(targets)}, "
          f"FA={false}/{months}, candidates={vs['candidate_count']}", flush=True)
    return row, test, events, targets


def paired_uncertainty(pred1, pred2, identities, h):
    keys = ["coin", "hour", "y_true"]
    p = pred1[keys + ["y_score"]].merge(pred2[keys + ["y_score"]], on=keys,
                                      suffixes=("_market", "_onchain"), validate="one_to_one")
    assert len(p) == len(pred1) == len(pred2)
    blocks = (p.coin + ":" + p.hour.dt.strftime("%Y-%m")).to_numpy()
    groups = [np.flatnonzero(blocks == b) for b in np.unique(blocks)]
    rng = np.random.default_rng(42 + h)
    ap_differences = []
    for _ in range(500):
        idx = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
        y = p.y_true.to_numpy()[idx]
        if y.sum():
            ap_differences.append(average_precision_score(y, p.y_score_onchain.to_numpy()[idx])
                                  - average_precision_score(y, p.y_score_market.to_numpy()[idx]))
    d = identities[identities.horizon_hours.eq(h)]
    market = d[d.config.eq("market_matched")].set_index("episode_id")
    onchain = d[d.config.eq("market_onchain")].set_index("episode_id")
    assert set(market.index) == set(onchain.index)
    delta = onchain.loc[market.index, "caught"].astype(int) - market.caught.astype(int)
    ordered = market.sort_values("episode_start")
    event_groups, last = [], None
    for ep, row in ordered.iterrows():
        if last is None or row.episode_start - last > pd.Timedelta(hours=24):
            event_groups.append([])
        event_groups[-1].append(ep)
        last = row.episode_start
    recall_differences = []
    for _ in range(5000):
        sample = [ep for j in rng.integers(0, len(event_groups), len(event_groups))
                  for ep in event_groups[j]]
        recall_differences.append(float(delta.loc[sample].mean()))
    return dict(horizon_hours=h, comparison="onchain minus matched market",
                average_precision_difference=average_precision_score(p.y_true, p.y_score_onchain)
                - average_precision_score(p.y_true, p.y_score_market),
                ap_ci_low=np.quantile(ap_differences, .025), ap_ci_high=np.quantile(ap_differences, .975),
                ap_bootstrap="500 paired observed coin-month resamples; conditional saved fits",
                episode_recall_difference=float(delta.mean()),
                recall_ci_low=np.quantile(recall_differences, .025),
                recall_ci_high=np.quantile(recall_differences, .975),
                recall_bootstrap="5000 paired 24h-chained onset-event cluster resamples; conditional saved fits",
                targetable_episodes=len(delta), targetable_event_clusters=len(event_groups))


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    source = ROOT / "data" / "processed" / "master.parquet"
    metadata = {
        "run_id": run_id, "status": "running", "source_commit":
        subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "source_master_sha256": sha256(source), "params": PARAMS,
        "clock": "hour=source bar_start+1h; first-below-bar close onset; lead=start_close-origin_close",
        "target": "new merged 2h dip episode starts in (origin_close, origin_close+h]",
        "risk": "observed above-band close, >24h after recovery anchor of last online-confirmed 2h dip",
        "threshold": "all distinct validation scores and +inf; max episode recall within FA/observed coin-month<=1, then fewer FA, then higher threshold",
        "packages": {p: importlib.metadata.version(p) for p in
                     ("numpy", "pandas", "scikit-learn", "xgboost", "shap", "numba", "joblib")},
        "source_code_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in
                              [ROOT / "modelling" / n for n in
                               ("corrected_data.py", "corrected_xgb.py", "rolling_threshold.py", "exact_threshold.py", "alarm_events.py")]},
        "limitations": [
            "Original source gaps/forward filling and original venue flags cannot be recovered; no zeros fabricated.",
            "End-of-bar idealized availability; actual exchange/Dune publication delays are not known.",
            "Episode onset is first-below-bar close, retrospectively confirmed by second bar; not onset-confirmation time.",
            "Negative labels require h+1 future bars; observed-label sample is an offline evaluation filter only.",
            "Observed coin-months count partial months, not equivalent full exposure months.",
            "Matched intervals condition on fits/thresholds and are exploratory, not a retrained model significance test.",
            "No h1, LR/RF repairs, optional smoothing/OR/sensitivity or live deployment validation.",
        ],
    }
    save_json(out / "run.json", metadata)
    source_snapshot = out / "source"
    source_snapshot.mkdir()
    for rel in metadata["source_code_sha256"]:
        shutil.copy2(ROOT / rel, source_snapshot / Path(rel).name)
    (out / "source_diff.patch").write_bytes(
        subprocess.check_output(["git", "-C", str(ROOT), "diff", "--binary"]))
    print("Preparing corrected calendar features, causal eligibility and merged-start labels", flush=True)
    master, episodes = prepare_corrected(pd.read_parquet(source))
    master.to_parquet(out / "corrected_master.parquet", index=False)
    episodes.to_parquet(out / "episodes.parquet", index=False)
    episodes.to_csv(out / "episodes.csv", index=False)
    year_rows, summary, identity_rows, pooled = [], [], [], {}
    for h in (6, 24):
        f1, f2 = corrected_folds(master, episodes, h, 1), corrected_folds(master, episodes, h, 2)
        configs = {"market_original": f1, "market_matched": [_common_sample_fold(a, b) for a, b in zip(f1, f2)],
                   "market_onchain": f2}
        for config, folds in configs.items():
            predictions, alarms, target_ids = [], [], set()
            for fold in folds:
                row, pred, events, targets = score_and_save(fold, config, out, run_id, episodes)
                year_rows.append(row)
                predictions.append(pred)
                alarms.append(events)
                target_ids |= targets
            pred, events = pd.concat(predictions, ignore_index=True), pd.concat(alarms, ignore_index=True)
            pred.to_parquet(out / f"predictions_{config}_h{h}.parquet", index=False)
            events.to_csv(out / f"alarms_{config}_h{h}.csv", index=False)
            caught = set(events.target_episode_id.dropna().astype(str)) & target_ids
            fa, months, rate, maximum = _false_alarm_stats(events, pred)
            summary.append(dict(run_id=run_id, config=config, horizon_hours=h,
                                rows=len(pred), positive_hours=int(pred.y_true.sum()),
                                positive_rate=float(pred.y_true.mean()),
                                average_precision=average_precision_score(pred.y_true, pred.y_score),
                                targetable_episodes=len(target_ids), episodes_caught=len(caught),
                                episode_recall=len(caught)/len(target_ids) if target_ids else np.nan,
                                alarm_events=len(events), true_alarms=int(events.is_true_alarm.sum()),
                                false_alarms=fa, observed_coin_months=months,
                                false_alarms_per_coin_month=rate, max_false_alarms_in_coin_month=maximum,
                                median_lead_hours=float(events.loc[events.is_true_alarm, "lead_hours"].median())))
            for ep in sorted(target_ids):
                ep_row = episodes[episodes.episode_id.eq(ep)].iloc[0]
                leads = events.loc[events.target_episode_id.eq(ep), "lead_hours"]
                identity_rows.append(dict(run_id=run_id, config=config, horizon_hours=h, episode_id=ep,
                                          coin=ep_row.coin, episode_start=ep_row.start,
                                          caught=ep in caught, earliest_alarm_lead_hours=leads.max()))
            pooled[(config, h)] = pred
    pd.DataFrame(year_rows).to_csv(out / "per_year.csv", index=False)
    summary = pd.DataFrame(summary)
    summary.to_csv(out / "summary.csv", index=False)
    identities = pd.DataFrame(identity_rows)
    identities.to_csv(out / "caught_identities.csv", index=False)
    print("Computing matched conditional block-bootstrap uncertainty", flush=True)
    uncertainty = [paired_uncertainty(pooled[("market_matched", h)], pooled[("market_onchain", h)],
                                     identities, h) for h in (6, 24)]
    pd.DataFrame(uncertainty).to_csv(out / "matched_uncertainty.csv", index=False)
    metadata["status"] = "scoring_complete"
    metadata["corrected_master_sha256"] = sha256(out / "corrected_master.parquet")
    save_json(out / "run.json", metadata)
    run_shap(out)
    metadata["status"] = "complete"
    save_json(out / "run.json", metadata)
    print(summary.to_string(index=False), flush=True)


def run_shap(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    rows = []
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    for h in (6, 24):
        for config, tier in (("market_original", 1), ("market_onchain", 2)):
            tag = f"xgboost_t{tier}_h{h}"
            parts, features, values, keys, checks = [], None, [], [], []
            for folder in sorted((out / "fits").glob(f"{config}_h{h}_*")):
                info = json.loads((folder / "fit.json").read_text())
                model = joblib.load(folder / "model.joblib")
                assert sha256(folder / "model.joblib") == info["model_sha256"]
                test = pd.read_parquet(folder / "test.parquet")
                features = info["features"]
                X = test[features]
                np.testing.assert_allclose(model.predict_proba(X)[:, 1], test.y_score, atol=1e-8, rtol=0)
                explainer = shap.TreeExplainer(model.clf_, feature_perturbation="tree_path_dependent",
                                               model_output="raw")
                sv = np.asarray(explainer.shap_values(X, check_additivity=True), dtype=float)
                margin = model.clf_.predict(X, output_margin=True)
                error = float(np.max(np.abs(sv.sum(axis=1) + explainer.expected_value - margin)))
                assert error <= 1e-4, error
                checks.append(dict(fit_id=info["fit_id"], explained_rows=len(X),
                                   max_additivity_error=error, tolerance=1e-4,
                                   expected_margin=float(explainer.expected_value),
                                   background="implicit tree-path inner_train cover counts; no test background"))
                parts.append(sv)
                values.append(X)
                keys.append(test[["coin", "hour", "run_id", "fit_id"]])
                print(f"SHAP {info['fit_id']}: {len(X)} exact test rows, max error={error:.8g}", flush=True)
            SV = np.vstack(parts)
            X = pd.concat(values, ignore_index=True)
            np.savez_compressed(out / f"shap_values_{tag}.npz", values=SV, features=features)
            pd.concat(keys, ignore_index=True).to_parquet(out / f"shap_rows_{tag}.parquet", index=False)
            mean_abs = np.abs(SV).mean(axis=0)
            ranking = pd.DataFrame(dict(feature=features, mean_abs_shap=mean_abs,
                                        share_of_total_abs=mean_abs / mean_abs.sum())).sort_values("mean_abs_shap", ascending=False)
            ranking.to_csv(out / f"shap_ranking_{tag}.csv", index=False)
            shares = {}
            for group, names in (("market", MARKET_FEATURES), ("onchain", ONCHAIN_FEATURES)):
                idx = [features.index(f) for f in names if f in features]
                shares[group] = float(np.abs(SV[:, idx]).sum() / np.abs(SV).sum())
                independent = float(ranking.loc[ranking.feature.isin(names), "share_of_total_abs"].sum())
                assert abs(independent - shares[group]) < 1e-10
            assert abs(sum(shares.values()) - 1) < 1e-10
            pd.DataFrame([shares]).to_csv(out / f"group_share_{tag}.csv", index=False)
            pd.DataFrame(checks).to_csv(out / f"shap_checks_{tag}.csv", index=False)
            rows.append(dict(config=config, tier=tier, horizon_hours=h, explained_rows=len(X),
                             market_share=shares["market"], onchain_share=shares["onchain"],
                             max_additivity_error=max(c["max_additivity_error"] for c in checks)))
            fig, ax = plt.subplots(figsize=(8, 6))
            top = ranking.head(15).iloc[::-1]
            ax.barh(top.feature, top.mean_abs_shap)
            ax.set_xlabel("Mean absolute SHAP (raw log-odds margin; all OOS rows)")
            ax.set_title(f"Corrected fixed300 {tag}")
            fig.tight_layout()
            fig.savefig(figures / f"shap_ranking_{tag}.png", dpi=160)
            plt.close(fig)
            rng = np.random.default_rng(42)
            idx = rng.choice(len(X), min(15000, len(X)), replace=False)
            shap.summary_plot(SV[idx], X.iloc[idx], show=False, max_display=22)
            plt.gcf().savefig(figures / f"beeswarm_{tag}.png", dpi=160, bbox_inches="tight")
            plt.close()
            fig, ax = plt.subplots(figsize=(7, 2.5))
            ax.barh(["SHAP magnitude"], [shares["market"]], label=f"market {shares['market']:.2%}")
            ax.barh(["SHAP magnitude"], [shares["onchain"]], left=[shares["market"]],
                    label=f"on-chain {shares['onchain']:.2%}")
            ax.set_xlim(0, 1)
            ax.set_title(f"Corrected {tag} — attribution, not causal benefit")
            ax.legend()
            fig.tight_layout()
            fig.savefig(figures / f"group_share_{tag}.png", dpi=160)
            plt.close(fig)
    pd.DataFrame(rows).to_csv(out / "shap_summary.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--shap-only", action="store_true")
    args = parser.parse_args()
    if args.shap_only:
        run_shap(args.out_dir.resolve())
    else:
        run(args.out_dir.resolve())


if __name__ == "__main__":
    main()
