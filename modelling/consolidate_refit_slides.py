"""Refresh the nine-slide value sheet from completed refit-after-calibration runs.

python -m modelling.consolidate_refit_slides --run-dir runs/REFIT-LOWEST-NEW

Uses saved final forests, full out-of-sample TreeSHAP, and freshly refitted
benchmarks. Incomplete or incompatible LogReg runs are explicitly pending.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from modelling.alarm_events import build_alarm_events
from modelling.consolidate_slide_values import table, nfmt
from modelling.corrected_xgb import sha256, save_json
from modelling.models import get_model
from modelling.prepare_data import ONCHAIN_FEATURES
from modelling.rolling_threshold import _choose_threshold, _targetable_episode_ids, _false_alarm_stats


def require_run(path):
    info = json.loads((path / "run.json").read_text())
    if info.get("status") != "complete" or not info.get("refit_after_calibration"):
        raise ValueError(f"Not a completed refit run: {path}")
    if info.get("policy") != "exhaustive_lowest":
        raise ValueError(f"Wrong threshold policy: {path}")
    return info


def calibration_audit_rows(run, models):
    rows = []
    for h in [6, 24]:
        for year in range(2021, 2026):
            for config in ["market_matched", "market_onchain"]:
                for model in models:
                    folder = run / "fits" / f"{model}_{config}_h{h}_{year}"
                    spec = json.loads((folder / "fit.json").read_text())
                    stats = spec["validation"]
                    rows.append(dict(model=model, config=config, horizon_hours=h, test_year=year,
                                     calibration_year=year-1, threshold=spec["threshold"],
                                     calibration_coin_months=stats["coin_months"],
                                     calibration_false_alarms=stats["false_alarms"],
                                     calibration_budget=stats["coin_months"],
                                     final_train_rows=spec["final_train_rows"]))
    return rows


def shap_values(run, horizons):
    """Aggregate exact absolute raw-margin SHAP over every held-out test row."""
    import shap
    summaries = []
    for h in horizons:
        sums, count, checks = None, 0, []
        for year in range(2021, 2026):
            folder = run / "fits" / f"xgboost_market_onchain_h{h}_{year}"
            info = json.loads((folder / "fit.json").read_text())
            if sha256(folder / "pipeline.joblib") != info["pipeline_sha256"]:
                raise ValueError(f"Changed final pipeline: {folder}")
            bundle = joblib.load(folder / "pipeline.joblib")
            model = bundle["model"]
            test = pd.read_parquet(folder / "test.parquet")
            features = info["features"]
            X = test[features]
            explainer = shap.TreeExplainer(model.clf_, feature_perturbation="tree_path_dependent",
                                           model_output="raw")
            error = 0.
            for start in range(0, len(X), 8192):
                batch = X.iloc[start:start + 8192]
                sv = np.asarray(explainer.shap_values(batch, check_additivity=True), dtype=float)
                margin = model.clf_.predict(batch, output_margin=True)
                error = max(error, float(np.max(np.abs(sv.sum(axis=1) + explainer.expected_value - margin))))
                total = np.abs(sv).sum(axis=0)
                sums = total if sums is None else sums + total
                count += len(batch)
            if error > 1e-4:
                raise ValueError(f"SHAP additivity error {error}: {folder}")
            checks.append(dict(test_year=year, rows=len(X), max_additivity_error=error,
                               final_pipeline_sha256=info["pipeline_sha256"]))
            print(f"SHAP refit h{h} {year}: {len(X)} rows", flush=True)
        ranking = pd.DataFrame(dict(feature=features, mean_abs_shap=sums / count,
                                    share_of_total_abs=sums / sums.sum())).sort_values(
                                        "mean_abs_shap", ascending=False)
        ranking.to_csv(run / f"shap_ranking_xgboost_t2_h{h}.csv", index=False)
        pd.DataFrame(checks).to_csv(run / f"shap_checks_xgboost_t2_h{h}.csv", index=False)
        onchain = float(ranking.loc[ranking.feature.isin(ONCHAIN_FEATURES), "share_of_total_abs"].sum())
        summaries.append(dict(tier=2, horizon_hours=h, explained_rows=count,
                              onchain_share=onchain, market_share=1-onchain,
                              background="final full-training tree-path cover counts; no test background",
                              model_output="raw", max_additivity_error=max(c["max_additivity_error"] for c in checks)))
    result = pd.DataFrame(summaries)
    result.to_csv(run / "shap_summary.csv", index=False)
    return result


def benchmarks(run, episodes):
    rows = []
    for h in [6, 24]:
        for name in ["baseline", "persistence"]:
            predictions, alarms_all, counts = [], [], []
            for year in range(2021, 2026):
                folder = run / "fits" / f"xgboost_market_onchain_h{h}_{year}"
                info = json.loads((folder / "fit.json").read_text())
                features = info["features"]
                practice_train, train, val, test = [pd.read_parquet(folder / f"{p}.parquet")
                                                   for p in ["inner_train", "train", "inner_val", "test"]]
                practice = get_model(name).fit(practice_train[features], practice_train.y_true)
                val["y_score"] = practice.predict_proba(val[features])[:, 1]
                cutoff, stats = _choose_threshold(val, episodes, h, 1., policy="exhaustive_lowest")
                final = get_model(name).fit(train[features], train.y_true)
                test["y_score"] = final.predict_proba(test[features])[:, 1]
                _, alarms = build_alarm_events(test, episodes, cutoff)
                targets = _targetable_episode_ids(test, episodes, h)
                caught = len(set(alarms.target_episode_id.dropna().astype(str)) & targets)
                false, months, _, _ = _false_alarm_stats(alarms, test)
                counts.append((caught, len(targets), false, months))
                predictions.append(test[["y_true", "y_score"]])
                alarms_all.append(alarms)
            pred = pd.concat(predictions, ignore_index=True)
            alarms = pd.concat(alarms_all, ignore_index=True)
            totals = np.array(counts).sum(axis=0)
            rows.append(dict(model=name, horizon_hours=h, average_precision=float(
                average_precision_score(pred.y_true, pred.y_score)), positive_rate=float(pred.y_true.mean()),
                episodes_caught=int(totals[0]), targetable_episodes=int(totals[1]),
                episode_recall=float(totals[0]/totals[1]), false_alarms=int(totals[2]),
                coin_months=int(totals[3]), false_alarms_per_coin_month=float(totals[2]/totals[3]),
                median_lead_hours=float(alarms.loc[alarms.is_true_alarm, "lead_hours"].median())))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--lr-run", type=Path, default=Path("runs/LR-REFIT-LOWEST-NEW"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/presentation_consolidated"))
    parser.add_argument("--reuse-shap", action="store_true", help="reuse SHAP generated by this script for this run")
    args = parser.parse_args()
    run, out = args.run_dir, args.out_dir
    info = require_run(run)
    summaries = pd.read_csv(run / "summary.csv")
    episodes = pd.read_parquet(run / "episodes.parquet")
    audit = calibration_audit_rows(run, ["random_forest", "xgboost"])
    lr_ready, lr_status = False, "missing"
    if (args.lr_run / "run.json").exists():
        lr_info = json.loads((args.lr_run / "run.json").read_text())
        lr_status = lr_info.get("status", "unknown")
        if lr_status == "complete":
            require_run(args.lr_run)
            if lr_info["input_sha256"] != info["input_sha256"]:
                raise ValueError("LogReg uses different dataset inputs")
            summaries = pd.concat([summaries, pd.read_csv(args.lr_run / "summary.csv")], ignore_index=True)
            audit.extend(calibration_audit_rows(args.lr_run, ["logreg"]))
            lr_ready = True
    shap = (pd.read_csv(run / "shap_summary.csv") if args.reuse_shap
            else shap_values(run, [6, 24]))
    benchmark = benchmarks(run, episodes)
    out.mkdir(parents=True, exist_ok=True)
    audit = pd.DataFrame(audit)
    audit.to_csv(out / "calibration_audit.csv", index=False)
    benchmark.to_csv(out / "benchmark_comparison.csv", index=False)
    summaries.to_csv(out / "model_comparison_summary.csv", index=False)
    yearly = pd.read_csv(run / "by_year.csv")
    if lr_ready:
        yearly = pd.concat([yearly, pd.read_csv(args.lr_run / "by_year.csv")], ignore_index=True)
    yearly.to_csv(out / "model_comparison_by_year.csv", index=False)
    prior_path = Path("runs/EXHAUSTIVE-LOWEST-NEW/summary.csv")
    if prior_path.exists():
        prior = pd.read_csv(prior_path)
        comparison = summaries.merge(prior, on=["model", "config", "horizon_hours", "policy"],
                                     suffixes=("_refit", "_no_refit"))
        for metric in ["episodes_caught", "false_alarms", "average_precision", "median_lead_hours"]:
            comparison[f"delta_{metric}"] = comparison[f"{metric}_refit"] - comparison[f"{metric}_no_refit"]
        comparison.to_csv(out / "refit_vs_no_refit.csv", index=False)
    importance = []
    for h in [6, 24]:
        for year in range(2021, 2026):
            folder = run / "fits" / f"random_forest_market_onchain_h{h}_{year}"
            spec = json.loads((folder / "fit.json").read_text())
            model = joblib.load(folder / "pipeline.joblib")["model"]
            importance.extend(dict(horizon_hours=h, test_year=year, feature=f, importance=float(v))
                              for f, v in zip(spec["features"], model.feature_importances_))
    imp = pd.DataFrame(importance)
    imp.to_csv(out / "rf_importance_by_year.csv", index=False)
    imp = imp.groupby(["horizon_hours", "feature"], as_index=False).importance.mean()
    main_features = ["realised_vol_24h", "peg_stress", "peg_deviation", "realised_vol_6h"]
    rf_rows = []
    for feature in main_features + ["other_market", "onchain"]:
        values = []
        for h in [6, 24]:
            d = imp.loc[imp.horizon_hours.eq(h)]
            chain = d.feature.isin(ONCHAIN_FEATURES)
            mask = chain if feature == "onchain" else (~chain & ~d.feature.isin(main_features)) if feature == "other_market" else d.feature.eq(feature)
            values.append(f"{d.loc[mask, 'importance'].sum()*100:.1f}%")
        rf_rows.append([feature, *values])
    chain_share = dict(zip(shap.horizon_hours, shap.onchain_share))
    ranking = pd.read_csv(run / "shap_ranking_xgboost_t2_h24.csv")
    selected = pd.concat([ranking[~ranking.feature.isin(ONCHAIN_FEATURES)].head(3),
                          ranking[ranking.feature.isin(ONCHAIN_FEATURES)].head(3)])

    headers = ["h", "Model", "Pooled AP", "Positive base rate (%)", "False alarms",
               "FA / coin-month", "Episode recall", "Median lead (hours)"]

    def values(r):
        return [int(r.horizon_hours), r.model, f"{r.average_precision:.6f}", f"{r.positive_rate*100:.4f}",
                int(r.false_alarms), f"{r.false_alarms_per_coin_month:.3f}",
                f"{int(r.episodes_caught)}/{int(r.targetable_episodes)} ({r.episode_recall*100:.2f}%)",
                nfmt(r.median_lead_hours)]

    def performance(config):
        rows = []
        for h in [6, 24]:
            for model in ["logreg", "random_forest", "xgboost"]:
                d = summaries.loc[summaries.config.eq(config) & summaries.horizon_hours.eq(h) & summaries.model.eq(model)]
                rows.append(values(d.iloc[0]) if len(d) else [h, model, *(["PENDING"]*6)])
        return table(headers, rows)

    def findings(config):
        parts = []
        for h in [6, 24]:
            d = summaries.loc[summaries.config.eq(config) & summaries.horizon_hours.eq(h)]
            best = d.loc[d.episodes_caught.eq(d.episodes_caught.max())]
            parts.append(f"h{h}: most episodes caught by {' / '.join(best.model)} "
                         f"({int(best.iloc[0].episodes_caught)}/{int(best.iloc[0].targetable_episodes)}).")
        return " ".join(parts) + (" LogReg remains pending; this ranking covers completed RF/XGB only." if not lr_ready else "")

    def onchain_findings():
        rows = []
        for model in summaries.model.unique():
            for h in [6, 24]:
                d = summaries.loc[summaries.model.eq(model) & summaries.horizon_hours.eq(h)].set_index("config")
                a, b = d.loc["market_matched"], d.loc["market_onchain"]
                rows.append([model, h, f"{int(a.episodes_caught)} → {int(b.episodes_caught)}",
                             f"{int(a.false_alarms)} → {int(b.false_alarms)}",
                             f"{a.average_precision:.6f} → {b.average_precision:.6f}"])
        return table(["Model", "h", "Episodes caught T1 → T2", "False alarms T1 → T2", "AP T1 → T2"], rows)

    def yearly_context():
        details = []
        for model, h in [("random_forest", 24), ("xgboost", 6)]:
            d = yearly.loc[yearly.model.eq(model) & yearly.horizon_hours.eq(h) & yearly.test_year.eq(2021)].set_index("config")
            a, b = d.loc["market_matched"], d.loc["market_onchain"]
            details.append(f"{model} h{h} in 2021: Tier 1 catches {int(a.episodes_caught)} "
                           f"versus Tier 2's {int(b.episodes_caught)}, with "
                           f"{int(a.false_alarms)} versus {int(b.false_alarms)} false alarms")
        return ("Yearly context: " + "; ".join(details) +
                ". The earliest test fold has the shortest training history. This describes where "
                "the differences arise, without proving why. Full yearly counts: model_comparison_by_year.csv.")

    eps = episodes.sort_values(["start", "coin"])
    current, coins, number, event_map = None, set(), 0, {}
    for e in eps.itertuples():
        if (current is None or pd.Timestamp(e.start).value-pd.Timestamp(current).value > 48*3600_000_000_000
                or e.coin in coins):
            current, coins, number = e.start, set(), number+1
        coins.add(e.coin)
        event_map[str(e.episode_id)] = (number, current.strftime("%Y-%m-%d"))
    lookup = episodes.assign(episode_id=episodes.episode_id.astype(str)).set_index("episode_id")
    alarms = pd.read_csv(run / "alarm_events.csv")
    if lr_ready:
        alarms = pd.concat([alarms, pd.read_csv(args.lr_run / "alarm_events.csv")], ignore_index=True)

    def episode_table(config):
        rows = []
        for h in [6, 24]:
            for model in ["logreg", "random_forest", "xgboost"]:
                if model == "logreg" and not lr_ready:
                    rows.append([h, model, "PENDING", "PENDING", "PENDING", "PENDING"])
                    continue
                hit = alarms.loc[alarms.config.eq(config) & alarms.model.eq(model) & alarms.horizon_hours.eq(h)]
                ids = sorted(hit.target_episode_id.dropna().astype(str).unique(), key=lambda i: lookup.loc[i, "start"])
                labels = [f"{lookup.loc[i, 'coin']} {lookup.loc[i, 'start']:%Y-%m-%d %H:%M} UTC" for i in ids]
                grouped = sorted(set(event_map[i] for i in ids))
                by_coin = Counter(str(lookup.loc[i, "coin"]) for i in ids)
                by_year = Counter(int(lookup.loc[i, "start"].year) for i in ids)
                rows.append([h, model, "; ".join(labels) or "None",
                             "; ".join(f"event {n} ({date})" for n, date in grouped) or "None",
                             "; ".join(f"{k}: {v}" for k, v in sorted(by_coin.items())) or "None",
                             "; ".join(f"{k}: {v}" for k, v in sorted(by_year.items())) or "None"])
        return table(["h", "Model", "Episodes caught", "Events caught", "By coin", "By year"], rows)

    text = ["# Consolidated presentation values — refit after calibration",
            f"Updated {datetime.now(timezone.utc).isoformat()}. Source run: `{run}` (run ID `{info['run_id']}`).",
            "Corrected_v1; matched Tier 1/2 eligible samples; test years 2021–2025; h6/h24. Practice-model calibration followed by a full-training-window refit. These values replace the earlier no-refit values.",
            "Fixed RF: 500 trees, max depth 8, minimum leaf size 50. Fixed XGBoost: 300 trees, max depth 3, learning rate 0.05; no early stopping. Other settings are recorded in each fit.json.",
            f"LogReg source: `{args.lr_run}`; status: **{lr_status}**. Partial runs are excluded.",
            "## Slide 1 — Random forest impurity importance (Tier 2)",
            table(["Feature/group", "h6", "h24"], rf_rows),
            "Mean normalized impurity importance across five final refitted forests. On-chain features are 12/22 = 54.5% of inputs. Importance is training attribution; it is not the percentage improvement in forecasting.",
            "## Slide 2 — XGBoost SHAP (Tier 2)",
            f"h24 on-chain: **{chain_share[24]*100:.2f}%**; market: **{(1-chain_share[24])*100:.2f}%**. h6 on-chain: **{chain_share[6]*100:.2f}%**.",
            table(["h24 feature", "Share of total absolute SHAP"], [[r.feature, f"{r.share_of_total_abs*100:.2f}%"] for r in selected.itertuples()]),
            "Top three market and top three on-chain features shown. Exact TreeSHAP raw log-odds attribution over every eligible out-of-sample row, using the final refitted models. Attribution does not establish forecasting gains or causality. Regenerate donut geometry with the new shares.",
            "## Slide 3 — Threshold methodology",
            "For a 2024 test example:\n\n1. Fit the practice model on eligible 2018–2022 rows.\n2. Score held-out 2023.\n3. Evaluate every distinct validation score plus the no-alarm cutoff; choose the lowest cutoff with grouped false alarms / observed coin-months ≤ 1.\n4. Refit a fresh model on full eligible 2018–2023 training rows and apply the unchanged cutoff to 2024. Preserve purge/embargo and episode-boundary exclusions.",
            "The budget is 72 only if six coins each have 12 observed months; a month with any eligible rows counts once, without hourly prorating. The budget is pooled, not a separate hard limit for each coin-month. Refit can change score scales; validation feasibility does not guarantee the realised test rate. For LogReg, C is selected by calendar-year CV inside practice training, then held fixed for refitting.",
            table(["Test year", "Calibration year", "Observed calibration coin-months", "False-alarm budget"],
                  [[r.test_year, r.calibration_year, r.calibration_coin_months, r.calibration_budget]
                   for r in audit.loc[audit.model.eq("xgboost") & audit.config.eq("market_onchain") &
                                      audit.horizon_hours.eq(24)].itertuples()]),
            "## Slide 4 — Tier 1 performance (matched sample)", performance("market_matched"),
            "## Slide 5 — Tier 1 findings", findings("market_matched"),
            "Lead times are medians conditional on caught episodes; different models can catch different episodes.",
            "## Slide 6 — Tier 2 performance", performance("market_onchain"),
            "## Slide 7 — Tier 2 and on-chain findings", findings("market_onchain"), onchain_findings(),
            yearly_context(),
            "Compare the recall/false-alarm trade-off and AP together. Both tiers use matching training, calibration and test rows, with their own calibrated thresholds. These are observed sample differences, without a statistical significance claim.",
            "## Slide 8 — Episodes and events caught",
            "Episode timestamps are UTC bar closes. Event IDs are chronological groups with a 48-hour anchored window and no repeated coin; they describe co-occurrence rather than establish a historical cause.",
            "### Tier 2", episode_table("market_onchain"), "### Tier 1 matched", episode_table("market_matched"),
            "## Slide 9 — Benchmarks on the same common sample",
            table(headers, [values(r) for r in benchmark.itertuples()]),
            "Benchmarks follow the same practice-calibration/full-refit protocol. Their values are identical across the matched tiers. Constant prior varies by training year, so pooled AP need not equal pooled prevalence. Compare models at their realised false-alarm rates; do not assert uniform superiority from recall alone.",
            "## Shared footnote",
            "Pooled OOS 2021–2025; corrected_v1; identical eligible coin-hours across tiers; AP = sklearn.average_precision_score (not trapezoidal PR-AUC); exhaustive lowest-feasible practice-model cutoff at ≤1 false alarm / observed coin-month, transferred unchanged to a full-window refit. Median lead conditional on caught episodes.",
            "## Refresh after LogReg completes",
            f"`python -m modelling.consolidate_refit_slides --run-dir {run} --lr-run {args.lr_run} --reuse-shap`",
            "h1 is deliberately excluded from this slide update and is run afterward in a separate output directory."]
    (out / "slide_values.md").write_text("\n\n".join(text) + "\n")
    save_json(out / "values.json", dict(source_run=str(run), run_id=info["run_id"],
        refit_after_calibration=True, rf_importance=imp.to_dict("records"),
        shap=shap.to_dict("records"), shap_top_features=selected.to_dict("records"),
        performance=summaries.to_dict("records"), benchmarks=benchmark.to_dict("records"),
        logreg_status=lr_status))
    print(out / "slide_values.md", flush=True)


if __name__ == "__main__":
    main()
