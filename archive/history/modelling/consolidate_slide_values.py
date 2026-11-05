"""Consolidate slide numbers from matched corrected exhaustive-lowest runs.

Run after new scoring runs complete to refresh the report. Incomplete LogReg
runs are explicitly marked pending; no partial pooled values are reported.
"""
import json
from pathlib import Path
from collections import Counter
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from modelling.alarm_events import build_alarm_events
from modelling.models import get_model
from modelling.rolling_threshold import _choose_threshold, _targetable_episode_ids, _false_alarm_stats

OUT = Path("reports/presentation_consolidated")
RF_XGB = Path("runs/EXHAUSTIVE-LOWEST-NEW")
XGB = Path("runs/XGB-NEW")
LR = Path("runs/LR-EXHAUSTIVE-LOWEST")


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join(["---"] * len(headers)) + " |"] +
                     ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def nfmt(x, digits=1):
    return "—" if pd.isna(x) else f"{x:.{digits}f}"


def main():
    refit_manifest = Path("runs/REFIT-LOWEST-NEW/run.json")
    if refit_manifest.exists() and json.loads(refit_manifest.read_text()).get("status") == "complete":
        raise RuntimeError(
            "Completed refit results are available. Refresh the slide sheet with "
            "'python -m modelling.consolidate_refit_slides --run-dir runs/REFIT-LOWEST-NEW' "
            "to avoid replacing refit values with older no-refit results.")
    OUT.mkdir(parents=True, exist_ok=True)
    summaries = pd.read_csv(RF_XGB / "summary.csv")
    episodes = pd.read_parquet(XGB / "episodes.parquet")
    for folder in [RF_XGB, XGB]:
        assert json.loads((folder / "run.json").read_text())["status"] == "complete"
    lr_info = json.loads((LR / "run.json").read_text()) if (LR / "run.json").exists() else {}
    lr_ready = lr_info.get("status") == "complete" and (LR / "summary.csv").exists()
    lr_summary = pd.read_csv(LR / "summary.csv") if lr_ready else pd.DataFrame()
    if lr_ready:
        assert lr_info.get("threshold_policy") == "exhaustive_lowest"

    importance = []
    for h in [6, 24]:
        for year in range(2021, 2026):
            fit = RF_XGB / "fits" / f"random_forest_market_onchain_h{h}_{year}"
            bundle = joblib.load(fit / "pipeline.joblib")
            model = bundle["model"]
            info = json.loads((fit / "fit.json").read_text())
            importance.extend(dict(horizon=h, feature=f, importance=float(v))
                              for f, v in zip(info["features"], model.feature_importances_))
    imp = pd.DataFrame(importance).groupby(["horizon", "feature"]).importance.mean().reset_index()
    names = ["realised_vol_24h", "peg_stress", "peg_deviation", "realised_vol_6h"]
    rf_rows, rf_values = [], {}
    for name in names + ["other_market", "onchain"]:
        values = []
        for h in [6, 24]:
            d = imp.loc[imp.horizon.eq(h)].set_index("feature").importance
            if name == "onchain":
                v = d[d.index.str.contains("logdiff")].sum()
            elif name == "other_market":
                v = d[~d.index.str.contains("logdiff") & ~d.index.isin(names)].sum()
            else:
                v = d[name]
            rf_values.setdefault(h, {})[name] = float(v)
            values.append(f"{v * 100:.1f}%")
        rf_rows.append([name, *values])
    shap = pd.read_csv(XGB / "shap_summary.csv")
    ranking = pd.read_csv(XGB / "shap_ranking_xgboost_t2_h24.csv")
    selected = pd.concat([ranking[~ranking.feature.str.contains("logdiff")].head(3),
                          ranking[ranking.feature.str.contains("logdiff")].head(3)])
    chain_shares = {h: float(shap.loc[shap.tier.eq(2) & shap.horizon_hours.eq(h),
                                   "onchain_share"].iloc[0]) for h in [6, 24]}

    benchmark_rows, benchmark_events = [], []
    for h in [6, 24]:
        for name in ["baseline", "persistence"]:
            scores, evs, row_counts = [], [], []
            for year in range(2021, 2026):
                fit = XGB / "fits" / f"market_onchain_h{h}_{year}"
                info = json.loads((fit / "fit.json").read_text())
                train = pd.read_parquet(fit / "inner_train.parquet")
                val = pd.read_parquet(fit / "inner_val.parquet")
                test = pd.read_parquet(fit / "test.parquet")
                model = get_model(name).fit(train[info["features"]], train.y_true)
                val.y_score = model.predict_proba(val[info["features"]])[:, 1]
                test.y_score = model.predict_proba(test[info["features"]])[:, 1]
                threshold, stats = _choose_threshold(val, episodes, h, 1.,
                                                     policy="exhaustive_lowest")
                _, events = build_alarm_events(test, episodes, threshold)
                targets = _targetable_episode_ids(test, episodes, h)
                caught = len(set(events.target_episode_id.dropna().astype(str)) & targets)
                false, months, rate, _ = _false_alarm_stats(events, test)
                row_counts.append((caught, len(targets), false, months))
                scores.append(test[["y_true", "y_score"]])
                evs.append(events)
            scores = pd.concat(scores, ignore_index=True)
            events = pd.concat(evs, ignore_index=True)
            totals = np.array(row_counts).sum(axis=0)
            benchmark_rows.append(dict(
                model=name, horizon=h, average_precision=float(average_precision_score(
                    scores.y_true, scores.y_score)), positive_rate=float(scores.y_true.mean()),
                caught=int(totals[0]), targetable=int(totals[1]), false_alarms=int(totals[2]),
                coin_months=int(totals[3]), median_lead_hours=float(
                    events.loc[events.is_true_alarm, "lead_hours"].median())))

    def performance(config):
        rows = []
        for h in [6, 24]:
            for model in ["logreg", "random_forest", "xgboost"]:
                if model == "logreg":
                    if not lr_ready:
                        rows.append([h, model, "PENDING", "0.1019" if h == 6 else "0.4071",
                                     "PENDING", "PENDING", "PENDING"])
                        continue
                    r = lr_summary.loc[lr_summary.model.eq(model) & lr_summary.config.eq(config)
                                       & lr_summary.horizon_hours.eq(h)].iloc[0]
                    positive = r.positive_rate
                else:
                    r = summaries.loc[summaries.model.eq(model) & summaries.config.eq(config)
                                      & summaries.horizon_hours.eq(h)].iloc[0]
                    # Use full pooled counts rather than averaging yearly prevalences.
                    parts = [pd.read_parquet(RF_XGB / "fits" /
                             f"{model}_{config}_h{h}_{y}" / "test.parquet", columns=["y_true"])
                             for y in range(2021, 2026)]
                    positive = pd.concat(parts).y_true.mean()
                rows.append([h, model, f"{r.average_precision:.6f}", f"{positive*100:.4f}",
                             int(r.false_alarms),
                             f"{int(r.episodes_caught)}/{int(r.targetable_episodes)} "
                             f"({r.episode_recall*100:.2f}%)", nfmt(r.median_lead_hours)])
        return table(["h", "Model", "Pooled AP", "Positive base rate (%)",
                      "False alarms", "Episode recall", "Median lead (hours)"], rows)

    events = pd.read_csv(RF_XGB / "alarm_events.csv")
    # Descriptive event grouping matches reporting/event_table.py:
    # anchor each event on its first episode; window 48h; no repeated coin.
    eps = episodes.sort_values(["start", "coin"]).copy()
    current_start, coins, event_number = None, set(), 0
    event_map = {}
    for e in eps.itertuples():
        if current_start is None or e.start-current_start > pd.Timedelta(hours=48) or e.coin in coins:
            current_start, coins, event_number = e.start, set(), event_number + 1
        coins.add(e.coin)
        event_map[str(e.episode_id)] = (event_number, current_start.strftime("%Y-%m-%d"))
    ep_lookup = episodes.assign(episode_id=episodes.episode_id.astype(str)).set_index("episode_id")
    episode_tables = {}
    for config in ["market_matched", "market_onchain"]:
        rows = []
        for h in [6, 24]:
            for model in ["logreg", "random_forest", "xgboost"]:
                if model == "logreg":
                    if not lr_ready:
                        rows.append([h, model, "PENDING", "PENDING", "PENDING", "PENDING"])
                        continue
                    hit = pd.concat([pd.read_csv(
                        LR / "fits" / f"logreg_{config}_h{h}_{y}" / "alarms.csv")
                        for y in range(2021, 2026)], ignore_index=True)
                else:
                    hit = events.loc[events.model.eq(model) & events.config.eq(config) &
                                     events.horizon_hours.eq(h) & events.policy.eq("exhaustive_lowest")]
                ids = sorted(hit.target_episode_id.dropna().astype(str).unique())
                labels = [f"{ep_lookup.loc[i, 'coin']} {ep_lookup.loc[i, 'start']:%Y-%m-%d %H:%M} UTC"
                          for i in ids]
                grouped = sorted(set(event_map[i] for i in ids))
                by_coin = Counter(str(ep_lookup.loc[i, "coin"]) for i in ids)
                by_year = Counter(int(ep_lookup.loc[i, "start"].year) for i in ids)
                rows.append([h, model, "; ".join(labels) or "None",
                             "; ".join(f"event {n} ({date})" for n, date in grouped) or "None",
                             "; ".join(f"{k}: {v}" for k,v in sorted(by_coin.items())) or "None",
                             "; ".join(f"{k}: {v}" for k,v in sorted(by_year.items())) or "None"])
        episode_tables[config] = table(
            ["h", "Model", "Episodes caught", "Events caught", "By coin", "By year"], rows)

    def lr_findings(config):
        if not lr_ready:
            return "LogReg performance and its three-model ranking remain PENDING until the new run completes and is verified."
        parts = []
        for h in [6, 24]:
            r = lr_summary.loc[lr_summary.model.eq("logreg") & lr_summary.config.eq(config)
                               & lr_summary.horizon_hours.eq(h)].iloc[0]
            parts.append(f"h{h}: {int(r.episodes_caught)}/{int(r.targetable_episodes)} caught; "
                         f"{int(r.false_alarms)} false alarms; median lead "
                         f"{nfmt(r.median_lead_hours)} hours")
        return "LogReg: " + "; ".join(parts) + "."

    text = [
        "# Consolidated presentation values",
        "> PROTOCOL UPDATE: These numbers use the earlier no-refit procedure. The requested procedure now refits on the full outer training window after practice-model calibration. Regenerate performance, RF importance and SHAP before using these values for the refit protocol. Run modelling.refit_after_calibration in a new output directory.",
        "",
        "Source PDF: DSE4101 Digital Currencies Group 1 Final Presentation Slides-2.pdf (9 pages).",
        "Protocol: corrected_v1; exhaustive lowest-feasible validation cutoff; budget 1 grouped false alarm per observed coin-month; test years 2021–2025. Tier 1 below means market_matched, not the unrestricted market sample. Both tiers have identical test rows and targetable denominators.",
        "XGB source: runs/XGB-NEW. RF/XGB performance: runs/EXHAUSTIVE-LOWEST-NEW. RF importance below is extracted from these actual fitted RF pipelines, rather than older refits. Threshold changes do not change SHAP or impurity importance.",
        f"LogReg source: runs/LR-EXHAUSTIVE-LOWEST; status at consolidation: {lr_info.get('status', 'missing')}. Never present partial years as pooled 2021–2025 results.",
        "",
        "## Slide 1 — RF importance",
        table(["Feature/group", "h6", "h24"], rf_rows),
        "These are mean normalized impurity shares over five fitted forests. On-chain features comprise 12/22 = 54.5% of inputs. Replace the old percentages with the values above; use 'share of impurity importance', not an out-of-sample gain claim.",
        "",
        "## Slide 2 — XGBoost SHAP",
        f"h24 on-chain: **{chain_shares[24]*100:.2f}%**; market: **{(1-chain_shares[24])*100:.2f}%**. h6 on-chain: **{chain_shares[6]*100:.2f}%**.",
        table(["h24 feature", "Share of total absolute SHAP"], [
            [r.feature, f"{r.share_of_total_abs*100:.2f}%"] for r in selected.itertuples()]),
        "Label this XGBoost Tier 2. Blue bars should identify market features and orange bars on-chain features. The percentages use total model attribution as denominator. SHAP describes attribution, not causality or evidence of better forecasting.",
        "Regenerate the donut geometry from these percentages rather than editing only its text labels.",
        "",
        "## Slide 3 — Threshold methodology",
        "Replace steps 1–4 with:",
        "1. Fit on inner_train (2018–2022 for a 2024 test example).",
        "2. Score the held-out 2023 calibration year with that fitted model.",
        "3. Evaluate every distinct calibration score plus the no-alarm cutoff; choose the LOWEST cutoff with grouped false alarms / observed coin-months <= 1.",
        "4. Freeze the cutoff; refit a fresh model on the full eligible 2018–2023 outer training window, then score 2024 using that unchanged cutoff. Preserve purge/embargo rules. The numerical tables below have NOT yet been regenerated for this step.",
        "For LogReg, select C by calendar-year CV inside inner_train before fitting that scoring model; the 2023 calibration year is reserved for threshold selection.",
        "Remove 'highest recall; fewer-FA tie-break' because that is the other policy. Show practice training through 2022, calibration on 2023, and final refit through 2023. The calibration budget is based on observed coin-months (72 only for six coins with all 12 months). Refit scores may change scale, so the cutoff does not guarantee the same test false-alarm rate. Remove the illustrative 0.728 cutoff unless sourced to a specific model, tier, horizon and year.",
        "",
        "## Slide 4 — Tier 1 performance (MATCHED sample)",
        performance("market_matched"),
        "",
        "## Slide 5 — Tier 1 findings",
        "RF catches 4/33 at h6 (257 false alarms; median 3.5h) and 6/34 at h24 (205 false alarms; median 1.5h). XGB catches 3/33 at h6 (195 false alarms; median 4h) and 4/34 at h24 (200 false alarms; median 9h). RF catches more episodes than XGB here. Longer median leads concern different caught episodes and are not a paired warning-time improvement.",
        lr_findings("market_matched"),
        "",
        "## Slide 6 — Tier 2 performance",
        performance("market_onchain"),
        "",
        "## Slide 7 — Tier 2 findings",
        "RF catches 4/33 at h6 (237 false alarms; median 4h) and 6/34 at h24 (198 false alarms; median 1.5h). XGB catches 2/33 at h6 (177 false alarms; median 3.5h) and 4/34 at h24 (197 false alarms; median 6h). The old claim that XGB has the highest h24 coverage is no longer supported: RF has greater coverage among these two models.",
        lr_findings("market_onchain"),
        "Adding on-chain features: RF retains the same h6/h24 caught counts while reducing false alarms by 20 and 7; XGB loses one h6 catch while reducing false alarms by 18, and retains h24 caught count while reducing false alarms by 3. These are descriptive, model-specific trade-offs.",
        "",
        "## Slide 8 — Episode/event table",
        "Use an explicit tier label. Main table below is Tier 2; a matched Tier 1 reference follows. Event IDs below are chronological identifiers rebuilt from the corrected manifest using the 48-hour anchored/no-repeat-coin grouping rule. Dates alone are descriptive and should not be labelled as a named historical cause without separate evidence.",
        "### Tier 2", episode_tables["market_onchain"],
        "### Tier 1 matched", episode_tables["market_matched"],
        "",
        "## Slide 9 — Benchmarks on the common sample",
        table(["h", "Model", "Pooled AP", "Positive base rate (%)", "False alarms",
               "Episode recall", "Median lead (hours)"], [
            [r["horizon"], r["model"], f'{r["average_precision"]:.6f}',
             f'{r["positive_rate"]*100:.4f}', r["false_alarms"],
             f'{r["caught"]}/{r["targetable"]} ({r["caught"]/r["targetable"]*100:.2f}%)',
             nfmt(r["median_lead_hours"])] for r in benchmark_rows]),
        "Benchmarks are identical across the matched tiers because their scores use no on-chain information. Constant prior is estimated separately from each inner_train partition, so pooled AP need not equal pooled prevalence.",
        "Persistence catches more episodes and has higher AP than the completed RF/XGB runs at h6/h24, but also produces more false alarms. Do not claim a fitted ML model is uniformly superior to persistence. LogReg's comparison remains pending while its run is incomplete.",
        "",
        "## Shared footnote",
        "Pooled out-of-sample test years 2021–2025; corrected_v1; common Tier-2-eligible coin-hours; average precision calculated with sklearn.average_precision_score; annual calibration on the previous held-out year using exhaustive lowest-feasible thresholds at <=1 grouped false alarm per observed coin-month. Realised test rates may differ. Median lead is conditional on caught episodes. Total observed test coin-months: 316, not 317.",
        "",
        "## Refresh after LogReg completes",
        "python -m modelling.verify_lr_rf_run --run-dir runs/LR-EXHAUSTIVE-LOWEST",
        "python -m modelling.consolidate_slide_values",
    ]
    (OUT / "slide_values.md").write_text("\n\n".join(text) + "\n")
    (OUT / "values.json").write_text(json.dumps(dict(
        rf_importance=rf_values, shap_onchain=chain_shares,
        shap_top_features=selected.to_dict("records"),
        performance=summaries.to_dict("records"), benchmarks=benchmark_rows,
        logreg_status=lr_info.get("status", "missing")), indent=2, default=str))
    print(OUT / "slide_values.md")
    print(table(["h", "Benchmark", "AP", "False alarms", "Caught", "Median lead"],
                [[r["horizon"],r["model"],f'{r["average_precision"]:.6f}',r["false_alarms"],
                  f'{r["caught"]}/{r["targetable"]}',nfmt(r["median_lead_hours"])]
                 for r in benchmark_rows]))
    print(table(["RF feature", "h6", "h24"],rf_rows))


if __name__ == "__main__":
    main()
