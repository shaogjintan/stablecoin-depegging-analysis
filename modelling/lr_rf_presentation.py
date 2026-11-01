"""Export sourced presentation tables only after independent run verification."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

FIELDS = ["model", "config", "horizon_hours", "average_precision", "positive_rate", "false_alarms",
          "episodes_caught", "targetable_episodes", "median_lead_hours", "observed_coin_months",
          "false_alarms_per_coin_month", "run_id"]


def export(out, xgb_run=None):
    info = json.loads((out / "run.json").read_text())
    verified = json.loads((out / "verification.json").read_text())
    assert info["status"] == "complete" and verified["status"] == "passed"
    if xgb_run is None and not info.get("frozen_path"):
        raise ValueError("Portable presentation export requires an explicit --xgb-run")
    frozen = xgb_run if xgb_run is not None else Path(info["frozen_path"])
    xinfo = json.loads((frozen / "run.json").read_text())
    xverified = json.loads((frozen / "verification.json").read_text())
    assert xinfo["status"] == "complete" and xverified["status"] == "passed"
    for field in ("source_master_sha256", "dataset_version", "purge_hours", "embargo_hours"):
        assert info.get(field) == xinfo.get(field), f"Cannot combine runs: {field} differs"
    assert info.get("source_episodes_sha256") == xinfo.get("source_episodes_sha256")
    for path in ("modelling\\corrected_data.py", "data_processing\\splits.py",
                 "data_processing\\config.py", "modelling\\rolling_threshold.py",
                 "modelling\\exact_threshold.py", "modelling\\alarm_events.py"):
        assert info["source_code_sha256"][path] == xinfo["source_code_sha256"][path], (
            f"Cannot combine runs: shared protocol differs at {path}")
    assert tuple(info.get("test_years", range(2021, 2026))) == tuple(
        xinfo.get("test_years", range(2021, 2026)))
    current = pd.read_csv(out / "summary.csv")
    xgb = pd.read_csv(frozen / "summary.csv").assign(model="xgboost")
    combined = pd.concat([current, xgb], ignore_index=True)
    scope = set(zip(xgb.config, xgb.horizon_hours))
    assert set(xgb.config) == {"market_original", "market_matched", "market_onchain"}
    assert set(current.model) == {"logreg", "random_forest", "baseline", "persistence"}
    for _, data in current.groupby("model"):
        assert set(zip(data.config, data.horizon_hours)) == scope, "Model scopes differ"
    assert not combined.duplicated(["model", "config", "horizon_hours"]).any()
    p = out / "presentation"
    p.mkdir(exist_ok=False)
    combined[FIELDS].to_csv(p / "pooled_slide_table.csv", index=False)
    for config, name in (("market_original", "page20_own_market.csv"),
                         ("market_onchain", "page24_onchain.csv")):
        combined.loc[combined.config.eq(config), FIELDS].to_csv(p / name, index=False)
    comparisons = []
    for model, d in combined.groupby("model", sort=True):
        for h in sorted(d.horizon_hours.unique()):
            a = d.loc[d.config.eq("market_matched") & d.horizon_hours.eq(h)].iloc[0]
            b = d.loc[d.config.eq("market_onchain") & d.horizon_hours.eq(h)].iloc[0]
            comparisons.append(dict(
                model=model, horizon_hours=h, run_id=a.run_id,
                market_ap=a.average_precision, onchain_ap=b.average_precision,
                ap_difference=b.average_precision - a.average_precision,
                market_catches=a.episodes_caught, onchain_catches=b.episodes_caught,
                targetable_episodes=a.targetable_episodes,
                recall_difference=((b.episodes_caught - a.episodes_caught) / a.targetable_episodes
                                   if a.targetable_episodes else np.nan),
                market_false_alarms=a.false_alarms, onchain_false_alarms=b.false_alarms,
                false_alarm_difference=b.false_alarms - a.false_alarms,
                observed_coin_months=a.observed_coin_months))
    matched = pd.DataFrame(comparisons)
    matched.to_csv(p / "pages25_26_matched_comparison.csv", index=False)
    unc = pd.concat([pd.read_csv(out / "matched_uncertainty.csv"),
                     pd.read_csv(frozen / "matched_uncertainty.csv").assign(model="xgboost",
                                                                           run_id=xgb.run_id.iloc[0])])
    for r in unc.itertuples():
        m = matched.loc[matched.model.eq(r.model) & matched.horizon_hours.eq(r.horizon_hours)].iloc[0]
        np.testing.assert_allclose([r.average_precision_difference, r.episode_recall_difference],
                                   [m.ap_difference, m.recall_difference], atol=1e-12, rtol=0)
    unc.to_csv(p / "pages25_26_conditional_uncertainty.csv", index=False)
    years = pd.read_csv(out / "per_year.csv")
    selected = years.loc[years.model.eq("logreg"), ["config", "horizon_hours", "test_year", "selected_C"]]
    selected.pivot(index=["config", "horizon_hours"], columns="test_year", values="selected_C").to_csv(
        p / "pages11_12_selected_C.csv")
    imp = pd.read_csv(out / "importance_per_fit.csv")
    lr = imp.loc[imp.model.eq("logreg")].copy()
    lr["absolute_value"] = lr.value.abs()
    lr.to_csv(p / "pages11_12_lr_coefficients_per_fit.csv", index=False)
    lr.groupby(["config", "horizon_hours", "feature", "is_onchain"]).agg(
        mean_signed_coefficient=("value", "mean"), mean_absolute_coefficient=("absolute_value", "mean"),
        min_coefficient=("value", "min"), max_coefficient=("value", "max"),
        nonzero_fits=("value", lambda s: int(s.ne(0).sum()))).reset_index().to_csv(
            p / "pages11_12_lr_coefficient_summary.csv", index=False)
    rf = imp.loc[imp.model.eq("random_forest")]
    rf.groupby(["config", "horizon_hours", "feature", "is_onchain"]).value.agg(
        ["mean", "min", "max"]).reset_index().to_csv(p / "pages13_14_rf_importance.csv", index=False)
    rf.groupby(["config", "horizon_hours", "test_year", "is_onchain"]).value.sum().reset_index().to_csv(
        p / "pages13_14_rf_group_share_per_fit.csv", index=False)
    xyears = pd.read_csv(frozen / "per_year.csv").assign(model="xgboost")
    common = ["model", "config", "horizon_hours", "test_year", "false_alarms", "observed_coin_months",
              "false_alarms_per_coin_month", "episodes_caught", "targetable_episodes", "average_precision", "run_id"]
    pd.concat([years[common], xyears[common]], ignore_index=True).to_csv(
        p / "pages30_31_yearly_cost_recall.csv", index=False)
    sources = dict(
        pooled_slide_table="LR/RF/baselines: ../summary.csv; XGB: frozen run summary.csv",
        matched_comparison="paired rows from pooled_slide_table.csv, market_onchain minus market_matched",
        uncertainty="../matched_uncertainty.csv and frozen matched_uncertainty.csv; 500 coin-month AP / 5000 onset-cluster recall resamples, no refit",
        selected_C="../per_year.csv and ../fits/logreg_{config}_h{h}_{year}/tuning.json",
        coefficients="../importance_per_fit.csv and exact scoring pipeline.joblib; standardized coefficients",
        rf_importance="../importance_per_fit.csv and exact scoring pipeline.joblib; normalized impurity decrease",
        yearly="../per_year.csv and frozen per_year.csv",
        current_run_id=info["run_id"], frozen_run_id=xinfo["run_id"],
        frozen_path=str(frozen), verified=True,
        caveats="Own market is unmatched availability sample. Validation budget not test guarantee. Importance not causal forecasting benefit.")
    (p / "sources.json").write_text(json.dumps(sources, indent=2), encoding="utf-8")
    print(f"Exported {len(combined)} pooled rows and sourced methodology/matched/yearly tables to {p}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--xgb-run", type=Path,
                        help="independently verified XGB run with identical inputs, splits and scope")
    args = parser.parse_args()
    export(args.run_dir.resolve(), args.xgb_run.resolve() if args.xgb_run else None)
