"""Independently regenerate saved-score metrics and check fit/SHAP alignment.

python -m modelling.verify_corrected_run --run-dir <completed directory>
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from modelling.alarm_events import build_alarm_events
from modelling.corrected_xgb import sha256, save_json
from modelling.rolling_threshold import _targetable_episode_ids, _false_alarm_stats


def verify(out):
    info = json.loads((out / "run.json").read_text())
    assert info["status"] == "complete"
    master = pd.read_parquet(out / "corrected_master.parquet")
    episodes = pd.read_parquet(out / "episodes.parquet")
    assert sha256(out / "corrected_master.parquet") == info["corrected_master_sha256"]
    assert not master.duplicated(["coin", "hour"]).any()
    assert (master.hour - master.bar_start).eq(pd.Timedelta(hours=1)).all()
    summary = pd.read_csv(out / "summary.csv")
    records = []
    for folder in sorted((out / "fits").iterdir()):
        fit = json.loads((folder / "fit.json").read_text())
        model = joblib.load(folder / "model.joblib")
        assert fit["run_id"] == info["run_id"]
        assert sha256(folder / "model.joblib") == fit["model_sha256"]
        assert model.clf_.get_booster().num_boosted_rounds() == fit["boosted_rounds"] == 300
        parts = {}
        for part in ("inner_train", "inner_val", "test"):
            p = pd.read_parquet(folder / f"{part}.parquet")
            assert not p.duplicated(["coin", "hour"]).any()
            assert np.isfinite(p[fit["features"]]).all().all()
            truth = np.zeros(len(p), dtype=int)
            for coin, g in p.groupby("coin", sort=False):
                starts = episodes.loc[episodes.coin.eq(coin), "start"].astype("int64").to_numpy()
                starts.sort()
                hours = g.hour.astype("int64").to_numpy()
                left = np.searchsorted(starts, hours, side="right")
                right = np.searchsorted(starts, hours + fit["horizon"] * 3600_000_000_000, side="right")
                truth[g.index] = right > left
            np.testing.assert_array_equal(truth, p.y_true.to_numpy())
            baseline = master[["coin", "hour"] + fit["features"]].merge(
                p[["coin", "hour"]], on=["coin", "hour"], validate="one_to_one")
            baseline = p[["coin", "hour"]].merge(baseline, on=["coin", "hour"], validate="one_to_one")
            np.testing.assert_allclose(baseline[fit["features"]], p[fit["features"]], atol=0, rtol=0)
            if part != "inner_train":
                scores = model.predict_proba(p[fit["features"]])[:, 1]
                np.testing.assert_allclose(scores, p.y_score, atol=1e-8, rtol=0)
            parts[part] = p
        assert parts["inner_train"].hour.max() < parts["inner_val"].hour.min()
        assert parts["inner_val"].hour.max() < parts["test"].hour.min()
        assert parts["inner_val"].hour.min() - parts["inner_train"].hour.max() >= pd.Timedelta(hours=49)
        assert parts["test"].hour.min() - parts["inner_val"].hour.max() >= pd.Timedelta(hours=49)
        _, val_events = build_alarm_events(parts["inner_val"], episodes, fit["threshold"])
        false, months, rate, _ = _false_alarm_stats(val_events, parts["inner_val"])
        assert false == fit["validation"]["false_alarms"]
        assert months == fit["validation"]["coin_months"]
        assert rate <= 1
        assert fit["validation"]["candidate_count"] == parts["inner_val"].y_score.nunique() + 1
        records.append(dict(fit_id=fit["fit_id"], boosted_rounds=300,
                            config=fit["config"], horizon_hours=fit["horizon"], test_year=fit["year"],
                            test_flagged_hours=int(parts["test"].y_score.ge(fit["threshold"]).sum()),
                            scores_and_features_exact=True, labels_match_merged_starts=True,
                            disjoint_clock=True, validation_budget_pass=True))
    for r in summary.itertuples():
        pred = pd.read_parquet(out / f"predictions_{r.config}_h{r.horizon_hours}.parquet")
        events = pd.read_csv(out / f"alarms_{r.config}_h{r.horizon_hours}.csv")
        targets = _targetable_episode_ids(pred, episodes, r.horizon_hours)
        caught = set(events.target_episode_id.dropna()) & targets
        assert len(targets) == r.targetable_episodes and len(caught) == r.episodes_caught
        assert (events.lead_hours.dropna() > 0).all()
        assert (events.lead_hours.dropna() <= r.horizon_hours).all()
        assert int((~events.is_true_alarm).sum()) == r.false_alarms
        assert len(pred) == r.rows and int(pred.y_true.sum()) == r.positive_hours
        assert abs(average_precision_score(pred.y_true, pred.y_score) - r.average_precision) < 1e-12
        assert _false_alarm_stats(events, pred)[1] == r.observed_coin_months
    for h in (6, 24):
        for year in range(2021, 2026):
            folders = [out / "fits" / f"{c}_h{h}_{year}" for c in ("market_matched", "market_onchain")]
            for part in ("inner_train", "inner_val", "test"):
                a, b = [pd.read_parquet(f / f"{part}.parquet") for f in folders]
                pd.testing.assert_frame_equal(a[["coin", "hour", "y_true"]], b[["coin", "hour", "y_true"]])
        for tier, config in ((1, "market_original"), (2, "market_onchain")):
            tag = f"xgboost_t{tier}_h{h}"
            sv = np.load(out / f"shap_values_{tag}.npz")["values"]
            keys = pd.read_parquet(out / f"shap_rows_{tag}.parquet")
            pred = pd.read_parquet(out / f"predictions_{config}_h{h}.parquet")
            pd.testing.assert_frame_equal(keys, pred[["coin", "hour", "run_id", "fit_id"]])
            rank = pd.read_csv(out / f"shap_ranking_{tag}.csv").set_index("feature")
            features = np.load(out / f"shap_values_{tag}.npz")["features"].tolist()
            np.testing.assert_allclose(np.abs(sv).mean(axis=0), rank.loc[features, "mean_abs_shap"], atol=1e-12)
            checks = pd.read_csv(out / f"shap_checks_{tag}.csv")
            for check in checks.itertuples():
                mask = keys.fit_id.eq(check.fit_id).to_numpy()
                folder = out / "fits" / check.fit_id
                model = joblib.load(folder / "model.joblib")
                X = pd.read_parquet(folder / "test.parquet")[features]
                margin = model.clf_.predict(X, output_margin=True)
                assert np.max(np.abs(sv[mask].sum(axis=1) + check.expected_margin - margin)) <= 1e-4
            share = pd.read_csv(out / f"group_share_{tag}.csv")
            assert abs(share.market.iloc[0] + share.onchain.iloc[0] - 1) < 1e-10
    pd.DataFrame(records).to_csv(out / "verification_fits.csv", index=False)
    pd.DataFrame(records).groupby(["config", "horizon_hours"]).test_flagged_hours.sum().reset_index().to_csv(
        out / "raw_flagged_counts.csv", index=False)
    for rel, expected in info["source_code_sha256"].items():
        assert sha256(out / "source" / Path(rel).name) == expected
    save_json(out / "verification.json", dict(run_id=info["run_id"], fits_checked=len(records),
                                             matched_partitions_checked=30, shap_configs_checked=4,
                                             independent_metric_regeneration=True, status="passed"))
    make_figures(out, summary)
    artifacts = [p for p in out.rglob("*") if p.is_file()
                 and (p.suffix in (".parquet", ".joblib", ".ubj", ".npz", ".csv", ".png")
                      or p.name == "run.json")]
    save_json(out / "artifact_sha256.json", {str(p.relative_to(out)): sha256(p) for p in artifacts})
    print(f"PASS: {len(records)} fits, 30 matched partitions, 4 all-row SHAP alignments and pooled metrics")


def make_figures(out, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for h in (6, 24):
        d = summary[summary.horizon_hours.eq(h)]
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        labels = ["Original market", "Matched market", "Market + on-chain"]
        for ax, col, title in zip(axes, ["average_precision", "episode_recall", "false_alarms_per_coin_month"],
                                  ["Pooled average precision", "Targetable episode recall", "False alarms / observed coin-month"]):
            ax.bar(labels, d[col], color=["#4C72B0", "#55A868", "#DD8452"])
            ax.set_title(title, fontsize=10)
            ax.tick_params(axis="x", rotation=25)
            for i, val in enumerate(d[col]):
                ax.text(i, val, f"{val:.4f}", ha="center", va="bottom", fontsize=9)
        fig.suptitle(f"Corrected fixed300 XGBoost h{h}; validation budget ≠ test guarantee")
        fig.tight_layout()
        fig.savefig(out / "figures" / f"corrected_metrics_h{h}.png", dpi=160)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    verify(args.run_dir.resolve())


if __name__ == "__main__":
    main()
