"""Export per-coin final-model metrics from saved predictions and alarms."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from data_processing.config import FIAT_COINS
from modelling.corrected_xgb import sha256

KEYS = ["model", "config", "horizon_hours", "policy"]
COUNTS = [
    "test_rows", "positive_hours", "coin_months", "targetable_episodes",
    "episodes_caught", "alarm_events", "true_alarms", "false_alarms",
]



def targetable_ids(predictions, episodes, horizon):
    """Map positive origins to the first future onset using the core rule."""
    targets = set()
    positive = predictions.loc[predictions.y_true.eq(1)]
    delta = horizon * 3600_000_000_000
    for coin, origins in positive.groupby("coin", sort=False):
        eps = episodes.loc[episodes.coin.eq(coin)].sort_values("start")
        if eps.empty:
            continue
        starts = pd.to_datetime(eps.start, utc=True).astype(
            "datetime64[ns, UTC]"
        ).astype("int64").to_numpy()
        times = pd.to_datetime(origins.hour, utc=True).astype(
            "datetime64[ns, UTC]"
        ).astype("int64").to_numpy()
        positions = np.searchsorted(starts, times, side="right")
        valid = positions < len(starts)
        selected = positions[valid]
        within = starts[selected] <= times[valid] + delta
        targets.update(eps.iloc[selected[within]].episode_id.astype(str))
    return targets


def coin_metrics(predictions, alarms, episodes, horizon):
    """Pool one coin's held-out years, retaining episode-level denominators."""
    targets = targetable_ids(predictions, episodes, horizon)
    true = alarms.is_true_alarm
    caught = set(alarms.loc[true, "target_episode_id"].dropna().astype(str))
    if not caught.issubset(targets):
        raise ValueError("Caught episode is outside this coin's targetable set")
    months = predictions.hour.dt.strftime("%Y-%m").nunique()
    positives = int(predictions.y_true.sum())
    false = int((~true).sum())
    return dict(
        test_rows=len(predictions),
        positive_hours=positives,
        positive_rate=(
            positives / len(predictions) if len(predictions) else np.nan
        ),
        coin_months=months,
        targetable_episodes=len(targets),
        episodes_caught=len(caught),
        episode_recall=len(caught) / len(targets) if targets else np.nan,
        alarm_events=len(alarms),
        true_alarms=int(true.sum()),
        false_alarms=false,
        false_alarms_per_coin_month=false / months if months else np.nan,
        alarm_precision=float(true.mean()) if len(alarms) else np.nan,
        average_precision=(
            float(average_precision_score(
                predictions.y_true, predictions.y_score
            )) if positives else np.nan
        ),
        median_lead_hours=(
            float(alarms.loc[true, "lead_hours"].median())
            if true.any() else np.nan
        ),
    )


def fit_index(root):
    """Locate final fits by provenance rather than historical folder names."""
    index = {}
    for path in sorted(root.rglob("fit.json")):
        if "source" in path.parts:
            continue
        spec = json.loads(path.read_text())
        if not spec.get("refit_after_calibration"):
            continue
        key = (spec["model"], spec["config"], spec["horizon"], spec["year"])
        index.setdefault(key, []).append((path.parent, spec))
    return index


def build_report(summary, root):
    """Build the report and require count agreement with each summary row."""
    index = fit_index(root)
    manifests = []
    for path in root.glob("*/*/run.json"):
        info = json.loads(path.read_text())
        if (info.get("refit_after_calibration")
                and info.get("status") == "complete"):
            manifests.append((path, info))
    rows, sources = [], []
    for item in summary.itertuples(index=False):
        compatible = [
            (path, info) for path, info in manifests
            if item.model in info["models"]
            and item.config in info["configs"]
            and item.horizon_hours in info["horizons"]
            and item.policy == info["policy"]
        ]
        if len(compatible) != 1:
            raise ValueError(f"Expected one final source run for {item}")
        manifest, info = compatible[0]
        predictions, alarms, episodes = [], [], None
        evidence = []
        for year in info["years"]:
            key = (item.model, item.config, item.horizon_hours, year)
            candidates = [
                (folder, spec) for folder, spec in index.get(key, [])
                if spec["run_id"] == info["run_id"]
                and spec["threshold_policy"] == item.policy
            ]
            if len(candidates) != 1:
                raise ValueError(f"Expected one saved final fit: {key}")
            folder, spec = candidates[0]
            prediction_path = folder / "test.parquet"
            alarm_path = folder / "alarms.csv"
            episode_path = folder.parent.parent / "episodes.parquet"
            current_episodes = pd.read_parquet(episode_path)
            if episodes is None:
                episodes = current_episodes
            else:
                pd.testing.assert_frame_equal(episodes, current_episodes)
            pred = pd.read_parquet(prediction_path, columns=[
                "coin", "hour", "y_true", "y_score", "threshold"
            ])
            pred["hour"] = pd.to_datetime(pred.hour, utc=True)
            event = pd.read_csv(alarm_path, float_precision="round_trip")
            event["alarm_hour"] = pd.to_datetime(event.alarm_hour, utc=True)
            labels = event.is_true_alarm.astype(str).str.lower()
            if not labels.isin(["true", "false", "1", "0"]).all():
                raise ValueError(f"Invalid alarm truth values: {alarm_path}")
            event["is_true_alarm"] = labels.isin(["true", "1"])
            if not pred.hour.dt.year.eq(year).all():
                raise ValueError(f"Wrong prediction year: {prediction_path}")
            if not pred.threshold.eq(spec["threshold"]).all():
                raise ValueError(f"Wrong saved cutoff: {prediction_path}")
            origins = pd.MultiIndex.from_frame(pred[["coin", "hour"]])
            fired = pd.MultiIndex.from_frame(event[["coin", "alarm_hour"]])
            if not fired.isin(origins).all():
                raise ValueError(f"Alarm outside saved origins: {alarm_path}")
            predictions.append(pred)
            alarms.append(event)
            evidence.append({
                str(path): sha256(path) for path in [
                    folder / "fit.json", prediction_path, alarm_path,
                    episode_path,
                ]
            })
        pred = pd.concat(predictions, ignore_index=True)
        event = pd.concat(alarms, ignore_index=True)
        if pred.duplicated(["coin", "hour"]).any():
            raise ValueError("Duplicate pooled prediction origins")
        if event.duplicated(["coin", "alarm_hour"]).any():
            raise ValueError("Duplicate pooled alarms")
        cell = []
        for coin in FIAT_COINS:
            row = dict(zip(KEYS, [getattr(item, key) for key in KEYS]))
            row.update(coin=coin, **coin_metrics(
                pred.loc[pred.coin.eq(coin)],
                event.loc[event.coin.eq(coin)],
                episodes.loc[episodes.coin.eq(coin)], item.horizon_hours,
            ))
            cell.append(row)
        for field in COUNTS:
            actual = sum(row[field] for row in cell)
            if actual != getattr(item, field):
                raise ValueError(f"Summary mismatch: {item.model} {field}")
        np.testing.assert_allclose(
            average_precision_score(pred.y_true, pred.y_score),
            item.average_precision, rtol=1e-10, atol=1e-12,
        )
        rows.extend(cell)
        sources.append(dict(
            model=item.model, config=item.config,
            horizon_hours=item.horizon_hours, run_id=info["run_id"],
            manifest=str(manifest), manifest_sha256=sha256(manifest),
            fit_inputs=evidence,
        ))
    return pd.DataFrame(rows).sort_values(KEYS + ["coin"]), sources


def main():
    """Write per-coin metrics for the scope of the current final summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts", type=Path, default=Path("artifacts/runs")
    )
    parser.add_argument(
        "--summary", type=Path,
        default=(
            Path("reports/model_results") / "model_comparison_summary.csv"
        ),
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("reports/model_results")
    )
    args = parser.parse_args()
    summary = pd.read_csv(args.summary, float_precision="round_trip")
    if summary.duplicated(KEYS).any():
        raise ValueError("Duplicate summary cells")
    report, sources = build_report(summary, args.artifacts)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = args.out_dir / "model_comparison_by_coin.csv"
    report.to_csv(output, index=False)
    (args.out_dir / "model_comparison_by_coin_manifest.json").write_text(
        json.dumps(dict(
            summary=str(args.summary), summary_sha256=sha256(args.summary),
            rows=len(report), sources=sources,
            notes=[
                "Saved final predictions and alarms; no retraining.",
                "Counts reconcile to the overall summary.",
                "AP pools each coin's test years; it is not additive.",
                "Recall/AP are blank with no targets/positives respectively.",
                "Observed coin-months include partial months.",
            ],
        ), indent=2) + "\n"
    )
    print(f"Saved {len(report)} per-coin rows; all summary counts reconcile.")


if __name__ == "__main__":
    main()
