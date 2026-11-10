"""Fit constant-prior and persistence benchmarks on saved partitions.

This exporter fits benchmark models; it does not retrain primary models.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from modelling.reporting.schema import write_comparison
from modelling.models import get_model
from modelling.utils.artifacts import evidence_dir, FINAL_RUNS
from modelling.evaluation.alarm_events import build_alarm_events
from modelling.evaluation.threshold_selection import (
    _choose_threshold, _targetable_episode_ids, _false_alarm_stats,
)

def benchmarks(run, episodes):
    """Fit and pool constant-prior and persistence benchmark results."""
    rows = []
    for h in [6, 24]:
        for name in ["baseline", "persistence"]:
            predictions, alarms_all, counts = [], [], []
            for year in range(2021, 2026):
                folder = evidence_dir(run) / "fits" / f"xgboost_market_onchain_h{h}_{year}"
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
    """Fit and export benchmarks using a completed run's saved partitions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path,
                        default=FINAL_RUNS[0])
    parser.add_argument('--out-dir', type=Path,
                        default=Path('reports/model_results'))
    args = parser.parse_args()
    info = json.loads((args.run_dir / 'run.json').read_text())
    if info.get('status') != 'complete' or not info.get('refit_after_calibration'):
        raise ValueError('A completed final refit run is required')
    episodes = pd.read_parquet(evidence_dir(args.run_dir) / 'episodes.parquet')
    result = benchmarks(args.run_dir, episodes)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_comparison(result, args.out_dir / 'benchmark.csv',
                     ['model', 'horizon_hours'])


if __name__ == '__main__':
    main()
