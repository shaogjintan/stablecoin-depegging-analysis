"""Compare fixed 300 trees with early stopping on the same historical validation year.

Run: python3 modelling/compare_xgb_early_stopping.py
Both arms fit inner_train, calibrate alarms on inner_val, and score test.
Early stopping uses XGBoost aucpr, patience 100, and a ceiling of 300 trees.
The validation year serves both model selection and alarm calibration; this
is a development comparison, with outer test years held out throughout.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_processing.config import ACTIVE_EPISODES_PATH
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from modelling import false_alarm_framework as framework
from modelling.models import get_model
from modelling.evaluate import pooled_metrics
from modelling.prepare_data import load_folds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path,
                        default=REPO / "reports/xgb_early_stopping")
    parser.add_argument("--tiers", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 6, 24])
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    # The episode manifest is read only for alarm accounting, as in the
    # existing final-results framework; model inputs come from load_folds.
    episodes = pd.read_parquet(ACTIVE_EPISODES_PATH)
    all_years, summary_rows, tree_rows, all_events = [], [], [], []
    original_scorer = framework.score_fold
    original_tag = framework.FIXED_TAGS["xgboost"]
    try:
        for tier in args.tiers:
            for horizon in args.horizons:
                framework._FOLDS_CACHE[(tier, horizon)] = load_folds(horizon=horizon, tier=tier)
                for arm in ("fixed_300", "early_stopping_300"):
                    def score(fold, model_name, model_params=None):
                        use_es = arm == "early_stopping_300" and (
                            0 < int(fold["y_inner_val"].sum()) < len(fold["y_inner_val"]))
                        model = get_model("xgboost", **framework.XGBOOST_FIXED_PARAMS,
                                          early_stopping_rounds=100, n_jobs=4)
                        model.fit(fold["X_inner_train"], fold["y_inner_train"],
                                  X_val=fold["X_inner_val"] if use_es else None,
                                  y_val=fold["y_inner_val"] if use_es else None)
                        count = model.best_iteration_ + 1 if use_es else 300
                        tree_rows.append(dict(tier=tier, horizon_hours=horizon,
                                              test_year=fold["year"], arm=arm,
                                              selected_trees=count,
                                              early_stopping_used=use_es,
                                              validation_positives=int(fold["y_inner_val"].sum())))
                        print(f"T{tier} h{horizon} {arm} {fold['year']}: {count} trees",
                              flush=True)
                        return (model.predict_proba(fold["X_inner_val"])[:, 1],
                                model.predict_proba(fold["X_test"])[:, 1])
                    framework.score_fold = score
                    framework.FIXED_TAGS["xgboost"] = arm
                    rows, events, score_frames = framework.evaluate_configuration(
                        "xgboost", tier, horizon, episodes, 1.0)
                    yr = pd.DataFrame(rows).assign(arm=arm)
                    all_years.append(yr)
                    ev = pd.concat(events, ignore_index=True)
                    ev["arm"] = arm
                    all_events.append(ev)
                    scores = pd.concat(score_frames, ignore_index=True)
                    cls = pooled_metrics(scores.y_true, scores.y_score)
                    caught = int(yr.test_episodes_caught.sum())
                    targetable = int(yr.test_targetable_episodes.sum())
                    false = int(yr.test_false_alarms.sum())
                    months = int(yr.test_coin_months.sum())
                    true = int(yr.test_true_alarms.sum())
                    alarms = int(yr.test_alarm_events.sum())
                    lead = ev.loc[ev.is_true_alarm, "lead_hours"].median() if len(ev) else np.nan
                    summary_rows.append(dict(
                        tier=tier, horizon_hours=horizon, arm=arm,
                        average_precision=cls["pr_auc"], positive_base_rate=cls["pos_rate"],
                        test_rows=cls["n"], episodes_caught=caught,
                        targetable_episodes=targetable, episode_recall=caught / targetable,
                        false_alarms=false, coin_months=months,
                        false_alarms_per_coin_month=false / months,
                        alarm_events=alarms, alarm_precision=true / alarms if alarms else np.nan,
                        median_lead_hours=lead))
                    # Checkpoint after each arm so a long run retains completed results.
                    pd.DataFrame(summary_rows).to_csv(args.out_dir / "summary.csv", index=False)
                    pd.concat(all_years).to_csv(args.out_dir / "by_year.csv", index=False)
                    pd.DataFrame(tree_rows).to_csv(args.out_dir / "tree_counts.csv", index=False)
        summary = pd.DataFrame(summary_rows)
        metrics = ["average_precision", "episode_recall", "episodes_caught",
                   "false_alarms", "false_alarms_per_coin_month", "median_lead_hours"]
        paired = summary.pivot(index=["tier", "horizon_hours"], columns="arm", values=metrics)
        delta = pd.DataFrame({m: paired[(m, "early_stopping_300")] -
                              paired[(m, "fixed_300")] for m in metrics}).reset_index()
        delta.to_csv(args.out_dir / "early_stopping_minus_fixed.csv", index=False)
        pd.concat(all_events).to_csv(args.out_dir / "alarm_events.csv", index=False)
        (args.out_dir / "protocol.json").write_text(json.dumps({
            "fixed_parameters": framework.XGBOOST_FIXED_PARAMS,
            "early_stopping_patience": 100, "selection_metric": "xgboost aucpr",
            "reporting_metric": "sklearn average precision",
            "alarm_budget_per_coin_month": 1,
            "training": "inner_train in both arms; no full-window refit",
            "validation": "inner_val selects tree count and alarm threshold",
            "single_class_validation": "fall back to fixed 300 trees",
        }, indent=2))
        print(delta.to_string(index=False))
    finally:
        framework.score_fold = original_scorer
        framework.FIXED_TAGS["xgboost"] = original_tag


if __name__ == "__main__":
    main()
