"""Compatibility shim; use modelling.evaluation.metrics."""
from modelling.evaluation.metrics import (
    PROCESSED_DATA_DIR,
    Path,
    RESULTS_DIR,
    _episode_for,
    _self_test,
    _valid,
    alarm_curve,
    alarm_sweep,
    average_precision_score,
    brier_score_loss,
    episode_alarm_table,
    episode_recall_curve,
    evaluate_run,
    missed_episodes,
    np,
    operating_points,
    pd,
    per_coin_table,
    per_episode_table,
    pooled_metrics,
    precision_recall_curve,
    print_report,
    roc_auc_score,
    select_threshold,
    sys,
)

if __name__ == "__main__":
    raise SystemExit(_self_test())
