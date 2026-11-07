"""Compatibility shim; use modelling.evaluation.alarm_events."""
from modelling.evaluation.alarm_events import (
    COUNT_COLUMNS,
    EVENT_COLUMNS,
    PROCESSED_DATA_DIR,
    Path,
    REPO_ROOT,
    REQUIRED_PREDICTION_COLUMNS,
    _group_counts,
    argparse,
    build_alarm_events,
    main,
    np,
    pd,
    summarize_alarm_events,
    sys,
)

if __name__ == "__main__":
    main()
