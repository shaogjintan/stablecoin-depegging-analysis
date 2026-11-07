"""Compatibility shim; use modelling.reporting.slide_support."""
from modelling.reporting.slide_support import (
    Counter,
    ONCHAIN_FEATURES,
    Path,
    _choose_threshold,
    _false_alarm_stats,
    _targetable_episode_ids,
    argparse,
    average_precision_score,
    benchmarks,
    build_alarm_events,
    calibration_audit_rows,
    canonical_input_hashes,
    datetime,
    evidence_dir,
    get_model,
    joblib,
    json,
    main,
    nfmt,
    np,
    pd,
    require_run,
    save_json,
    sha256,
    shap_values,
    table,
    timezone,
)

if __name__ == "__main__":
    main()
