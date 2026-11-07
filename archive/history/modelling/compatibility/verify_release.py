"""Compatibility shim; use modelling.validation.verify_saved_runs."""
from modelling.validation.verify_saved_runs import (
    Path,
    RUNS,
    _choose_threshold,
    argparse,
    check_preprocessing,
    evidence_dir,
    joblib,
    json,
    main,
    np,
    pd,
    reference_ap,
    reference_events,
    reference_metrics,
    resolve_recorded_input,
    sha256,
    verify,
)

if __name__ == "__main__":
    main()
