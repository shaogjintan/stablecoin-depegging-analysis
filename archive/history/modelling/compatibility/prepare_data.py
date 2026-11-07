"""Compatibility shim; use modelling.data.sample_preparation."""
from modelling.data.sample_preparation import (
    DEPEG_GAP_HOURS,
    FORECAST_HORIZONS_HOURS,
    MARKET_FEATURES,
    META_COLS,
    MODELLING_DATA_VERSION,
    ONCHAIN_FEATURES,
    Path,
    RECOVERY_HOURS,
    TIERS,
    _xy,
    at_risk_mask,
    corrected_folds,
    expanding_window_folds,
    load_folds,
    load_master,
    main,
    np,
    pd,
    prediction_eligible_mask,
    recovery_window_mask,
    sys,
)

if __name__ == "__main__":
    main()
