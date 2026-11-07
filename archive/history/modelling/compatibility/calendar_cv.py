"""Compatibility shim; use modelling.training.calendar_cv."""
from modelling.training.calendar_cv import (
    DEFAULT_C_GRID,
    EMBARGO_HOURS,
    MedianImputer,
    PURGE_HOURS,
    _average_precision,
    _calendar_cv_folds,
    _fit_model,
    average_precision_score,
    get_model,
    np,
    pd,
)
