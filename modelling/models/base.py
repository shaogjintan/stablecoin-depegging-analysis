"""
Model contract for DSE4101 modelling (Phase 3).

Every model used with `modelling/train.py` must subclass `BaseModel` and
implement:

    fit(X_train, y_train, X_val=None, y_val=None) -> self
        X_val/y_val are the fold's inner validation set; NaN-native models
        (XGBoost) may use them for early stopping, others may ignore them.

    predict_proba(X) -> np.ndarray of shape (n, 2)
        Column 1 = P(label = 1).

and override `get_params()` so the train run log records the configuration.

Imputation: models that cannot handle NaN (logistic, random forest) set
`needs_imputation = True`; `train.py` then median-imputes each column with
`MedianImputer`, fitted on the training fold only (never on test/inner_val).
NaN-native models leave it `False` and receive NaNs as-is.

Tuning: `param_grid` is a list of constructor-kwarg dicts tried by
`train.py --tune`, selected by inner-val PR-AUC. Models without tuning
leave `param_grid = [{}]`.

Teammates: implement your model as a subclass in modelling/models/ and
register it in modelling/models/__init__.py (see Rule 3 in agent.md --
all data access goes through modelling/prepare_data.py).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class MedianImputer:
    """Column-wise median imputation; medians are fitted on the training set only."""

    def __init__(self) -> None:
        self.medians_: pd.Series | None = None

    def fit(self, X: pd.DataFrame) -> "MedianImputer":
        self.medians_ = X.median(axis=0)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.medians_ is None:
            raise RuntimeError("MedianImputer.fit must be called before transform")
        X = X.copy()
        for col in self.medians_.index:
            med = self.medians_[col]
            if pd.isna(med):          # constant-NaN column: impute with 0
                X[col] = X[col].fillna(0.0)
            else:
                X[col] = X[col].fillna(med)
        return X


class BaseModel(ABC):
    """The shared model contract. Subclass and register in models/__init__.py."""

    name: str = "base"
    needs_imputation: bool = True
    param_grid: list[dict] = [{}]

    def get_params(self) -> dict:
        """Configuration snapshot for the results log."""
        return {"name": self.name}

    @abstractmethod
    def fit(self, X_train, y_train, X_val=None, y_val=None) -> "BaseModel":
        """Fit on the training fold; optionally use the inner validation set."""

    @abstractmethod
    def predict_proba(self, X) -> np.ndarray:
        """Return predicted probabilities, shape (n, 2); column 1 = P(y=1)."""
