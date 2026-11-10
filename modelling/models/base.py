"""Shared estimator interface and training-fitted median imputer.

Models implement fit(), predict_proba(), and get_params(). The training runner
coordinates practice fitting, cutoff calibration, final refitting, and scoring.
For models requiring imputation, the pipeline fits MedianImputer on each
training partition and reuses it for that fitted model's predictions.

Final RF/XGB settings and the LR C grid live in training/parameters.py.
Logistic C selection lives in training/logistic_tuning.py. X_val/y_val remain
optional interface arguments; the maintained final runner does not use them
for XGBoost early stopping. Preserve class module paths for saved models.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class MedianImputer:
    """Impute missing values using training-fitted column medians.

    Column-wise median imputation; medians are fitted on the training set only.
    """

    def __init__(self) -> None:
        """Initialize the unfitted median-imputation state."""
        self.medians_: pd.Series | None = None

    def fit(self, X: pd.DataFrame) -> "MedianImputer":
        """Learn one median per input column from the training data."""
        self.medians_ = X.median(axis=0)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fill missing input values using previously learned column medians."""
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
    """Define the shared estimator interface for model implementations.

    The shared model contract. Subclass and register in models/__init__.py.
    """

    name: str = "base"
    needs_imputation: bool = True
    # Capability flag; the final runner does not request early stopping.
    supports_early_stopping: bool = False

    def get_params(self) -> dict:
        """Configuration snapshot for the results log."""
        return {"name": self.name}

    @abstractmethod
    def fit(self, X_train, y_train, X_val=None, y_val=None) -> "BaseModel":
        """Fit on the training fold; optionally use the inner validation set."""

    @abstractmethod
    def predict_proba(self, X) -> np.ndarray:
        """Return predicted probabilities, shape (n, 2); column 1 = P(y=1)."""
