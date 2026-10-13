"""
Naive baselines for the report table and pipeline smoke tests.

ConstantPriorModel: predicts the training-fold positive rate for every row.
This is the "constant prior" baseline from the Phase 3/4 plan -- a model that
learns nothing but the base rate; its pooled OOS PR-AUC equals the positive
rate and is the floor any real model must beat. It also doubles as the smoke
model for `modelling/train.py` (trivial to fit, NaN-native).
"""
from __future__ import annotations

import numpy as np

from .base import BaseModel


class ConstantPriorModel(BaseModel):
    name = "baseline"
    needs_imputation = False
    param_grid = [{}]

    def __init__(self) -> None:
        self.positive_rate_: float | None = None

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> "ConstantPriorModel":
        self.positive_rate_ = float(np.mean(np.asarray(y_train, dtype=float)))
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self.positive_rate_ is None:
            raise RuntimeError("ConstantPriorModel.fit must be called before predict_proba")
        p = self.positive_rate_
        return np.full((len(X), 2), [1.0 - p, p])

    def get_params(self) -> dict:
        return {"name": self.name, "positive_rate": self.positive_rate_}
