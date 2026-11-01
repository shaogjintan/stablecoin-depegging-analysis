"""
Naive baselines for the report table and pipeline smoke tests.

ConstantPriorModel: predicts the training-fold positive rate for every row.
This is the "constant prior" baseline from the Phase 3/4 plan -- a model that
learns nothing but the base rate. AP equals prevalence within a single fold;
pooled AP can differ because the training prior changes across years.
It also doubles as the smoke
model for `modelling/train.py` (trivial to fit, NaN-native).

PersistenceModel: the "peg stress persists" benchmark -- score = -peg_deviation
(current distance below $1, clipped at 0). It ranks hours by how stressed the
peg already is, with no learning and no on-chain data (tier-independent).
Brier is not meaningful for it (scores live in [0, 0.01]); PR-AUC / ROC-AUC
rankings are what it contributes to the comparison table.
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


class PersistenceModel(BaseModel):
    name = "persistence"
    needs_imputation = False
    param_grid = [{}]

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> "PersistenceModel":
        return self

    def predict_proba(self, X) -> np.ndarray:
        dev = np.asarray(X["peg_deviation"], dtype=float)
        p = np.clip(-dev, 0.0, 0.01)  # stress = distance below $1 (at-risk rows: <= 0.01)
        return np.column_stack([1.0 - p, p])

    def get_params(self) -> dict:
        return {"name": self.name, "score": "-peg_deviation clipped to [0, 0.01]"}
