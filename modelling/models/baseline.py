"""Naive baselines for the report table and pipeline smoke tests.

ConstantPriorModel: predicts the training-fold positive rate for every row.
This model learns only the training-label base rate. AP equals the evaluation
positive rate when all evaluation scores are tied (in particular, within a
fold). Pooled yearly predictions can have different training priors, so pooled
AP need not equal the pooled positive rate and is not a universal AP floor.
Benchmark comparisons are exported by modelling.reporting.benchmarks.

PersistenceModel: the "peg stress persists" benchmark -- score = -peg_deviation
(min(max(1 - price, 0), 0.01)). It ranks completed bars by how stressed the
peg already is, with no learning and no on-chain inputs. Its evaluation
sample is still tier-dependent.
Its scores are ranking signals rather than calibrated probabilities;
average precision measures its ranking performance.
"""
from __future__ import annotations

import numpy as np

from .base import BaseModel


class ConstantPriorModel(BaseModel):
    """Predict the positive-label rate learned from the training partition."""
    name = "baseline"
    needs_imputation = False

    def __init__(self) -> None:
        """Initialize the unfitted constant-prior estimator."""
        self.positive_rate_: float | None = None

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> "ConstantPriorModel":
        """Learn the positive-label rate from the training partition."""
        self.positive_rate_ = float(np.mean(np.asarray(y_train, dtype=float)))
        return self

    def predict_proba(self, X) -> np.ndarray:
        """Return negative- and positive-class scores for the supplied rows."""
        if self.positive_rate_ is None:
            raise RuntimeError("ConstantPriorModel.fit must be called before predict_proba")
        p = self.positive_rate_
        return np.full((len(X), 2), [1.0 - p, p])

    def get_params(self) -> dict:
        """Return estimator settings and fitted metadata for run recording."""
        return {"name": self.name, "positive_rate": self.positive_rate_}


class PersistenceModel(BaseModel):
    """Rank coin-hours by their current downward peg deviation."""
    name = "persistence"
    needs_imputation = False

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> "PersistenceModel":
        """Accept training inputs without learning estimator parameters."""
        return self

    def predict_proba(self, X) -> np.ndarray:
        """Return downward peg-stress scores in two-column classifier form."""
        dev = np.asarray(X["peg_deviation"], dtype=float)
        p = np.clip(-dev, 0.0, 0.01)  # stress = distance below $1 (at-risk rows: <= 0.01)
        return np.column_stack([1.0 - p, p])

    def get_params(self) -> dict:
        """Return estimator settings and fitted metadata for run recording."""
        return {"name": self.name, "score": "-peg_deviation clipped to [0, 0.01]"}
