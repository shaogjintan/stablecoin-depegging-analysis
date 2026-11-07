"""Random forest estimator implementing the shared model interface.

The maintained runner supplies fixed settings from training/parameters.py:
500 trees, maximum depth 8, and minimum leaf size 50. It fits the pipeline's
median imputer on the training partition only. Validation arguments are
accepted by this class but unused. Constructor defaults remain unchanged
for compatibility; they are not the study's fixed training configuration.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .base import BaseModel


class RFModel(BaseModel):
    name = "random_forest"
    needs_imputation = True
    supports_early_stopping = False

    def __init__(self, n_estimators: int = 300, max_depth: int | None = 12,
                 min_samples_leaf: int = 20, max_features="sqrt",
                 class_weight="balanced_subsample", random_state: int = 42,
                 n_jobs: int = -1) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.class_weight = class_weight
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.clf_: RandomForestClassifier | None = None
        self.feature_names_: list[str] | None = None

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> "RFModel":
        # X_val / y_val are accepted for the contract but unused: no early stopping.
        clf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            class_weight=self.class_weight,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
        clf.fit(X_train, np.asarray(y_train, dtype=int))
        self.clf_ = clf
        self.feature_names_ = list(getattr(X_train, "columns", [])) or None
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self.clf_ is None:
            raise RuntimeError("RFModel.fit must be called before predict_proba")
        proba = self.clf_.predict_proba(X)
        if proba.shape[1] == 1:
            # Training fold had a single class; return P(y=1) explicitly.
            only = self.clf_.classes_[0]
            p1 = np.ones(len(proba)) if only == 1 else np.zeros(len(proba))
            return np.column_stack([1.0 - p1, p1])
        return proba

    @property
    def feature_importances_(self) -> np.ndarray:
        if self.clf_ is None:
            raise RuntimeError("RFModel.fit must be called first")
        return self.clf_.feature_importances_

    def get_params(self) -> dict:
        return {
            "name": self.name,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "class_weight": self.class_weight,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
        }