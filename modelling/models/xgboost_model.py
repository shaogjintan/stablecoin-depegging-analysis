"""XGBoost estimator with native missing-value handling.

The maintained runner supplies fixed settings from training/parameters.py:
300 trees, depth 3, and learning rate 0.05. It supplies no validation set to
fit(), so both practice and final models use fixed rounds without early
stopping. The class retains optional validation-based early stopping for
compatibility with historical callers. Automatic class weighting uses only
the labels passed to fit(). Reported average precision is calculated by the
evaluation code; the estimator's aucpr evaluation setting is unchanged.
"""
from __future__ import annotations

import numpy as np
from xgboost import XGBClassifier

from .base import BaseModel


class XGBModel(BaseModel):
    name = "xgboost"
    needs_imputation = False
    supports_early_stopping = True

    def __init__(self, learning_rate: float = 0.1, max_depth: int = 5,
                 min_child_weight: float = 1, scale_pos_weight="auto",
                 n_estimators: int = 1000, early_stopping_rounds: int = 100,
                 random_state: int = 42, n_jobs: int = -1) -> None:
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.scale_pos_weight = scale_pos_weight
        self.n_estimators = n_estimators
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.clf_: XGBClassifier | None = None
        self.scale_pos_weight_used_: float | None = None
        self.best_iteration_: int | None = None
        self.best_score_: float | None = None

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> "XGBModel":
        y = np.asarray(y_train, dtype=int)
        n_pos = int(y.sum())
        if self.scale_pos_weight == "auto":
            spw = (len(y) - n_pos) / n_pos if n_pos > 0 else 1.0
        else:
            spw = float(self.scale_pos_weight)

        use_es = X_val is not None
        n_estimators = max(int(self.n_estimators), 1)  # 0 trees predicts constants
        clf = XGBClassifier(
            tree_method="hist",
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_child_weight=self.min_child_weight,
            scale_pos_weight=spw,
            n_estimators=n_estimators,
            early_stopping_rounds=self.early_stopping_rounds if use_es else None,
            eval_metric="aucpr",
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
        eval_set = [(X_val, y_val)] if use_es else None
        clf.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        self.clf_ = clf
        self.scale_pos_weight_used_ = spw
        if use_es:
            self.best_iteration_ = int(clf.best_iteration)
            self.best_score_ = float(clf.best_score)
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self.clf_ is None:
            raise RuntimeError("XGBModel.fit must be called before predict_proba")
        return self.clf_.predict_proba(X)

    def get_params(self) -> dict:
        return {
            "name": self.name,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "min_child_weight": self.min_child_weight,
            "scale_pos_weight": self.scale_pos_weight_used_,
            "n_estimators": self.n_estimators,
            "early_stopping_rounds": self.early_stopping_rounds,
            "best_iteration": self.best_iteration_,
            "inner_val_aucpr": self.best_score_,
        }
