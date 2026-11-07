"""Logistic estimator with training-fitted feature transforms and scaling.

The study protocol selects C for L1 regression in training/logistic_tuning.py.
The public L2 constructor default is retained for existing model compatibility;
the maintained training pipeline explicitly requests L1.
"""
import inspect
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .base import BaseModel


class LogRegModel(BaseModel):
    """Fit logistic regression with training-fitted feature transforms."""
    name = "logreg"
    needs_imputation = True  # The training pipeline fits a median imputer.


    def __init__(self, C=0.1, penalty="l2", class_weight="balanced", winsor=0.01):
        """Store logistic regularisation and preprocessing settings."""
        self.C = C
        self.penalty = penalty
        self.class_weight = class_weight
        self.winsor = winsor

    def _add_feats(self, X):
        """Add peg-deviation magnitude and volatility interaction features."""
        X = pd.DataFrame(X).copy()
        if "peg_deviation" in X.columns:
            X["abs_peg_dev"] = X["peg_deviation"].abs()
            if "realised_vol_6h" in X.columns:
                X["dev_x_vol6"] = X["peg_deviation"] * X["realised_vol_6h"]
        return X

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Fit feature transforms, scaling, and logistic coefficients."""
        X = self._add_feats(X_train)
        self.cols_ = list(X.columns)
        # winsorise using TRAIN quantiles only (no leakage)
        self.lo_ = X.quantile(self.winsor)
        self.hi_ = X.quantile(1 - self.winsor)
        X = X.clip(self.lo_, self.hi_, axis=1)

        self.scaler_ = StandardScaler().fit(X)
        # scikit-learn 1.8 deprecated `penalty` in favor of `l1_ratio`.
        # Translate the existing public penalty parameter so saved configs and
        # older scikit-learn releases continue to work without warnings.
        lr_kwargs = {
            "C": self.C,
            "solver": "liblinear",
            "class_weight": self.class_weight,
            "max_iter": 2000,
            "random_state": 42,
        }
        penalty_default = inspect.signature(LogisticRegression).parameters[
            "penalty"
        ].default
        if penalty_default == "deprecated":
            lr_kwargs["l1_ratio"] = 1.0 if self.penalty == "l1" else 0.0
        else:
            lr_kwargs["penalty"] = self.penalty
        self.clf_ = LogisticRegression(**lr_kwargs)
        self.clf_.fit(self.scaler_.transform(X), np.asarray(y_train))
        return self

    def predict_proba(self, X):
        """Return negative- and positive-class scores for the supplied rows."""
        X = self._add_feats(X)[self.cols_]
        X = X.clip(self.lo_, self.hi_, axis=1)
        return self.clf_.predict_proba(self.scaler_.transform(X))

    def get_params(self):
        """Return estimator settings and fitted metadata for run recording."""
        return {"name": self.name, "C": self.C, "penalty": self.penalty,
                "class_weight": self.class_weight, "winsor": self.winsor}
