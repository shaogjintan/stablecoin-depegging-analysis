"""Frozen estimator settings used by the retained protocols."""
XGBOOST_FIXED_PARAMS = dict(max_depth=3, learning_rate=.05, n_estimators=300, n_jobs=4,
              min_child_weight=1, scale_pos_weight="auto", random_state=42)


RANDOM_FOREST_FIXED_PARAMS = {
    "n_estimators": 500, "max_depth": 8, "min_samples_leaf": 50,
    "max_features": "sqrt", "class_weight": "balanced_subsample",
    "random_state": 42, "n_jobs": 4,
}


# C is selected only inside the practice model's training partition.
DEFAULT_C_GRID = (0.001, 0.01, 0.1, 1.0)
