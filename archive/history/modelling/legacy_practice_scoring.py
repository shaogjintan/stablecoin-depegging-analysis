"""Unused historical practice-score helpers retained for reference."""
from pathlib import Path
import numpy as np
from modelling.models import get_model
from modelling.models.base import MedianImputer

def _info_path(model: str, tier: int, horizon: int, year: int,
               model_dir: Path) -> Path:
    stem = f"{model}_t{tier}_h{horizon}_{year}"
    return model_dir / f"{stem}.json"


def _fold_model_params(run_info: dict, model_name: str) -> dict:
    """Recover the selected constructor parameters from train.py's run log."""
    if not run_info.get("tune"):
        return dict(run_info.get("model_params") or {})
    log = run_info.get("tune_log") or {}
    metric = log.get("tune_metric") or run_info.get("tune_metric")
    combos = log.get("combos", [])
    score_col = f"inner_{metric}"
    scored = [entry for entry in combos if entry.get(score_col) is not None]
    if not scored:
        raise ValueError(
            f"Tuned run log has no candidate scores for {model_name} ({metric})"
        )
    best = max(scored, key=lambda entry: float(entry[score_col]))
    params = dict(best["params"])
    best_iteration = best.get("best_iteration")
    if best_iteration is not None:
        params["n_estimators"] = max(int(best_iteration) + 1, 1)
    return params


def _validation_and_test_scores(
    fold: dict, model_name: str, params: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Fit once on inner_train and score inner_val and test with that model."""
    model = get_model(model_name, **params)
    X_train = fold["X_inner_train"]
    X_val = fold["X_inner_val"]
    X_test = fold["X_test"]
    imp = None
    if model.needs_imputation:
        imp = MedianImputer().fit(X_train)
        X_train = imp.transform(X_train)
        X_val = imp.transform(X_val)
        X_test = imp.transform(X_test)
    model.fit(
        X_train,
        fold["y_inner_train"],
        X_val=X_val,
        y_val=fold["y_inner_val"],
    )
    val_scores = model.predict_proba(X_val)[:, 1]
    test_scores = model.predict_proba(X_test)[:, 1]
    return val_scores, test_scores


def _validation_prediction_frame(fold: dict, scores: np.ndarray,
                                 horizon: int) -> pd.DataFrame:
    frame = fold["inner_val_meta"].copy()
    frame["horizon"] = horizon
    frame["y_true"] = fold["y_inner_val"].reset_index(drop=True)
    frame["y_score"] = scores
    return frame

