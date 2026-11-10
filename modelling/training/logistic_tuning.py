"""Training-only logistic regression fitting and regularisation selection."""
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score
from modelling.training.calendar_cv import _calendar_cv_folds
from modelling.training.parameters import DEFAULT_C_GRID
from modelling.models import get_model
from modelling.models.base import MedianImputer
from modelling.utils.io import save_json


def fit_lr(X, y, C):
    """Fit an L1 logistic pipeline and reject convergence failures."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        imputer = MedianImputer().fit(X)
        model = get_model("logreg", C=C, penalty="l1")
        model.fit(imputer.transform(X), y)
        return model, imputer

def select_c(candidates):
    """Select the highest mean-AP candidate, breaking ties by smaller C."""
    return max(candidates, key=lambda r: (r["mean_average_precision"], -r["C"]))["C"]

def tune_lr(train, features, episodes, folder, h):
    """Select C by training-only calendar CV and save the tuning evidence."""
    X, y = train[features], train.y_true
    folds, skipped = _calendar_cv_folds(X, y, train, episodes)
    save_json(folder / "cv_skipped.json", skipped)
    for row in skipped:
        print(f"{folder.name}: CV {row['validation_year']} skipped: {row['reason']}", flush=True)
    if not folds:
        raise RuntimeError(f"{folder.name}: no usable inner-training CV folds; no fallback C")
    candidates = []
    for f in folds:
        tr, va = f["train_idx"], f["val_idx"]
        a, b = train.iloc[tr], train.iloc[va]
        assert a.hour.max() + pd.Timedelta(hours=h + 1) < b.hour.min()
        d = folder / "cv" / str(f["validation_year"])
        d.mkdir(parents=True)
        a.to_parquet(d / "train.parquet", index=False)
        b.to_parquet(d / "validation.parquet", index=False)
        save_json(d / "split.json", {k: v for k, v in f.items() if not k.endswith("_idx")})
    for C in DEFAULT_C_GRID:
        per_year = []
        for f in folds:
            tr, va = f["train_idx"], f["val_idx"]
            model, imputer = fit_lr(X.iloc[tr], y.iloc[tr], C)
            scores = model.predict_proba(imputer.transform(X.iloc[va]))[:, 1]
            d = folder / "cv" / str(f["validation_year"])
            joblib.dump({"model": model, "imputer": imputer}, d / f"C{C}.joblib")
            pred = train.iloc[va][["coin", "hour", "y_true"]].copy()
            pred["y_score"] = scores
            pred.to_parquet(d / f"C{C}.parquet", index=False)
            per_year.append(dict(validation_year=f["validation_year"],
                                 average_precision=float(average_precision_score(y.iloc[va], scores)),
                                 n_iter=model.clf_.n_iter_.tolist()))
        candidates.append(dict(C=C, mean_average_precision=float(np.mean(
            [r["average_precision"] for r in per_year])), per_year=per_year))
    selected = select_c(candidates)
    tuning = dict(selected_C=selected, grid=list(DEFAULT_C_GRID), penalty="l1",
                  selection="unweighted mean AP across usable inner-training calendar years; ties smaller C",
                  candidates=candidates, skipped=skipped,
                  calibration_used_for_selection=False)
    save_json(folder / "tuning.json", tuning)
    return selected
