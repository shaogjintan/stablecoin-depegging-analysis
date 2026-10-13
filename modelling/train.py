"""
Expanding-window training orchestrator (Phase 3).

CLI (run from the repo root):
    python modelling/train.py --model baseline --horizon 24 --tier 1 --tag smoke
    python modelling/train.py --model xgboost --horizon 24 --tier 2 --tag run01 --tune

Per fold (2021..2025):
  1. fresh model instance from modelling/models registry;
  2. median-imputation on X_train if the model needs it (fitted on train only);
  3. fit on X_train; NaN-native models may use X_inner_val for early stopping;
  4. predict X_test and X_inner_val; save per-fold predictions to
     results/predictions/{model}_{tier}_{h}_{year}.parquet and the model to
     results/models/ (joblib + params JSON); log inner-val PR-AUC.

With --tune: grid-search `model.param_grid` on inner-val PR-AUC, refit the
best configuration on the full training fold, and log the chosen params.
(Phase 4 defines the XGBoost grid; the plumbing lives here.)

Finally the test predictions are pooled across folds and passed to
modelling/evaluate.py (pooled PR-AUC / ROC-AUC / Brier, per-coin and
per-episode breakdowns, alarm-curve CSV).

Scratch outputs live under results/ (gitignored); final tables/figures go to
reports/ (see the Phase 3 roadmap results convention).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from modelling.evaluate import evaluate_run, pooled_metrics, print_report  # noqa: E402
from modelling.models import get_model  # noqa: E402
from modelling.prepare_data import load_folds  # noqa: E402

RESULTS_ROOT = REPO_ROOT / "results"
PRED_DIR = RESULTS_ROOT / "predictions"
MODEL_DIR = RESULTS_ROOT / "models"


def _fit_one(model, X_train, y_train, X_val=None, y_val=None):
    """Fit, applying train-only median imputation if the model needs it."""
    X_tr = X_train
    X_va = X_val
    imp = None
    if model.needs_imputation:
        from modelling.models.base import MedianImputer
        imp = MedianImputer().fit(X_train)
        X_tr = imp.transform(X_train)
        if X_val is not None:
            X_va = imp.transform(X_val)
    model.fit(X_tr, y_train, X_val=X_va, y_val=y_val)
    return model, imp


def _predict(model, imp, X):
    if imp is not None and X is not None:
        X = imp.transform(X)
    return model.predict_proba(X)[:, 1]


def _tune_fit(get_model_fn, name, X_train, y_train, X_val, y_val):
    """Grid over model.param_grid, select by inner-val PR-AUC, refit best."""
    best = None
    log = []
    for combo in get_model_fn(name).param_grid:
        cand, imp = _fit_one(get_model_fn(name, **combo), X_train, y_train, X_val, y_val)
        p_val = _predict(cand, imp, X_val)
        pr = pooled_metrics(y_val, p_val)["pr_auc"]
        log.append({"params": combo, "inner_pr_auc": float(pr)})
        if best is None or pr > best[0]:
            best = (pr, combo)
    best_pr, best_combo = best
    print(f"    tuned: {len(log)} combos -> best inner-val PR-AUC {best_pr:.4f} "
          f"params {best_combo}")
    return best_combo, log


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True,
                        help="registered model name (see modelling/models/__init__.py)")
    parser.add_argument("--horizon", type=int, choices=[1, 6, 24], default=24)
    parser.add_argument("--tier", type=int, choices=[1, 2], default=1)
    parser.add_argument("--tag", default=None, help="run tag for outputs (default: auto)")
    parser.add_argument("--tune", action="store_true",
                        help="grid-search model.param_grid on inner-val PR-AUC")
    args = parser.parse_args(argv)

    tag = args.tag or f"{args.model}_t{args.tier}_h{args.horizon}"
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    folds = load_folds(horizon=args.horizon, tier=args.tier)
    print(f"model={args.model} horizon={args.horizon} tier={args.tier} tag={tag} "
          f"tune={args.tune} | {len(folds)} folds | "
          f"features: {len(folds[0]['features'])}")

    test_parts, tune_log_all = [], []
    for fold in folds:
        year = fold["year"]
        X_train, y_train = fold["X_train"], fold["y_train"]
        X_val, y_val = fold["X_inner_val"], fold["y_inner_val"]
        X_test = fold["X_test"]
        print(f"\nFold {year}: train {len(X_train):,} rows / {y_train.sum():,} pos | "
              f"inner_val {len(X_val):,} / {y_val.sum():,} pos | "
              f"test {len(X_test):,} / {fold['y_test'].sum():,} pos")

        if args.tune:
            best_combo, tune_log = _tune_fit(get_model, args.model,
                                             X_train, y_train, X_val, y_val)
            tune_log_all.append({"year": year, "combos": tune_log})
        else:
            best_combo = {}

        model, imp = _fit_one(get_model(args.model, **best_combo),
                              X_train, y_train, X_val, y_val)

        p_val = _predict(model, imp, X_val)
        p_test = _predict(model, imp, X_test)
        inner_pr = pooled_metrics(y_val, p_val)["pr_auc"]
        print(f"    inner-val PR-AUC {inner_pr:.4f}")

        # per-fold predictions: test rows + inner-val rows, one file per fold
        parts = []
        for part_name, (meta, y_part, p_part) in {
            "test": (fold["test_meta"], fold["y_test"], p_test),
            "inner_val": (fold["inner_val_meta"], y_val, p_val),
        }.items():
            d = meta.copy()
            d["year"] = year
            d["horizon"] = args.horizon
            d["split"] = part_name
            d["y_true"] = y_part.reset_index(drop=True)
            d["y_score"] = p_part
            parts.append(d)
            if part_name == "test":
                test_parts.append(d)
        fold_preds = pd.concat(parts, ignore_index=True)
        fold_preds.to_parquet(
            PRED_DIR / f"{args.model}_t{args.tier}_h{args.horizon}_{year}.parquet",
            index=False)

        # model artifact + params log
        artifact = MODEL_DIR / f"{args.model}_t{args.tier}_h{args.horizon}_{year}"
        joblib.dump(model, artifact.with_suffix(".joblib"))
        run_info = {
            "model": args.model, "tier": args.tier, "horizon": args.horizon,
            "year": year, "tag": tag,
            "params": model.get_params(),
            "tune": args.tune,
            "tune_log": tune_log_all[-1] if args.tune else None,
            "inner_val_pr_auc": float(inner_pr),
        }
        artifact.with_suffix(".json").write_text(json.dumps(run_info, indent=2, default=str))

    # pool OOS test predictions across folds and evaluate
    pooled = pd.concat(test_parts, ignore_index=True)
    pooled_path = PRED_DIR / f"{args.model}_t{args.tier}_h{args.horizon}_{tag}_pooled_test.parquet"
    pooled.to_parquet(pooled_path, index=False)
    res = evaluate_run(pooled, tag=tag)
    print("\n" + "=" * 78)
    print(f"Run {tag} complete. Pooled OOS evaluation:")
    print_report(res)
    print(f"\nPooled test predictions: {pooled_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
