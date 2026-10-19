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

With --tune: nested-CV pattern -- grid-search `model.param_grid` with
candidates fit on X_inner_train and early stopping on X_inner_val (held
out), selected by inner-val PR-AUC; the winner is refit on the full X_train
(n_estimators = best_iteration for early-stopping models), params logged.
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
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from modelling.evaluate import (  # noqa: E402
    evaluate_run,
    pooled_metrics,
    print_report,
    select_threshold,
)
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


def _candidate_score(tune_metric: str, y_val, p_val, beta: float,
                     max_alarm_rate: float | None) -> tuple[float, dict]:
    """Score a tuning candidate on inner-val predictions.

    tune_metric:
      - "pr_auc":             pooled PR-AUC (original behaviour)
      - "f_beta":             best F-beta over thresholds (beta weights recall
                              beta^2 times precision)
      - "recall_at_rate":     max hour-level recall subject to alarm rate <=
                              max_alarm_rate (early-warning objective)
    """
    if tune_metric == "pr_auc":
        return pooled_metrics(y_val, p_val)["pr_auc"], {}
    if tune_metric == "f_beta":
        th = select_threshold(y_val, p_val, criterion="f_beta", beta=beta)
        p, r = th["precision"], th["recall"]
        if np.isnan(p) or np.isnan(r) or p + r == 0:
            return -np.inf, {}
        fb = (1 + beta**2) * p * r / (beta**2 * p + r)
        return float(fb), {"threshold": th["threshold"]}
    if tune_metric == "recall_at_rate":
        th = select_threshold(y_val, p_val, criterion="recall_at_alarm_rate",
                              max_alarm_rate=max_alarm_rate)
        return float(th["recall"]), {"threshold": th["threshold"],
                                     "alarm_rate": th["alarm_rate"]}
    raise ValueError(f"unknown tune metric '{tune_metric}'")


def _tune_fit(get_model_fn, name, X_inner_train, y_inner_train, X_val, y_val,
              tune_metric: str = "pr_auc", beta: float = 5,
              max_alarm_rate: float | None = None):
    """Nested-CV tuning: grid over model.param_grid, candidates fit on the
    inner training set with early stopping on inner_val (genuinely held out),
    selected by the tune metric (inner-val PR-AUC, F-beta, or recall at an
    alarm-rate budget). Returns (best_combo, log, best_iteration).

    best_iteration is the winning candidate's early-stopping iteration for
    early-stopping models; the caller refits the winner on the full X_train
    with n_estimators=best_iteration, so the final model never early-stops
    on rows it was trained on."""
    best = None
    best_iter = None
    log = []
    for combo in get_model_fn(name).param_grid:
        cand, imp = _fit_one(get_model_fn(name, **combo),
                             X_inner_train, y_inner_train, X_val, y_val)
        p_val = _predict(cand, imp, X_val)
        score, extra = _candidate_score(tune_metric, y_val, p_val, beta,
                                        max_alarm_rate)
        log.append({"params": combo, f"inner_{tune_metric}": float(score),
                    "best_iteration": getattr(cand, "best_iteration_", None),
                    **extra})
        if best is None or score > best[0]:
            best = (score, combo)
            best_iter = getattr(cand, "best_iteration_", None)
    best_score, best_combo = best
    print(f"    tuned: {len(log)} combos -> best inner-val {tune_metric} "
          f"{best_score:.4f} params {best_combo} (best_iteration={best_iter})")
    return best_combo, log, best_iter


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True,
                        help="registered model name (see modelling/models/__init__.py)")
    parser.add_argument("--horizon", type=int, choices=[1, 6, 24], default=24)
    parser.add_argument("--tier", type=int, choices=[1, 2], default=1)
    parser.add_argument("--tag", default=None, help="run tag for outputs (default: auto)")
    parser.add_argument("--model-params", default=None,
                        help="JSON object of fixed model parameters; cannot be combined with --tune")
    parser.add_argument("--model-dir", default=str(MODEL_DIR),
                        help="directory for per-fold model artifacts")
    parser.add_argument("--pred-dir", default=str(PRED_DIR),
                        help="directory for per-fold and pooled prediction files")
    parser.add_argument("--evaluation-dir", default="results/evaluation",
                        help="directory for evaluation curve outputs")
    parser.add_argument("--tune", action="store_true",
                        help="grid-search model.param_grid on the inner-val tune metric")
    parser.add_argument("--tune-metric", default="pr_auc",
                        choices=["pr_auc", "f_beta", "recall_at_rate"],
                        help="inner-val candidate scoring metric (default pr_auc)")
    parser.add_argument("--beta", type=float, default=5,
                        help="beta for tune-metric f_beta (recall weight)")
    parser.add_argument("--max-alarm-rate", type=float, default=0.01,
                        help="alarm-rate budget for tune-metric recall_at_rate")
    args = parser.parse_args(argv)

    try:
        fixed_params = json.loads(args.model_params) if args.model_params else {}
    except json.JSONDecodeError as exc:
        parser.error(f"--model-params must be valid JSON: {exc}")
    if not isinstance(fixed_params, dict):
        parser.error("--model-params must decode to a JSON object")
    if args.tune and fixed_params:
        parser.error("--model-params cannot be combined with --tune")

    tag = args.tag or f"{args.model}_t{args.tier}_h{args.horizon}"
    pred_dir, model_dir = Path(args.pred_dir), Path(args.model_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    folds = load_folds(horizon=args.horizon, tier=args.tier)
    print(f"model={args.model} horizon={args.horizon} tier={args.tier} tag={tag} "
        f"tune={args.tune} fixed_params={fixed_params} | {len(folds)} folds | "
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
            best_combo, tune_log, best_iter = _tune_fit(
                get_model, args.model,
                fold["X_inner_train"], fold["y_inner_train"], X_val, y_val,
                tune_metric=args.tune_metric, beta=args.beta,
                max_alarm_rate=args.max_alarm_rate if args.tune_metric == "recall_at_rate" else None)
            tune_log_all.append({"year": year, "tune_metric": args.tune_metric,
                                 "combos": tune_log})
            if best_iter is not None:  # early-stopping model: refit without eval set
                # best_iteration is 0-based; a 0-tree model predicts constants
                best_combo = {**best_combo, "n_estimators": max(int(best_iter) + 1, 1)}
                refit_val, refit_yval = None, None
            else:
                refit_val, refit_yval = X_val, y_val
        else:
            best_combo = fixed_params
            refit_val, refit_yval = X_val, y_val

        model, imp = _fit_one(get_model(args.model, **best_combo),
                              X_train, y_train, X_val=refit_val, y_val=refit_yval)

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
            pred_dir / f"{args.model}_t{args.tier}_h{args.horizon}_{year}.parquet",
            index=False)

        # model artifact + params log
        artifact = model_dir / f"{args.model}_t{args.tier}_h{args.horizon}_{year}"
        joblib.dump(model, artifact.with_suffix(".joblib"))
        run_info = {
            "model": args.model, "tier": args.tier, "horizon": args.horizon,
            "year": year, "tag": tag,
            "params": model.get_params(),
            "tune": args.tune,
            "model_params": best_combo,
            "tune_metric": args.tune_metric if args.tune else None,
            "beta": args.beta if args.tune_metric == "f_beta" else None,
            "max_alarm_rate": args.max_alarm_rate if args.tune_metric == "recall_at_rate" else None,
            "tune_log": tune_log_all[-1] if args.tune else None,
            "inner_val_pr_auc": float(inner_pr),
        }
        artifact.with_suffix(".json").write_text(json.dumps(run_info, indent=2, default=str))

    # pool OOS test predictions across folds and evaluate
    pooled = pd.concat(test_parts, ignore_index=True)
    pooled_path = pred_dir / f"{args.model}_t{args.tier}_h{args.horizon}_{tag}_pooled_test.parquet"
    pooled.to_parquet(pooled_path, index=False)
    res = evaluate_run(pooled, tag=tag, out_dir=args.evaluation_dir)
    print("\n" + "=" * 78)
    print(f"Run {tag} complete. Pooled OOS evaluation:")
    print_report(res)
    print(f"\nPooled test predictions: {pooled_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
