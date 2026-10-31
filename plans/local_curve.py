"""LOCAL-ONLY budget sweep for the P18 recall-vs-budget curve.

Mirrors modelling/rolling_threshold.py's protocol exactly (fit on
inner_train, threshold on inner_val, apply to test year) but fits each
fold ONCE and then sweeps false-alarm budgets {0.5, 1, 2, 4, 8} FA per
observed coin-month. Reuses rolling_threshold's private helpers — no
existing code is modified.

Usage: python plans/local_curve.py [--horizons 24 6] [--tiers 1 2]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "data_processing"))

from config import OOS_END_YEAR, OOS_START_YEAR, PROCESSED_DATA_DIR  # noqa: E402
from modelling.alarm_events import build_alarm_events  # noqa: E402
from modelling.prepare_data import load_folds  # noqa: E402
from modelling.rolling_threshold import (  # noqa: E402
    _choose_threshold,
    _fold_model_params,
    _info_path,
    _observed_coin_months,
    _targetable_episode_ids,
    _validation_and_test_scores,
    _validation_prediction_frame,
)

MODELS = [
    ("xgboost", "xgb_d3_lr005_300"),
    ("random_forest", "rf_500_d8_l50"),
    ("logreg", "logreg_c0001_l1"),
    ("persistence", "pers_fixed"),
    ("baseline", "base_fixed"),
]
BUDGETS = [0.5, 1.0, 2.0, 4.0, 8.0]


def sweep_cell(model: str, tag: str, tier: int, horizon: int,
               model_dir: Path, episodes: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    folds = load_folds(horizon=horizon, tier=tier)
    folds_by_year = {f["year"]: f for f in folds}
    # Fit once per fold (protocol B) and keep score frames.
    frames: dict[int, dict] = {}
    for year in range(OOS_START_YEAR, OOS_END_YEAR + 1):
        if year not in folds_by_year:
            continue
        fold = folds_by_year[year]
        info_path = _info_path(model, tier, horizon, year, model_dir)
        if not info_path.exists():
            raise FileNotFoundError(f"Missing {info_path} — train {model} t{tier} h{horizon} first")
        run_info = json.loads(info_path.read_text())
        if run_info.get("tag") != tag:
            raise ValueError(f"Fold {year} tag {run_info.get('tag')!r} != {tag!r}")
        params = _fold_model_params(run_info, model)
        val_scores, test_scores = _validation_and_test_scores(fold, model, params)
        validation = _validation_prediction_frame(fold, val_scores, horizon)
        test = fold["test_meta"].copy()
        test["hour"] = pd.to_datetime(test["hour"], utc=True)
        test["horizon"] = horizon
        test["split"] = "test"
        test["y_true"] = fold["y_test"].reset_index(drop=True)
        test["y_score"] = test_scores
        frames[year] = {"validation": validation, "test": test}

    rows: list[dict] = []
    per_year: list[dict] = []
    for budget in BUDGETS:
        warned, targetable = 0, 0
        false_alarms, coin_months = 0, 0
        val_recalls, thresholds = [], []
        for year, fr in frames.items():
            thr, val_stats = _choose_threshold(fr["validation"], episodes, horizon, budget)
            thresholds.append(thr)
            val_recalls.append(val_stats["episode_recall"])
            test = fr["test"]
            _, events = build_alarm_events(test, episodes, thr)
            t_ids = _targetable_episode_ids(test, episodes, horizon)
            caught = set(events.loc[events["is_true_alarm"], "target_episode_id"]
                         .dropna().astype(str)) if not events.empty else set()
            warned += len(caught & t_ids)
            targetable += len(t_ids)
            false_alarms += int((~events["is_true_alarm"]).sum()) if not events.empty else 0
            coin_months += _observed_coin_months(test)
            if budget == 1.0:
                per_year.append({
                    "model": model, "tier": tier, "horizon": horizon,
                    "test_year": year, "threshold": thr,
                    "test_episodes_targetable": len(t_ids),
                    "test_episodes_warned": len(caught & t_ids),
                    "test_episode_recall": len(caught & t_ids) / len(t_ids) if t_ids else np.nan,
                    "test_false_alarms": int((~events["is_true_alarm"]).sum()) if not events.empty else 0,
                    "test_coin_months": _observed_coin_months(test),
                })
        rows.append({
            "model": model, "tag": tag, "tier": tier, "horizon": horizon,
            "budget_fa_per_coin_month": budget,
            "pooled_test_episode_recall": warned / targetable if targetable else np.nan,
            "pooled_test_warned": warned,
            "pooled_test_targetable": targetable,
            "pooled_test_fa_per_coin_month": false_alarms / coin_months if coin_months else np.nan,
            "pooled_validation_recall": float(np.mean(val_recalls)) if val_recalls else np.nan,
            "min_threshold": min(thresholds) if thresholds else np.nan,
            "max_threshold": max(thresholds) if thresholds else np.nan,
        })
    return rows, per_year


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--horizons", type=int, nargs="+", default=[24, 6])
    ap.add_argument("--tiers", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--model-dir", default=str(REPO / "results/models"))
    ap.add_argument("--out", default=str(REPO / "reports/local_budget_curve.csv"))
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    episodes = pd.read_parquet(PROCESSED_DATA_DIR / "depeg_episodes.parquet").copy()
    episodes["start"] = pd.to_datetime(episodes["start"], utc=True)

    all_rows: list[dict] = []
    per_year_rows: list[dict] = []
    for model, tag in MODELS:
        for tier in args.tiers:
            for horizon in args.horizons:
                print(f"sweeping {model} t{tier} h{horizon} ...")
                rows, per_year = sweep_cell(model, tag, tier, horizon, model_dir, episodes)
                all_rows.extend(rows)
                per_year_rows.extend(per_year)
    out = pd.DataFrame(all_rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Saved {args.out}")
    print(out.to_string(index=False))

    py_out = pd.DataFrame(per_year_rows)
    py_path = Path(args.out).with_name("local_budget_per_year.csv")
    py_out.to_csv(py_path, index=False)
    print(f"Saved {py_path}")

    # Plot headline curve: h=24, tier 1 & 2, recall vs budget.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    colors = {"xgboost": "#E8A33D", "random_forest": "#4C9F70",
              "logreg": "#5B8DEF", "persistence": "#9AA0A6",
              "baseline": "#C9CCD1"}
    for tier in args.tiers:
        fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
        for model, _tag in MODELS:
            sub = out[(out.model == model) & (out.tier == tier) & (out.horizon == 24)]
            if sub.empty:
                continue
            ax.plot(sub.budget_fa_per_coin_month,
                    sub.pooled_test_episode_recall * 100,
                    marker="o", label=model, color=colors[model])
        ax.axvline(1.0, color="k", ls=":", alpha=0.6)
        ax.text(1.02, 2, "<=1 FA/coin-month", fontsize=8, rotation=90, va="bottom")
        ax.set_xlabel("False-alarm budget (FA per coin-month)")
        ax.set_ylabel("Pooled out-of-sample episode recall (%)")
        ax.set_title(f"h = 24, Tier {tier} - recall vs false-alarm budget")
        ax.set_xscale("log"); ax.set_xticks(BUDGETS)
        ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
        ax.grid(alpha=0.3); ax.legend()
        fig.tight_layout()
        png = Path(args.out).with_name(f"local_budget_curve_t{tier}_h24.png")
        fig.savefig(png); plt.close(fig)
        print(f"Saved {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
