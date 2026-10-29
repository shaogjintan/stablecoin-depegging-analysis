"""LOCAL-ONLY driver: real-time vs ex-post threshold comparison (prof's request).

Mirrors plans/local_curve.py exactly (fit on inner_train once per fold,
threshold on inner_val for the real-time protocol) and adds the prof's
ex-post comparison: for each test year, the threshold is instead chosen on
the TEST year's own scores at the same <= 1 false alarm per observed
coin-month budget. Ex-post thresholds use test-year information, so a live
system could not do this -- it exists only to make the market-only vs
market+on-chain comparison fair. Both modes report episode recall,
false alarms per coin-month (realised vs the 1.0 budget), and median lead
time over true alarms. Reuses rolling_threshold's private helpers -- no
existing code is modified.

Usage: python plans/ex_post_comparison.py            # all models, h6/h24
       python plans/ex_post_comparison.py --horizons 24 --tiers 2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

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

# Fixed-config runs (prof's settings) -- the only protocol used on the deck.
MODELS = [
    ("xgboost", "xgb_d3_lr005_300"),
    ("random_forest", "rf_500_d8_l50"),
    ("logreg", "logreg_c0001_l1"),
    ("persistence", "pers_fixed"),
    ("baseline", "base_fixed"),
]
BUDGET = 1.0  # false alarms per observed coin-month, on average


def _mode_cell(model: str, tag: str, tier: int, horizon: int,
               model_dir: Path, episodes: pd.DataFrame,
               mode: str) -> tuple[list[dict], list[dict]]:
    """One mode ('real_time' | 'ex_post') for one model/tier/horizon."""
    folds = load_folds(horizon=horizon, tier=tier)
    folds_by_year = {f["year"]: f for f in folds}
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

    per_year: list[dict] = []
    pooled_pr_auc, pooled_base_rate = [], []
    warned_all, targetable_all, fa_all, cm_all = 0, 0, 0, 0
    leads: list[float] = []
    thresholds = []
    for year, fr in frames.items():
        select_frame = fr["validation"] if mode == "real_time" else fr["test"]
        thr, _stats = _choose_threshold(select_frame, episodes, horizon, BUDGET)
        thresholds.append(thr)
        test = fr["test"]
        _, events = build_alarm_events(test, episodes, thr)
        t_ids = _targetable_episode_ids(test, episodes, horizon)
        caught = (set(events.loc[events["is_true_alarm"], "target_episode_id"]
                      .dropna().astype(str)) if not events.empty else set())
        fa = int((~events["is_true_alarm"]).sum()) if not events.empty else 0
        cm = _observed_coin_months(test)
        warned = len(caught & t_ids)
        tgt = len(t_ids)
        true_events = events[events["is_true_alarm"]] if not events.empty else events
        leads.extend(true_events["lead_hours"].dropna().tolist())
        warned_all += warned; targetable_all += tgt; fa_all += fa; cm_all += cm
        pooled_pr_auc.append(average_precision_score(test["y_true"], test["y_score"]))
        pooled_base_rate.append(float(test["y_true"].mean()))
        per_year.append({
            "model": model, "tier": tier, "horizon": horizon, "mode": mode,
            "test_year": year, "threshold": thr,
            "false_alarms": fa, "coin_months": cm,
            "fa_per_coin_month": fa / cm if cm else np.nan,
            "episodes_targetable": tgt, "episodes_warned": warned,
            "episode_recall": warned / tgt if tgt else np.nan,
            "median_lead_hours": (
                float(np.median(true_events["lead_hours"].dropna()))
                if not true_events.empty else np.nan),
        })
    # pooled PR-AUC / base rate over all test rows of the mode (scores are
    # mode-independent; averaged per fold and pooled the same way local_curve
    # pools recall)
    pooled = {
        "model": model, "tier": tier, "horizon": horizon, "mode": mode,
        "pr_auc": float(np.mean(pooled_pr_auc)) if pooled_pr_auc else np.nan,
        "base_rate": float(np.mean(pooled_base_rate)) if pooled_base_rate else np.nan,
        "budget_fa_per_coin_month": BUDGET,
        "realised_fa_per_coin_month": fa_all / cm_all if cm_all else np.nan,
        "false_alarms": fa_all, "coin_months": cm_all,
        "episode_recall": warned_all / targetable_all if targetable_all else np.nan,
        "episodes_warned": warned_all, "episodes_targetable": targetable_all,
        "median_lead_hours": float(np.median(leads)) if leads else np.nan,
        "min_threshold": min(thresholds) if thresholds else np.nan,
        "max_threshold": max(thresholds) if thresholds else np.nan,
    }
    return [pooled], per_year


def _plot(summary: pd.DataFrame, path: Path) -> None:
    """Recall (real-time vs ex-post) per model, one panel per tier at h=24."""
    df = summary[summary["horizon"] == 24].copy()
    order = [m for m, _ in MODELS]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, tier in zip(axes, (1, 2)):
        d = df[df["tier"] == tier]
        x = np.arange(len(order))
        w = 0.38
        rt = [d.loc[(d.model == m) & (d.mode == "real_time"), "episode_recall"].iloc[0] for m in order]
        ex = [d.loc[(d.model == m) & (d.mode == "ex_post"), "episode_recall"].iloc[0] for m in order]
        ax.bar(x - w / 2, rt, w, label="real-time (prior-year threshold)", color="#2a78d6")
        ax.bar(x + w / 2, ex, w, label="ex post (test-year threshold)", color="#eb6834")
        for i, (r, e) in enumerate(zip(rt, ex)):
            ax.text(i, max(r, e) + 0.01, f"{r:.2f}", ha="center", fontsize=8, color="#2a78d6")
            ax.text(i, max(r, e) + 0.07, f"{e:.2f}", ha="center", fontsize=8, color="#eb6834")
        ax.set_xticks(x, [m.replace("_", " ") for m in order], fontsize=8)
        ax.set_title(f"Tier {tier}", loc="left")
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("episode recall @ 1 FA/coin-month (h=24, pooled OOS)")
    fig.suptitle("Real-time vs ex-post thresholds: how much recall the same budget buys "
                 "when the threshold is chosen on the test year (not deployable live)",
                 x=0.01, ha="left", fontsize=10, fontweight="bold")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--horizons", type=int, nargs="+", default=[24, 6])
    ap.add_argument("--tiers", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--model-dir", default=str(REPO / "results/models"))
    ap.add_argument("--out", default=str(REPO / "reports/ex_post_comparison.csv"))
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    episodes = pd.read_parquet(PROCESSED_DATA_DIR / "depeg_episodes.parquet").copy()
    episodes["start"] = pd.to_datetime(episodes["start"], utc=True)

    pooled_rows: list[dict] = []
    per_year_rows: list[dict] = []
    for model, tag in MODELS:
        for tier in args.tiers:
            for horizon in args.horizons:
                if model == "xgboost" and horizon == 1:
                    continue  # h=1 handled separately below (fixed h1 runs exist)
                for mode in ("real_time", "ex_post"):
                    print(f"{model} t{tier} h{horizon} {mode} ...", flush=True)
                    pooled, per_year = _mode_cell(
                        model, tag, tier, horizon, model_dir, episodes, mode)
                    pooled_rows.extend(pooled)
                    per_year_rows.extend(per_year)
    # h=1 near-detection line (XGBoost only, both tiers)
    for tier in args.tiers:
        if tier not in (1, 2):
            continue
        for mode in ("real_time", "ex_post"):
            print(f"xgboost t{tier} h1 {mode} ...", flush=True)
            pooled, per_year = _mode_cell("xgboost", "xgb_d3_lr005_300",
                                          tier, 1, model_dir, episodes, mode)
            pooled_rows.extend(pooled)
            per_year_rows.extend(per_year)

    summary = pd.DataFrame(pooled_rows)
    per_year = pd.DataFrame(per_year_rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False)
    per_year.to_csv(out.with_name("ex_post_comparison_per_year.csv"), index=False)
    print(f"\nSaved {out}")
    print(summary.round(4).to_string(index=False))
    fig = out.with_name("ex_post_vs_real_time_h24.png")
    if 24 in args.horizons:
        _plot(summary, fig)
        print(f"Saved {fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
