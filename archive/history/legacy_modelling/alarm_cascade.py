"""
Alarm cascade: use one h=24 model to alarm h=1 and h=6 depegs.

label_h24 >= label_h6 >= label_h1 (an onset within 1h/6h is also within 24h),
so a single model trained on the richest h=24 labels (738 OOS positives) can
power all three horizons: the same hourly score, re-thresholded per horizon.
This sidesteps the tiny-sample problem at h=1 (32 OOS events).

For each target horizon h, the h=24 scores are joined onto the h-eligible
rows (prediction_eligible_h{h}) on (coin, hour); rows right-censored at h=24
have no score, and coverage is reported. The episode recall vs alarm-rate
curve is then computed exactly like the dedicated models'.

Usage (from the repo root):
    python modelling/alarm_cascade.py --tier 2 --tag recall_tuned
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "data_processing"))

from splits import load_master  # noqa: E402
from modelling.evaluate import alarm_sweep, operating_points, per_episode_table  # noqa: E402

HORIZONS = (1, 6, 24)


def cascade_curve(scores: pd.DataFrame, master: pd.DataFrame,
                  episodes: pd.DataFrame, horizon: int, tier: int) -> pd.DataFrame:
    """Episode recall vs alarm-rate curve for horizon h using h=24 scores."""
    col = f"prediction_eligible_{'market' if tier == 1 else 'market_onchain'}_h{horizon}"
    if col not in master.columns:  # pre-PR-#13 masters
        col = f"prediction_eligible_h{horizon}"
    sub = master[master[col]].copy()
    sub = sub[["coin", "hour", f"label_h{horizon}", "episode_id"]]
    m = sub.merge(scores, on=["coin", "hour"], how="left")
    covered = m["y_score"].notna().mean()
    if covered == 0:
        raise RuntimeError("no overlapping scored rows for horizon", horizon)
    m = m.dropna(subset=["y_score"]).reset_index(drop=True)
    m["y_true"] = m[f"label_h{horizon}"].fillna(0).astype(int)

    per_ep = per_episode_table(m.assign(horizon=horizon), episodes)
    ep_scores = np.sort(per_ep["y_score"].to_numpy(dtype=float))
    sweep = alarm_sweep(m["y_true"], m["y_score"])
    n_ep = len(ep_scores)
    counts_ge = n_ep - np.searchsorted(ep_scores, sweep["threshold"].to_numpy(), side="left")
    sweep["episode_recall"] = counts_ge / n_ep if n_ep else 0.0
    sweep["coverage"] = covered
    return sweep[["threshold", "alarm_rate", "episode_recall", "coverage"]]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", type=int, choices=[1, 2], default=2)
    parser.add_argument("--tag", default="recall_tuned")
    args = parser.parse_args(argv)

    fp = (Path("results/predictions")
          / f"xgboost_t{args.tier}_h24_{args.tag}_pooled_test.parquet")
    if not fp.exists():
        raise FileNotFoundError(f"{fp} not found -- train the h=24 config first")
    scores = pd.read_parquet(fp)[["coin", "hour", "y_score"]]
    master, episodes = load_master()

    print(f"alarm cascade: h=24 xgboost (tier {args.tier}, {args.tag}) scores "
          f"re-thresholded for h=1/6/24\n")
    for h in HORIZONS:
        curve = cascade_curve(scores, master, episodes, h, args.tier)
        ops = operating_points(curve)
        print(f"--- horizon {h} "
              f"(coverage of h-eligible rows by h24 scores: {curve['coverage'].iloc[0]:.1%}) ---")
        print("  recall@budget:  " +
              " | ".join(f"{r.alarm_rate:.3f}->{r.episode_recall:.3f}"
                         for r in ops["recall_at_rate"].itertuples()))
        print("  rate@recall:    " +
              " | ".join(f"{r.target_recall:.2f}->{r.alarm_rate:.3f}"
                         for r in ops["rate_at_recall"].itertuples()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
