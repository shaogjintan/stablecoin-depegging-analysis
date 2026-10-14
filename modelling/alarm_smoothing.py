"""
Causal alarm smoothing: require 2 consecutive above-threshold hours to fire.

An alarm at hour t fires only if score_t >= thr AND score_{t-1} >= thr (the
previous hour's score, per coin, on true calendar hours). This is causal
(no lookahead) and drastically cuts single-hour false alarms while an episode
— which lasts many hours — still has most of its onset-window hours alarmed.

Compares the smoothed recall vs alarm-rate curve against the raw curve for a
saved pooled run and prints the operating table for both.

Usage (from the repo root):
    python modelling/alarm_smoothing.py --model xgboost --tier 2 --horizon 24 \
        --tag recall_tuned [--k 2]
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

from config import PROCESSED_DATA_DIR  # noqa: E402
from modelling.evaluate import (  # noqa: E402
    alarm_sweep,
    episode_recall_curve,
    operating_points,
)


def smooth_scores(preds: pd.DataFrame, k: int = 2) -> pd.DataFrame:
    """Return preds with an added 'y_score_smooth' column: score kept only if
    the previous k-1 calendar hours (same coin) also cleared it (causal)."""
    out = preds.copy()
    out["y_score_smooth"] = np.nan
    for coin, g in out.groupby("coin", sort=False):
        g = g.sort_values("hour").copy()
        prev = g["y_score"].shift(1)
        g["hour_gap"] = g["hour"].diff().dt.total_seconds() / 3600
        ok = (g["hour_gap"] == 1.0) & prev.notna()
        out.loc[g.index, "y_score_smooth"] = np.where(
            ok, np.minimum(g["y_score"], prev).to_numpy(), 0.0)
    return out


def smoothed_curve(preds: pd.DataFrame, episodes: pd.DataFrame) -> pd.DataFrame:
    """Sweep thresholds on the smoothed scores: smoothed alarm rate vs episode
    recall (an onset hour counts as caught only if it is smoothed-alarmed)."""
    per_ep = preds[preds["y_true"] == 1].copy()
    curve = alarm_sweep(preds["y_true"], preds["y_score_smooth"])
    # episode-level: onset hour caught iff smoothed score >= threshold
    ep_vals = per_ep.set_index(["coin", "hour"])["y_score_smooth"].rename("ep_score")
    joined = preds[["coin", "hour"]].join(ep_vals, on=["coin", "hour"])
    caught_per_ep = per_ep[["coin", "hour"]].merge(
        joined.dropna(subset=["ep_score"]), on=["coin", "hour"], how="left")["ep_score"]
    ep_scores = np.sort(caught_per_ep.to_numpy(dtype=float))
    n_ep = len(ep_scores)
    counts_ge = n_ep - np.searchsorted(ep_scores, curve["threshold"].to_numpy(), side="left")
    curve["episode_recall"] = counts_ge / n_ep if n_ep else 0.0
    return curve[["threshold", "alarm_rate", "episode_recall"]]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="xgboost")
    parser.add_argument("--tier", type=int, choices=[1, 2], default=2)
    parser.add_argument("--horizon", type=int, choices=[1, 6, 24], default=24)
    parser.add_argument("--tag", default="recall_tuned")
    parser.add_argument("--k", type=int, default=2,
                        help="consecutive hours required (default 2)")
    args = parser.parse_args(argv)

    fp = (Path("results/predictions")
          / f"{args.model}_t{args.tier}_h{args.horizon}_{args.tag}_pooled_test.parquet")
    preds = pd.read_parquet(fp)
    episodes = pd.read_parquet(PROCESSED_DATA_DIR / "depeg_episodes.parquet")

    smoothed = smooth_scores(preds, k=args.k)
    curve_raw = episode_recall_curve(preds, episodes)
    curve_sm = smoothed_curve(smoothed, episodes)

    print(f"alarm smoothing (k={args.k} consecutive hours), "
          f"{args.model} t{args.tier} h{args.horizon} {args.tag}\n")
    print("RAW operating points:")
    for r in operating_points(curve_raw)["recall_at_rate"].itertuples():
        print(f"  recall@{r.alarm_rate:.3f} = {r.episode_recall:.3f}")
    print("SMOOTHED operating points:")
    for r in operating_points(curve_sm)["recall_at_rate"].itertuples():
        print(f"  recall@{r.alarm_rate:.3f} = {r.episode_recall:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
