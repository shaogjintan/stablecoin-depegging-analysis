"""
OR-combination test: does alarming on (system A OR system B) catch more depeg
episodes than either system alone at the same alarm-rate budget?

Loads two pooled OOS runs (same horizon), aligns them on (coin, hour), then
sweeps a coarse threshold grid for both systems. An hour is alarmed if EITHER
score passes its threshold; an episode is caught if EITHER system alarms any
of its onset hours. Reports, per budget, the best union recall and the gain
over the best single system. Each system's grid includes +inf ("never
alarm"), so the single-system operating points are contained in the union
search — union recall is always >= the better single system by construction.

Usage (from the repo root):
    python modelling/combine_alarms.py --all
    python modelling/combine_alarms.py --horizon 24 --tier 2 \
        --model-a persistence --tag-a persistence \
        --model-b xgboost --tag-b recall_tuned \
        [--save reports/combination_results.csv]
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
from modelling.evaluate import per_episode_table  # noqa: E402

GRID_RATES = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5)
BUDGETS = (0.005, 0.01, 0.02, 0.05)


def load_run(model: str, tier: int, horizon: int, tag: str,
             pred_dir: str = "results/predictions") -> pd.DataFrame:
    fp = Path(pred_dir) / f"{model}_t{tier}_h{horizon}_{tag}_pooled_test.parquet"
    if not fp.exists():
        raise FileNotFoundError(f"{fp} not found -- train that config first")
    return pd.read_parquet(fp)


def threshold_grid(scores: np.ndarray) -> np.ndarray:
    """Candidate thresholds: score quantiles at 1 - grid rates, plus +inf."""
    return np.concatenate([[np.quantile(scores, 1 - r) for r in GRID_RATES], [np.inf]])


def combine_pair(model_a, tag_a, tier_a, model_b, tag_b, tier_b,
                 horizon: int, episodes: pd.DataFrame) -> pd.DataFrame:
    preds_a = load_run(model_a, tier_a, horizon, tag_a)
    preds_b = load_run(model_b, tier_b, horizon, tag_b)
    keys = ["coin", "hour"]
    m = (preds_a[keys + ["y_score"]]
         .merge(preds_b[keys + ["y_score"]], on=keys,
                how="inner", validate="one_to_one",
                suffixes=("_a", "_b")))
    s_a, s_b = m["y_score_a"].to_numpy(dtype=float), m["y_score_b"].to_numpy(dtype=float)

    ep_a = per_episode_table(preds_a, episodes).set_index("episode_id")["y_score"]
    ep_b = per_episode_table(preds_b, episodes).set_index("episode_id")["y_score"]
    shared = sorted(set(ep_a.index) & set(ep_b.index))
    ea, eb = ep_a.reindex(shared).to_numpy(dtype=float), ep_b.reindex(shared).to_numpy(dtype=float)
    n_ep = len(shared)

    grid_a, grid_b = threshold_grid(s_a), threshold_grid(s_b)
    combos = []
    for ta in grid_a:
        alarm_a = s_a >= ta
        for tb in grid_b:
            union = alarm_a | (s_b >= tb)
            caught = (ea >= ta) | (eb >= tb)
            combos.append((float(union.mean()), float(caught.mean()), float(ta), float(tb)))
    df = pd.DataFrame(combos, columns=["alarm_rate", "episode_recall", "t_a", "t_b"])

    single_a = df[df["t_b"] == np.inf]
    single_b = df[df["t_a"] == np.inf]
    rows = []
    for b in BUDGETS:
        best = df[df["alarm_rate"] <= b].sort_values("episode_recall", ascending=False)
        ba = single_a[single_a["alarm_rate"] <= b].sort_values("episode_recall", ascending=False)
        bb = single_b[single_b["alarm_rate"] <= b].sort_values("episode_recall", ascending=False)
        best_single = max(ba.iloc[0]["episode_recall"] if len(ba) else 0.0,
                          bb.iloc[0]["episode_recall"] if len(bb) else 0.0)
        rows.append({
            "horizon": horizon, "tier_b": tier_b,
            "budget": b, "n_episodes": n_ep,
            f"{model_a}_recall": ba.iloc[0]["episode_recall"] if len(ba) else float("nan"),
            f"{model_b}_recall": bb.iloc[0]["episode_recall"] if len(bb) else float("nan"),
            "union_recall": best.iloc[0]["episode_recall"],
            "union_rate": best.iloc[0]["alarm_rate"],
            "t_a": best.iloc[0]["t_a"], "t_b": best.iloc[0]["t_b"],
            "gain_over_best_single": best.iloc[0]["episode_recall"] - best_single,
        })
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true",
                        help="run persistence+xgboost for all horizons and tiers")
    parser.add_argument("--horizon", type=int, choices=[1, 6, 24], default=24)
    parser.add_argument("--tier", type=int, choices=[1, 2], default=2)
    parser.add_argument("--model-a", default="persistence")
    parser.add_argument("--tag-a", default="persistence")
    parser.add_argument("--tier-a", type=int, default=1)
    parser.add_argument("--model-b", default="xgboost")
    parser.add_argument("--tag-b", default="recall_tuned")
    parser.add_argument("--save", default="reports/combination_results.csv")
    args = parser.parse_args(argv)

    episodes = pd.read_parquet(PROCESSED_DATA_DIR / "depeg_episodes.parquet")
    configs = ([(h, t) for h in (1, 6, 24) for t in (1, 2)] if args.all
               else [(args.horizon, args.tier)])

    frames = []
    for h, t in configs:
        print(f"\n=== persistence OR xgboost (tier {t}) at h={h} ===")
        df = combine_pair(args.model_a, args.tag_a, args.tier_a,
                          args.model_b, args.tag_b, t, h, episodes)
        frames.append(df)
        print(df.round(4).to_string(index=False))

    out = pd.concat(frames, ignore_index=True)
    if args.save:
        out.to_csv(args.save, index=False)
        print(f"\nSaved to {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
