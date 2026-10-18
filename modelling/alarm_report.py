"""
Recall-first operating-point report for saved runs (early-warning rework).

LEAK-FREE PROTOCOL (updated 2026-10-18, per Alyssa's review + prof):
Thresholds are NEVER chosen on out-of-sample test rows. For every fold we
choose the decision threshold on that fold's INNER-VAL predictions (the
expanding-window convention), then apply it to that fold's TEST year, then
pool across folds:

  - recall@budget b: per fold, threshold = argmax inner-val episode recall
    subject to inner-val hourly alarm rate <= b; report pooled test episode
    recall and pooled test alarm rate.
  - rate@target r: per fold, threshold = argmin inner-val hourly alarm rate
    subject to inner-val episode recall >= r; same pooling.

"Episode recall" = fraction of out-of-sample depeg episodes with at least one
alarmed onset hour (missed episodes = false negatives). The raw OOS curves
remain available via modelling/evaluate.py for visualisation only.

Input: the per-fold files written by modelling/train.py
(results/predictions/{model}_t{tier}_h{horizon}_{year}.parquet) which contain
both test and inner_val rows (split column).

Usage (from the repo root):
    python modelling/alarm_report.py
    python modelling/alarm_report.py --save reports/recall_operating_points.csv
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "data_processing"))

from config import PROCESSED_DATA_DIR  # noqa: E402
from modelling.evaluate import alarm_sweep, per_episode_table  # noqa: E402

BUDGETS = (0.005, 0.01, 0.02, 0.05)
TARGETS = (0.90, 0.95, 0.99)


def load_episodes() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DATA_DIR / "depeg_episodes.parquet")


def _episode_scores(df: pd.DataFrame, episodes: pd.DataFrame,
                    horizon: int) -> np.ndarray:
    """Best score per episode over its onset rows (ascending)."""
    per_ep = per_episode_table(df.assign(horizon=horizon), episodes)
    return np.sort(per_ep["y_score"].to_numpy(dtype=float))


def _inner_sweep(inner: pd.DataFrame, episodes: pd.DataFrame,
                 horizon: int) -> pd.DataFrame:
    """Hourly threshold sweep on inner-val, with episode recall attached."""
    sw = alarm_sweep(inner["y_true"], inner["y_score"])
    if len(sw) == 0:
        return sw.assign(episode_recall=0.0)
    ep_scores = _episode_scores(inner, episodes, horizon)
    n_ep = len(ep_scores)
    if n_ep == 0:
        return sw.assign(episode_recall=0.0)
    counts_ge = n_ep - np.searchsorted(ep_scores, sw["threshold"].to_numpy(), side="left")
    sw["episode_recall"] = counts_ge / n_ep
    return sw


def _apply_threshold(test: pd.DataFrame, episodes: pd.DataFrame,
                     horizon: int, thr: float) -> tuple[int, int, int]:
    """Apply one threshold to one fold's test rows -> (caught episodes,
    alarmed hours, test hours)."""
    per_ep = per_episode_table(test.assign(horizon=horizon), episodes)
    caught = int((per_ep["y_score"] >= thr).sum())
    alarmed = int((test["y_score"] >= thr).sum())
    return caught, alarmed, len(test)


def summarize_run(files: list[Path], episodes: pd.DataFrame, tag: str,
                  model: str, tier: int, horizon: int) -> dict:
    """Leak-free operating points: inner-val-selected thresholds, pooled test
    evaluation."""
    n_ep_total = 0
    fold_results = []
    for fp in files:
        fold = pd.read_parquet(fp)
        inner = fold[fold["split"] == "inner_val"]
        test = fold[fold["split"] == "test"]
        sw = _inner_sweep(inner, episodes, horizon)
        n_ep_total += len(_episode_scores(test, episodes, horizon))
        if len(sw) == 0:
            continue
        out = {}
        for b in BUDGETS:
            feas = sw[sw["alarm_rate"] <= b]
            if feas.empty:
                # even the least-alarming positive threshold exceeds the budget
                out[f"budget_{b}"] = None
                continue
            r = feas.iloc[feas["episode_recall"].argmax()]
            out[f"budget_{b}"] = _apply_threshold(test, episodes, horizon, r["threshold"])
        for t in TARGETS:
            feas = sw[sw["episode_recall"] >= t]
            if feas.empty:
                feas = sw.tail(1)  # best effort: lowest threshold
            r = feas.iloc[feas["alarm_rate"].argmin()]
            out[f"target_{t}"] = _apply_threshold(test, episodes, horizon, r["threshold"])
        fold_results.append(out)

    summary = {"model": model, "tier": tier, "horizon": horizon, "tag": tag,
               "n_episodes": n_ep_total}
    for b in BUDGETS:
        vals = [r[f"budget_{b}"] for r in fold_results if r.get(f"budget_{b}") is not None]
        caught = sum(v[0] for v in vals)
        alarmed = sum(v[1] for v in vals)
        n_test = sum(v[2] for v in vals)
        summary[f"recall@{b:.3f}"] = caught / n_ep_total if n_ep_total else float("nan")
        summary[f"rate@{b:.3f}"] = alarmed / n_test if n_test else float("nan")
    for t in TARGETS:
        vals = [r[f"target_{t}"] for r in fold_results if r.get(f"target_{t}") is not None]
        caught = sum(v[0] for v in vals)
        alarmed = sum(v[1] for v in vals)
        n_test = sum(v[2] for v in vals)
        summary[f"alarm_rate@{t:.2f}"] = alarmed / n_test if n_test else float("nan")
    return summary


def find_runs(pred_dir: Path) -> list[dict]:
    """Group per-fold files into complete (model, tier, horizon) runs."""
    runs: dict[tuple, list[Path]] = {}
    for fp in sorted(pred_dir.glob("*.parquet")):
        m = re.match(r"(.+?)_t(\d+)_h(\d+)_(\d{4})\.parquet$", fp.name)
        if not m:
            continue
        key = (m.group(1), int(m.group(2)), int(m.group(3)))
        runs.setdefault(key, []).append(fp)
    out = []
    for (model, tier, horizon), files in runs.items():
        if len(files) != 5:
            continue  # incomplete run
        tag = f"{model}_t{tier}_h{horizon}"
        json_fp = REPO_ROOT / "results" / "models" / f"{model}_t{tier}_h{horizon}_2021.json"
        if json_fp.exists():
            tag = json.loads(json_fp.read_text()).get("tag", tag)
        out.append({"model": model, "tier": tier, "horizon": horizon,
                    "tag": tag, "files": files})
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-dir", default="results/predictions")
    parser.add_argument("--save", default=None,
                        help="write the operating table to this CSV")
    args = parser.parse_args(argv)

    episodes = load_episodes()
    runs = find_runs(Path(args.pred_dir))
    if not runs:
        print("No complete 5-fold runs found -- run modelling/train.py first.")
        return 1

    rows = [summarize_run(r["files"], episodes, r["tag"], r["model"],
                          r["tier"], r["horizon"]) for r in runs]
    df = pd.DataFrame(rows).sort_values(["model", "horizon", "tier"]).reset_index(drop=True)

    show = df[["model", "tier", "horizon", "tag", "n_episodes"]
              + [f"recall@{b:.3f}" for b in BUDGETS]
              + [f"rate@{b:.3f}" for b in BUDGETS]
              + [f"alarm_rate@{t:.2f}" for t in TARGETS]]
    print("Leak-free operating table (thresholds on inner_val, pooled test):")
    print(show.round(4).to_string(index=False))

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"\nSaved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
