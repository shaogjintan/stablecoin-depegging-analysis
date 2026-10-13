"""
Recall-first operating-point report for saved runs (early-warning rework).

Reads the pooled OOS test predictions saved by modelling/train.py
(results/predictions/*_pooled_test.parquet) and, per run, computes:

  - the episode-level recall vs hourly alarm-rate curve,
  - recall achieved at fixed alarm-rate budgets (0.5% / 1% / 2% / 5%),
  - alarm rate needed to hit 90% / 95% / 99% episode recall,
  - (with --alarm-tables) a per-episode caught/missed table at the
    90%-recall operating point.

"Episode recall" = fraction of the 49 out-of-sample depeg episodes with at
least one alarmed onset hour -- the headline metric of the recall-first
evaluation (missed episodes = false negatives).

Usage (from the repo root):
    python modelling/alarm_report.py
    python modelling/alarm_report.py --save reports/recall_operating_points.csv
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "data_processing"))

from config import PROCESSED_DATA_DIR  # noqa: E402
from modelling.evaluate import (  # noqa: E402
    episode_alarm_table,
    episode_recall_curve,
    operating_points,
)

ALARM_TABLES_DIR = REPO_ROOT / "reports" / "alarm_tables"


def load_episodes() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DATA_DIR / "depeg_episodes.parquet")


def summarize_run(fp: Path, episodes: pd.DataFrame) -> dict:
    m = re.match(r"(.+?)_t(\d+)_h(\d+)_(.+)_pooled_test\.parquet$", fp.name)
    if not m:
        raise ValueError(f"unexpected file name: {fp.name}")
    model, tier, horizon, tag = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
    preds = pd.read_parquet(fp)
    curve = episode_recall_curve(preds, episodes)
    ops = operating_points(curve)
    row = {"model": model, "tier": tier, "horizon": horizon, "tag": tag,
           "n_episodes": int(len(episode_alarm_table(preds, episodes, 0.0)))}
    for r in ops["recall_at_rate"].itertuples():
        row[f"recall@{r.alarm_rate:.3f}"] = r.episode_recall
    for r in ops["rate_at_recall"].itertuples():
        row[f"alarm_rate@{r.target_recall:.2f}"] = r.alarm_rate
    return row


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-dir", default="results/predictions")
    parser.add_argument("--save", default=None,
                        help="write the operating table to this CSV")
    parser.add_argument("--alarm-tables", action="store_true",
                        help="write per-episode caught/missed CSVs for each run")
    args = parser.parse_args(argv)

    files = sorted(Path(args.pred_dir).glob("*_pooled_test.parquet"))
    if not files:
        print("No *_pooled_test.parquet files found -- run modelling/train.py first.")
        return 1

    episodes = load_episodes()
    rows = [summarize_run(fp, episodes) for fp in files]
    df = pd.DataFrame(rows).sort_values(["model", "horizon", "tier"]).reset_index(drop=True)

    print("Episode-level recall-first operating table (pooled OOS):")
    print(df.round(4).to_string(index=False))

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"\nSaved to {out}")

    if args.alarm_tables:
        ALARM_TABLES_DIR.mkdir(parents=True, exist_ok=True)
        for fp in files:
            m = re.match(r"(.+?)_t(\d+)_h(\d+)_(.+)_pooled_test\.parquet$", fp.name)
            model, tier, h, tag = m.group(1), m.group(2), m.group(3), m.group(4)
            preds = pd.read_parquet(fp)
            curve = episode_recall_curve(preds, episodes)
            ops = operating_points(curve)
            thr = ops["rate_at_recall"].loc[0, "threshold"]  # 90% operating point
            table = episode_alarm_table(preds, episodes, thr)
            out = ALARM_TABLES_DIR / f"alarm_table_{model}_t{tier}_h{h}_{tag}.csv"
            table.to_csv(out, index=False)
        print(f"Per-episode alarm tables written to {ALARM_TABLES_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
