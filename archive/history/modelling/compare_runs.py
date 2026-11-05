"""
Build the pooled Tier 1 vs Tier 2 comparison table from saved runs.

Scans results/predictions/*_pooled_test.parquet (written by
modelling/train.py) and prints pooled OOS metrics per model/horizon/tier,
including the Tier 2 minus Tier 1 PR-AUC delta and a PR-AUC matrix.
Optionally saves the full table to CSV (final tables go to reports/ per the
Phase 3 results convention).

Baselines (constant prior, persistence) are tier-independent; their rows
carry tier = 1 from the training command and apply to both tiers.

Usage (from the repo root):
    python modelling/compare_runs.py
    python modelling/compare_runs.py --save reports/modelling_comparison.csv
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modelling.evaluate import pooled_metrics  # noqa: E402


def load_runs(pred_dir: str | Path = "results/predictions") -> pd.DataFrame:
    """One row per *_pooled_test.parquet file with pooled OOS metrics."""
    files = sorted(Path(pred_dir).glob("*_pooled_test.parquet"))
    rows = []
    for fp in files:
        m = re.match(r"(.+?)_t(\d+)_h(\d+)_(.+)_pooled_test\.parquet$", fp.name)
        if not m:
            continue
        model, tier, horizon, tag = (m.group(1), int(m.group(2)),
                                     int(m.group(3)), m.group(4))
        preds = pd.read_parquet(fp)
        rows.append({"model": model, "tier": tier, "horizon": horizon,
                     "tag": tag, **pooled_metrics(preds["y_true"], preds["y_score"])})
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-dir", default="results/predictions")
    parser.add_argument("--save", default=None, help="write the table to this CSV")
    args = parser.parse_args(argv)

    df = load_runs(args.pred_dir)
    if df.empty:
        print("No *_pooled_test.parquet files found -- run modelling/train.py first.")
        return 1

    df = df.sort_values(["model", "horizon", "tier"]).reset_index(drop=True)
    show = df[["model", "tier", "horizon", "tag", "n", "n_pos",
               "pr_auc", "roc_auc", "brier"]]
    print("Pooled OOS metrics by run:")
    print(show.round(4).to_string(index=False))

    print("\nPR-AUC matrix (tier 2 - tier 1 delta):")
    piv = df.pivot_table(index=["model", "horizon"], columns="tier",
                         values="pr_auc", aggfunc="first")
    for t in (1, 2):
        if t not in piv.columns:
            piv[t] = float("nan")
    piv["delta_t2_t1"] = piv[2] - piv[1]
    print(piv.round(4).to_string())

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"\nSaved full table to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
