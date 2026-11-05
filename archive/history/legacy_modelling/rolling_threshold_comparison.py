"""Train fixed-parameter XGBoost runs and compare rolling thresholds.

By default this runs all combinations of tiers 1/2 and horizons 1/6/24,
then collects their yearly rolling-threshold reports into one CSV.

Run from the repository root:
    python modelling/rolling_threshold_comparison.py --tag xgb_d3_lr005_300

To reuse already-trained fold artifacts, pass --skip-training.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO_ROOT / "modelling" / "train.py"
THRESHOLD_SCRIPT = REPO_ROOT / "modelling" / "rolling_threshold.py"


def plot_comparison(comparison: pd.DataFrame, output: Path) -> None:
    """Plot pooled test recall, warned episodes, and false alarm events."""
    grouped = (
        comparison.groupby(["tier", "horizon"], as_index=False)
        .agg(
            test_episodes_warned=("test_episodes_warned", "sum"),
            test_episodes_targetable=("test_episodes_targetable", "sum"),
            test_false_alarms=("test_false_alarms", "sum"),
        )
    )
    grouped["test_episode_recall"] = (
        grouped["test_episodes_warned"]
        / grouped["test_episodes_targetable"].replace(0, np.nan)
    )
    horizons = sorted(grouped["horizon"].unique())
    tiers = sorted(grouped["tier"].unique())
    x = np.arange(len(horizons))
    width = 0.78 / max(len(tiers), 1)
    colors = {1: "#4C78A8", 2: "#F58518"}

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    metrics = [
        ("test_episode_recall", "Test episode recall", (0, 1)),
        ("test_episodes_warned", "Test episodes warned", None),
        ("test_false_alarms", "Test false alarm events", None),
    ]
    for ax, (column, title, ylim) in zip(axes, metrics):
        for i, tier in enumerate(tiers):
            vals = (
                grouped[grouped["tier"] == tier]
                .set_index("horizon")[column]
                .reindex(horizons)
            )
            offset = (i - (len(tiers) - 1) / 2) * width
            bars = ax.bar(
                x + offset, vals.fillna(0), width,
                label=f"Tier {tier}", color=colors.get(tier),
            )
            ax.bar_label(bars, fmt="%.2f" if column == "test_episode_recall" else "%.0f",
                         padding=2, fontsize=8)
        ax.set_title(title)
        ax.set_xticks(x, [f"{h}h" for h in horizons])
        ax.set_xlabel("Forecast horizon")
        ax.grid(axis="y", alpha=0.25)
        if ylim:
            ax.set_ylim(*ylim)
    axes[0].set_ylabel("Episodes warned / targetable episodes")
    axes[1].set_ylabel("Count across test years")
    axes[2].set_ylabel("Count across test years")
    axes[0].legend(frameon=False)
    fig.suptitle("Rolling-threshold performance pooled across test years")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="xgb_d3_lr005_300")
    parser.add_argument("--horizons", nargs="+", type=int, choices=(1, 6, 24),
                        default=(1, 6, 24))
    parser.add_argument("--tiers", nargs="+", type=int, choices=(1, 2),
                        default=(1, 2))
    parser.add_argument(
        "--model-params",
        default='{"max_depth":3,"learning_rate":0.05,"n_estimators":300}',
        help="JSON fixed XGBoost constructor parameters (used when training)",
    )
    parser.add_argument("--false-alarms-per-coin-month", type=float, default=1.0)
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "reports/rolling_thresholds"))
    parser.add_argument("--skip-training", action="store_true",
                        help="use existing model metadata for each requested config")
    args = parser.parse_args(argv)

    try:
        params = json.loads(args.model_params)
    except json.JSONDecodeError as exc:
        parser.error(f"--model-params must be valid JSON: {exc}")
    if not isinstance(params, dict):
        parser.error("--model-params must decode to a JSON object")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for tier in args.tiers:
        for horizon in args.horizons:
            print(f"\n=== xgboost tier {tier}, horizon {horizon} ===", flush=True)
            if not args.skip_training:
                train_cmd = [
                    sys.executable, str(TRAIN_SCRIPT), "--model", "xgboost",
                    "--tier", str(tier), "--horizon", str(horizon),
                    "--tag", args.tag, "--model-params", json.dumps(params),
                ]
                subprocess.run(train_cmd, cwd=REPO_ROOT, check=True)

            rolling_cmd = [
                sys.executable, str(THRESHOLD_SCRIPT), "--model", "xgboost",
                "--tier", str(tier), "--horizon", str(horizon),
                "--tag", args.tag,
                "--false-alarms-per-coin-month",
                str(args.false_alarms_per_coin_month), "--out-dir", str(out_dir),
            ]
            subprocess.run(rolling_cmd, cwd=REPO_ROOT, check=True)
            report = out_dir / (
                f"xgboost_t{tier}_h{horizon}_{args.tag}_rolling_thresholds.csv"
            )
            frame = pd.read_csv(report)
            reports.append(frame)

    comparison = pd.concat(reports, ignore_index=True)
    comparison = comparison.sort_values(["test_year", "tier", "horizon"])
    output = out_dir / f"xgboost_{args.tag}_rolling_threshold_comparison.csv"
    comparison.to_csv(output, index=False)
    print(f"\nSaved comparison: {output}")
    plot_path = out_dir / f"xgboost_{args.tag}_rolling_threshold_comparison.png"
    plot_comparison(comparison, plot_path)
    print(f"Saved comparison plot: {plot_path}")
    print(comparison[[
        "test_year", "tier", "horizon", "threshold",
        "validation_false_alarms_per_coin_month", "validation_episode_recall",
        "test_false_alarms_per_coin_month", "test_episode_recall",
    ]].round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
