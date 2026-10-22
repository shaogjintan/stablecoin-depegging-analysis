"""Check rolling-alarm results under alternative depeg definitions.

Keeps each trained model's predictions and baseline rolling thresholds fixed,
then redefines test depeg episodes across price bands and minimum consecutive
below-band durations. This isolates how outcome-definition choices affect
test episode recall and false-alarm event counts; it does not retrain models
or reselect thresholds for each alternative definition.

Run after training and generating rolling-threshold reports:
    # TODO: change to best performing model and tag
    python modelling/robustness_check.py --tag xgb_d3_lr005_300
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "data_processing"))

from config import DEPEG_GAP_HOURS, PROCESSED_DATA_DIR  # noqa: E402
from modelling.alarm_events import build_alarm_events  # noqa: E402


def make_episodes(master: pd.DataFrame, band: float,
                  min_duration: int) -> pd.DataFrame:
    """Create episodes using a down-band and consecutive-hour rule.

    Qualifying dips separated by at most DEPEG_GAP_HOURS from the first
    above-band observation are clustered into one episode, matching the main
    data-processing episode definition.
    """
    if not 0 < band < 1:
        raise ValueError("depeg bands must be fractions between 0 and 1")
    if min_duration < 1:
        raise ValueError("persistence durations must be positive")

    rows = []
    for coin, group in master.sort_values(["coin", "hour"]).groupby("coin", sort=False):
        group = group.reset_index(drop=True)
        below = group["price"].lt(1 - band).fillna(False).to_numpy()
        runs: list[tuple[int, int]] = []
        i = 0
        while i < len(below):
            if not below[i]:
                i += 1
                continue
            start = i
            while i < len(below) and below[i]:
                i += 1
            end = i - 1
            if end - start + 1 >= min_duration:
                runs.append((start, end))

        clusters: list[tuple[int, int]] = []
        for start, end in runs:
            if clusters and start - (clusters[-1][1] + 1) <= DEPEG_GAP_HOURS:
                clusters[-1] = (clusters[-1][0], end)
            else:
                clusters.append((start, end))

        for number, (start, end) in enumerate(clusters, start=1):
            rows.append({
                "coin": coin,
                "episode_id": f"{coin}_{number}",
                "start": group.loc[start, "hour"],
                "end": group.loc[end, "hour"] + pd.Timedelta(hours=1),
                "n_hours": end - start + 1,
                "min_price": group.loc[start:end, "price"].min(),
            })
    return pd.DataFrame(rows, columns=[
        "coin", "episode_id", "start", "end", "n_hours", "min_price"
    ])


def _targetable_ids(predictions: pd.DataFrame, episodes: pd.DataFrame,
                    horizon: int) -> set[str]:
    """Episodes with at least one scored origin in their forecast window."""
    delta = pd.Timedelta(hours=horizon)
    targets: set[str] = set()
    by_coin = {
        coin: group["hour"].sort_values()
        for coin, group in predictions.groupby("coin", sort=False)
    }
    for episode in episodes.itertuples(index=False):
        hours = by_coin.get(episode.coin)
        if hours is not None and ((hours < episode.start) & (hours >= episode.start - delta)).any():
            targets.add(str(episode.episode_id))
    return targets


def evaluate_definition(master: pd.DataFrame, config_rows: list[tuple],
                        bands: list[float], durations: list[int]) -> pd.DataFrame:
    results = []
    episode_cache = {
        (band, duration): make_episodes(master, band, duration)
        for band in bands for duration in durations
    }
    for model, tier, horizon, tag, pred_dir, threshold_dir in config_rows:
        pred_path = pred_dir / f"{model}_t{tier}_h{horizon}_{tag}_pooled_test.parquet"
        threshold_path = threshold_dir / (
            f"{model}_t{tier}_h{horizon}_{tag}_rolling_thresholds.csv"
        )
        if not pred_path.exists() or not threshold_path.exists():
            raise FileNotFoundError(
                f"Missing predictions or rolling thresholds for {model}, "
                f"tier {tier}, horizon {horizon}; expected {pred_path} and {threshold_path}"
            )
        predictions = pd.read_parquet(pred_path).copy()
        predictions["hour"] = pd.to_datetime(predictions["hour"], utc=True)
        thresholds = pd.read_csv(threshold_path).set_index("test_year")["threshold"]

        for band in bands:
            for duration in durations:
                episodes = episode_cache[(band, duration)].copy()
                episodes["start"] = pd.to_datetime(episodes["start"], utc=True)
                warned: set[str] = set()
                targetable: set[str] = set()
                false_alarms = alarm_events = true_alarms = 0
                for year, yearly in predictions.groupby("year", sort=True):
                    threshold = thresholds.get(int(year))
                    if threshold is None or pd.isna(threshold):
                        continue
                    yearly = yearly.copy()
                    yearly["horizon"] = horizon
                    yearly["split"] = "test"
                    targetable.update(_targetable_ids(yearly, episodes, horizon))
                    _, events = build_alarm_events(yearly, episodes, float(threshold))
                    alarm_events += len(events)
                    if not events.empty:
                        true_events = events[events["is_true_alarm"]]
                        true_alarms += len(true_events)
                        false_alarms += int((~events["is_true_alarm"]).sum())
                        warned.update(true_events["target_episode_id"].dropna().astype(str))
                caught = len(warned & targetable)
                results.append({
                    "model": model, "tag": tag, "tier": tier, "horizon": horizon,
                    "depeg_band": band,
                    "price_cutoff": 1 - band,
                    "persistence_hours": duration,
                    "episodes_defined_total": len(episodes),
                    "test_episodes_targetable": len(targetable),
                    "test_episodes_warned": caught,
                    "test_episode_recall": caught / len(targetable) if targetable else float("nan"),
                    "test_alarm_events": alarm_events,
                    "test_true_alarm_events": true_alarms,
                    "test_false_alarm_events": false_alarms,
                })
    return pd.DataFrame(results)


def plot_robustness(results: pd.DataFrame, output: Path) -> None:
    """Show recall and false alarms by persistence, faceted by horizon."""
    horizons = sorted(results["horizon"].unique())
    tiers = sorted(results["tier"].unique())
    bands = sorted(results["depeg_band"].unique())
    colors = {0.005: "#4C78A8", 0.01: "#F58518", 0.02: "#54A24B"}
    fig, axes = plt.subplots(2, len(horizons), figsize=(5 * len(horizons), 8), squeeze=False)
    metrics = [
        ("test_episode_recall", "Pooled test episode recall"),
        ("test_false_alarm_events", "Pooled test false alarm events"),
    ]
    for row, (metric, title) in enumerate(metrics):
        for col, horizon in enumerate(horizons):
            ax = axes[row, col]
            subset = results[results["horizon"] == horizon]
            for tier in tiers:
                for band in bands:
                    line = subset[(subset["tier"] == tier)
                                  & (subset["depeg_band"] == band)]
                    line = line.groupby("persistence_hours", as_index=False)[metric].mean()
                    ax.plot(
                        line["persistence_hours"], line[metric], marker="o",
                        color=colors.get(band), linestyle="-" if tier == 1 else "--",
                        label=f"{band:.1%} band, tier {tier}",
                    )
            ax.set_title(f"{title} — {horizon}h forecast")
            ax.set_xlabel("Minimum consecutive hours below peg")
            ax.set_ylabel("Recall" if metric == "test_episode_recall" else "Event count")
            if metric == "test_episode_recall":
                ax.set_ylim(0, 1)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=7, frameon=False)
    fig.suptitle("Sensitivity to depeg band and persistence definition\n"
                 "Solid = Tier 1; dashed = Tier 2; fixed model scores and thresholds")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="xgboost")
    parser.add_argument("--tag", default="xgb_d3_lr005_300")
    parser.add_argument("--tiers", nargs="+", type=int, choices=(1, 2), default=(1, 2))
    parser.add_argument("--horizons", nargs="+", type=int, choices=(1, 6, 24), default=(1, 6, 24))
    parser.add_argument("--bands", nargs="+", type=float, default=(0.005, 0.01, 0.02),
                        help="downward depeg bands as fractions (default: 0.005 0.01 0.02)")
    parser.add_argument("--persistence-hours", nargs="+", type=int, default=(1, 2, 3, 4))
    parser.add_argument("--pred-dir", default=str(REPO_ROOT / "results/predictions"))
    parser.add_argument("--threshold-dir", default=str(REPO_ROOT / "reports/rolling_thresholds"))
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "reports/robustness"))
    args = parser.parse_args(argv)

    master = pd.read_parquet(PROCESSED_DATA_DIR / "master.parquet")[["coin", "hour", "price"]].copy()
    master["hour"] = pd.to_datetime(master["hour"], utc=True)
    config_rows = [
        (args.model, tier, horizon, args.tag, Path(args.pred_dir), Path(args.threshold_dir))
        for tier in args.tiers for horizon in args.horizons
    ]
    result = evaluate_definition(master, config_rows, args.bands, args.persistence_hours)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.model}_{args.tag}_depeg_robustness"
    csv_path = out_dir / f"{stem}.csv"
    plot_path = out_dir / f"{stem}.png"
    result.to_csv(csv_path, index=False)
    plot_robustness(result, plot_path)
    print(f"Saved robustness table: {csv_path}")
    print(f"Saved robustness plot: {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
