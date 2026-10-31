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

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "data_processing"))

from config import (  # noqa: E402
    DEPEG_BAND, DEPEG_GAP_HOURS, DEPEG_MIN_DURATION_HOURS, PROCESSED_DATA_DIR,
)
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
        alarm_path = threshold_dir / (
            f"{model}_t{tier}_h{horizon}_{tag}_test_alarm_events.csv"
        )
        if not pred_path.exists() or not threshold_path.exists() or not alarm_path.exists():
            raise FileNotFoundError(
                f"Missing predictions, thresholds, or alarm events for {model}, "
                f"tier {tier}, horizon {horizon}; expected {pred_path}, "
                f"{threshold_path}, and {alarm_path}"
            )
        predictions = pd.read_parquet(pred_path).copy()
        predictions["hour"] = pd.to_datetime(predictions["hour"], utc=True)
        fixed_events = pd.read_csv(alarm_path)
        fixed_events["alarm_hour"] = pd.to_datetime(fixed_events["alarm_hour"], utc=True)
        month_keys = predictions[["coin", "hour"]].copy()
        month_keys["coin_month"] = month_keys["hour"].dt.strftime("%Y-%m")
        coin_months = int(month_keys[["coin", "coin_month"]].drop_duplicates().shape[0])

        for band in bands:
            for duration in durations:
                episodes = episode_cache[(band, duration)].copy()
                episodes["start"] = pd.to_datetime(episodes["start"], utc=True)
                warned: set[str] = set()
                targetable: set[str] = set()
                false_alarms = alarm_events = true_alarms = 0
                warning_leads = []
                for year, yearly in predictions.groupby("year", sort=True):
                    targetable.update(_targetable_ids(yearly, episodes, horizon))
                    events = fixed_events.loc[fixed_events["year"].eq(int(year))]
                    alarm_events += len(events)
                    eps_by_coin = {
                        coin: group.sort_values("start")
                        for coin, group in episodes.groupby("coin", sort=False)
                    }
                    for event in events.itertuples(index=False):
                        coin_eps = eps_by_coin.get(event.coin)
                        if coin_eps is None:
                            false_alarms += 1
                            continue
                        hit = coin_eps.loc[
                            (coin_eps["start"] > event.alarm_hour)
                            & (coin_eps["start"] <= event.alarm_hour + pd.Timedelta(hours=horizon))
                        ]
                        if hit.empty:
                            false_alarms += 1
                        else:
                            true_alarms += 1
                            caught_episode = hit.iloc[0]
                            warned.add(str(caught_episode["episode_id"]))
                            warning_leads.append(
                                (caught_episode["start"] - event.alarm_hour)
                                / pd.Timedelta(hours=1)
                            )
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
                    "coin_months": coin_months,
                    "test_alarm_events": alarm_events,
                    "test_true_alarm_events": true_alarms,
                    "test_false_alarm_events": false_alarms,
                    "test_false_alarms_per_coin_month": (
                        false_alarms / coin_months if coin_months else float("nan")
                    ),
                    "test_alarm_precision": (
                        true_alarms / alarm_events if alarm_events else float("nan")
                    ),
                    "median_warning_lead_hours": (
                        float(pd.Series(warning_leads).median()) if warning_leads else float("nan")
                    ),
                    "is_reference_definition": (
                        abs(band - DEPEG_BAND) < 1e-12
                        and duration == DEPEG_MIN_DURATION_HOURS
                    ),
                })
    return pd.DataFrame(results)


def _svg_label(x: float, y: float, text: str, *, size: int = 14,
               anchor: str = "middle", color: str = "#202124", weight: str = "normal") -> str:
    from html import escape
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial,sans-serif" '
            f'font-size="{size}" text-anchor="{anchor}" fill="{color}" '
            f'font-weight="{weight}">{escape(text)}</text>')


def _heatmap_svg(results: pd.DataFrame, metric: str, title: str,
                 value_label: str, *, recall_labels: bool = False) -> str:
    bands = sorted(results["depeg_band"].unique())
    durations = sorted(results["persistence_hours"].unique())
    width, height = 720, 430
    left, top, cell_w, cell_h = 185, 130, 112, 72
    vals = pd.to_numeric(results[metric], errors="coerce")
    maximum = 1.0 if recall_labels else (float(vals.max()) if vals.notna().any() else 1.0)
    maximum = max(maximum, 1e-12)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             _svg_label(width / 2, 38, title, size=20, weight="bold"),
             _svg_label(left + cell_w * len(durations) / 2, 86, "Required consecutive duration (hours)", size=13)]
    for ci, duration in enumerate(durations):
        x = left + ci * cell_w + cell_w / 2
        parts.append(_svg_label(x, top - 16, str(duration), size=14, weight="bold"))
    for ri, band in enumerate(bands):
        y = top + ri * cell_h
        parts.append(_svg_label(left - 20, y + cell_h / 2 + 5, f"{band:.1%}", size=14, anchor="end", weight="bold"))
        for ci, duration in enumerate(durations):
            row = results.loc[
                results["depeg_band"].eq(band)
                & results["persistence_hours"].eq(duration)
            ].iloc[0]
            value = row[metric]
            ratio = 0.0 if pd.isna(value) else min(max(float(value) / maximum, 0.0), 1.0)
            if recall_labels:
                # Lower recall is pale coral; higher recall is pale green.
                red = int(250 - 90 * ratio)
                green = int(226 + 20 * ratio)
                blue = int(220 - 45 * ratio)
                label = (f"{int(row['test_episodes_warned'])} / "
                         f"{int(row['test_episodes_targetable'])}")
            else:
                # Higher false-alarm rate is stronger red.
                red = 239
                green = int(247 - 95 * ratio)
                blue = int(247 - 105 * ratio)
                label = "n/a" if pd.isna(value) else f"{float(value):.3f}"
            fill = f"#{red:02x}{green:02x}{blue:02x}"
            x = left + ci * cell_w
            is_ref = bool(row["is_reference_definition"])
            stroke = "#111111" if is_ref else "#ffffff"
            stroke_width = 4 if is_ref else 2
            parts.append(f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"/>')
            parts.append(_svg_label(x + cell_w / 2, y + cell_h / 2 + 5, label, size=16, weight="bold"))
            if is_ref:
                parts.append(_svg_label(x + cell_w - 7, y + 16, "REF", size=9, anchor="end", color="#111", weight="bold"))
    parts.extend([
        _svg_label(left + cell_w * len(durations) / 2, top + cell_h * len(bands) + 44,
                   "Price band = required drop below $1 peg", size=12, color="#555"),
        _svg_label(width / 2, height - 20,
                   "Reference definition: 1% band, 2-hour minimum (outlined)", size=12, color="#444"),
        "</svg>",
    ])
    return "\n".join(parts)


def plot_robustness_heatmaps(results: pd.DataFrame, out_dir: Path,
                             stem: str) -> tuple[Path, Path]:
    """Write recall and false-alarm-per-coin-month heatmaps as SVGs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    recall_path = out_dir / f"{stem}_recall_heatmap.svg"
    false_path = out_dir / f"{stem}_false_alarms_per_coin_month_heatmap.svg"
    recall_path.write_text(
        _heatmap_svg(results, "test_episode_recall", "Episode recall by definition",
                     "caught / targetable", recall_labels=True), encoding="utf-8"
    )
    false_path.write_text(
        _heatmap_svg(results, "test_false_alarms_per_coin_month",
                     "False alarms per coin-month by definition", "false alarms / coin-month"),
        encoding="utf-8",
    )
    return recall_path, false_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="xgboost")
    parser.add_argument("--tag", default="xgb_d3_lr005_300")
    parser.add_argument("--tiers", nargs="+", type=int, choices=(1, 2), default=(2,))
    parser.add_argument("--horizons", nargs="+", type=int, choices=(1, 6, 24), default=(24,))
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
    tier_label = "t" + "_".join(map(str, args.tiers))
    horizon_label = "h" + "_".join(map(str, args.horizons))
    stem = f"{args.model}_{args.tag}_{tier_label}_{horizon_label}_depeg_robustness"
    csv_path = out_dir / f"{stem}.csv"
    result.to_csv(csv_path, index=False)
    recall_path, false_path = plot_robustness_heatmaps(result, out_dir, stem)
    print(f"Saved robustness table: {csv_path}")
    print(f"Saved recall heatmap: {recall_path}")
    print(f"Saved false-alarm heatmap: {false_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
