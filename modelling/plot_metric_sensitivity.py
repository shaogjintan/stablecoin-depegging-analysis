"""Plot pooled metrics across forecast horizons for non-baseline models.

Reads reports/modelling_comparison.csv and reports/recall_operating_points.csv.
The x-axis is forecast horizon (hours); each model/tier is a separate series.
The CSV is a wide, plotting-ready table. Missing model/horizon combinations
remain blank. No model is retrained and no metrics are recalculated here.

Run from the repository root:
    python3 modelling/plot_metric_sensitivity.py
"""
from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "reports/robustness"
EXCLUDED_MODELS = {"baseline", "persistence"}
RECALL_MAP = {
    "recall@0.005": "episode_recall_at_0.5pct_alarm_rate",
    "recall@0.010": "episode_recall_at_1pct_alarm_rate",
    "recall@0.020": "episode_recall_at_2pct_alarm_rate",
}
METRICS = [
    ("pr_auc", "PR-AUC (reported pooled score)"),
    ("episode_recall_at_0.5pct_alarm_rate", "Episode recall @ 0.5% hourly alarm-rate budget"),
    ("episode_recall_at_1pct_alarm_rate", "Episode recall @ 1% hourly alarm-rate budget"),
    ("episode_recall_at_2pct_alarm_rate", "Episode recall @ 2% hourly alarm-rate budget"),
]
COLORS = ["#2563A6", "#E17C24", "#3B8C62", "#9B59A5", "#C44E52"]


def prepare_metrics() -> pd.DataFrame:
    scores = pd.read_csv(REPO_ROOT / "reports/modelling_comparison.csv")
    ops = pd.read_csv(REPO_ROOT / "reports/recall_operating_points.csv")
    scores = scores.loc[~scores.model.isin(EXCLUDED_MODELS)].copy()
    ops = ops.loc[~ops.model.isin(EXCLUDED_MODELS)].copy()
    keys = ["model", "tier", "horizon", "tag"]
    ops = ops[keys + ["n_episodes", *RECALL_MAP]].rename(columns=RECALL_MAP)
    result = scores.merge(ops, on=keys, how="outer", validate="one_to_one")
    return result.sort_values(["model", "tier", "horizon"]).reset_index(drop=True)


def _svg_text(x: float, y: float, label: str, *, size: int = 12,
              color: str = "#222", anchor: str = "start", weight: str = "normal") -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial,sans-serif" '
            f'font-size="{size}" fill="{color}" text-anchor="{anchor}" '
            f'font-weight="{weight}">{escape(label)}</text>')


def render_svg(metrics: pd.DataFrame) -> str:
    width, height = 1440, 900
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>']
    parts.append(_svg_text(width / 2, 34, "Pooled out-of-sample metrics across forecast horizons", size=21, anchor="middle", weight="bold"))
    parts.append(_svg_text(width / 2, 58, "Color = model; solid = Tier 1; dashed = Tier 2", size=12, anchor="middle", color="#555"))
    models = sorted(metrics.model.dropna().unique())
    palette = {m: COLORS[i % len(COLORS)] for i, m in enumerate(models)}
    panel_w, panel_h = 660, 365
    positions = [(55, 90), (735, 90), (55, 475), (735, 475)]
    horizons = [1, 6, 24]

    for (metric, title), (left, top) in zip(METRICS, positions):
        plot_left, plot_top = left + 58, top + 38
        plot_w, plot_h = panel_w - 88, panel_h - 94
        vals = pd.to_numeric(metrics[metric], errors="coerce").dropna()
        ymax = 1.0 if metric.startswith("episode_recall") else (float(vals.max()) * 1.12 if len(vals) else 1.0)
        ymax = max(ymax, 0.01)
        parts.append(_svg_text(left + 4, top + 19, title, size=14, weight="bold"))
        for tick_i in range(5):
            value = ymax * tick_i / 4
            y = plot_top + plot_h * (1 - tick_i / 4)
            parts.append(f'<line x1="{plot_left}" y1="{y:.1f}" x2="{plot_left + plot_w}" y2="{y:.1f}" stroke="#ddd"/>')
            parts.append(_svg_text(plot_left - 8, y + 4, f"{value:.2f}", size=10, anchor="end", color="#555"))
        for i, horizon in enumerate(horizons):
            x = plot_left + plot_w * i / (len(horizons) - 1)
            parts.append(f'<line x1="{x:.1f}" y1="{plot_top}" x2="{x:.1f}" y2="{plot_top + plot_h}" stroke="#eee"/>')
            parts.append(_svg_text(x, plot_top + plot_h + 19, str(horizon), size=11, anchor="middle"))
        parts.append(f'<line x1="{plot_left}" y1="{plot_top + plot_h}" x2="{plot_left + plot_w}" y2="{plot_top + plot_h}" stroke="#555"/>')
        parts.append(_svg_text(plot_left + plot_w / 2, plot_top + plot_h + 39, "Forecast horizon (hours)", size=11, anchor="middle"))
        for model in models:
            for tier in (1, 2):
                line = metrics.loc[metrics.model.eq(model) & metrics.tier.eq(tier), ["horizon", metric]].copy()
                line[metric] = pd.to_numeric(line[metric], errors="coerce")
                line = line.dropna().sort_values("horizon")
                if line.empty:
                    continue
                points = []
                for row in line.itertuples(index=False, name=None):
                    h, value = row
                    x = plot_left + plot_w * horizons.index(int(h)) / (len(horizons) - 1)
                    y = plot_top + plot_h * (1 - float(value) / ymax)
                    points.append((x, y))
                dash = ' stroke-dasharray="7,5"' if tier == 2 else ""
                if len(points) > 1:
                    path = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
                    parts.append(f'<polyline points="{path}" fill="none" stroke="{palette[model]}" stroke-width="2"{dash}/>')
                for x, y in points:
                    parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="white" stroke="{palette[model]}" stroke-width="2"/>')
    # Shared legend, grouped by model and tier.
    legend_y = 870
    cursor = 95
    for model in models:
        for tier in (1, 2):
            dash = ' stroke-dasharray="7,5"' if tier == 2 else ""
            parts.append(f'<line x1="{cursor}" y1="{legend_y - 4}" x2="{cursor + 24}" y2="{legend_y - 4}" stroke="{palette[model]}" stroke-width="2"{dash}/>')
            label = f"{model.replace('_', ' ')} T{tier}"
            parts.append(_svg_text(cursor + 30, legend_y, label, size=11))
            cursor += 30 + len(label) * 6.4 + 28
    parts.append("</svg>")
    return "\n".join(parts)


def main() -> int:
    metrics = prepare_metrics()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "nonbaseline_metrics_by_horizon.csv"
    svg_path = OUTPUT_DIR / "nonbaseline_metrics_by_horizon.svg"
    metrics.to_csv(csv_path, index=False)
    svg_path.write_text(render_svg(metrics), encoding="utf-8")
    print(f"Saved metric table: {csv_path}")
    print(f"Saved line plots: {svg_path}")
    print(f"Rows: {len(metrics)} | models: {', '.join(sorted(metrics.model.unique()))} | tiers: 1, 2")
    print("Model-horizon coverage:")
    print(metrics.groupby(["model", "horizon"]).size().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
