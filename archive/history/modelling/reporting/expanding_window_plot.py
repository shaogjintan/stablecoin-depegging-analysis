"""Unused expanding-window plotting helper retained for historical reference."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from data_processing.config import OOS_START_YEAR, OOS_END_YEAR
from modelling.reporting.event_table import SURFACE, GRID, MUTED, INK


def plot_expanding_window(by_year: pd.DataFrame, path: Path) -> None:
    """Expanding-window CV folds: train / threshold year / test per fold, with
    each fold's test episodes and events alongside."""
    yrs = by_year.loc[by_year["year"] != "total"].copy()
    yrs["year"] = yrs["year"].astype(int)
    first, last = int(yrs["year"].min()), int(yrs["year"].max())
    counts = yrs.set_index("year")
    train_c, thresh_c, test_c = "#2a78d6", "#9ec5f4", "#eb6834"
    folds = list(range(OOS_START_YEAR, OOS_END_YEAR + 1))
    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    gap = 0.06
    for r, test_year in enumerate(folds):
        for y in range(first, last + 1):
            x0, w, h = y - first + gap / 2, 1 - gap, 0.62
            if y < test_year - 1:
                fc, ec, label, tc = train_c, train_c, "train", SURFACE
            elif y == test_year - 1:
                fc, ec, label, tc = thresh_c, thresh_c, "train +\nthreshold", INK
            elif y == test_year:
                fc, ec, label, tc = test_c, test_c, "TEST", SURFACE
            else:
                fc, ec, label, tc = SURFACE, GRID, "", MUTED
            ax.add_patch(matplotlib.patches.FancyBboxPatch(
                (x0, r - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.06",
                facecolor=fc, edgecolor=ec, lw=1))
            if label:
                ax.text(x0 + w / 2, r, label, ha="center", va="center", fontsize=8,
                        color=tc, fontweight="bold" if label == "TEST" else "normal",
                        linespacing=0.95)
        n = last - first + 1
        ep, ev = int(counts.loc[test_year, "episodes"]), int(counts.loc[test_year, "events"])
        ax.text(n + 0.35, r, f"{ep}", ha="center", va="center", fontsize=10.5, color=INK)
        ax.text(n + 1.15, r, f"{ev}", ha="center", va="center", fontsize=10.5, color=INK)
    n = last - first + 1
    tot_ep = int(counts.loc[folds, "episodes"].sum())
    tot_ev = int(counts.loc[folds, "events"].sum())
    ax.plot([n + 0.05, n + 1.45], [len(folds) - 0.5] * 2, color=GRID, lw=1)
    ax.text(n + 0.35, len(folds) - 0.1, str(tot_ep), ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=INK)
    ax.text(n + 1.15, len(folds) - 0.1, str(tot_ev), ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=INK)
    ax.text(n - 0.1, len(folds) - 0.1, "pooled test", ha="right", va="center",
            fontsize=9, fontweight="bold", color=INK)
    for j, lab in enumerate(["test\nepisodes", "test\nevents"]):
        ax.text(n + 0.35 + 0.8 * j, -0.62, lab, ha="center", va="bottom", fontsize=8.5,
                fontweight="bold", color=INK, linespacing=0.95)
    for y in range(first, last + 1):
        ax.text(y - first + 0.5, -0.62, str(y), ha="center", va="bottom", fontsize=9.5,
                color=INK)
    ax.set_yticks(range(len(folds)), [f"Fold {i + 1}" for i in range(len(folds))])
    ax.tick_params(length=0, labelsize=9.5, colors=INK)
    ax.set_xticks([])
    ax.set_xlim(-0.05, n + 1.6)
    ax.set_ylim(len(folds) + 0.25, -1.05)
    ax.set_title("Expanding-window evaluation: each year from "
                 f"{OOS_START_YEAR} is tested once, by a model that never saw it",
                 loc="left", fontsize=11.5, pad=14)
    fig.text(0.012, 0.02,
             "Threshold year: a practice model trained on the earlier years scores it, and the "
             "alarm threshold is set there (false-alarm budget).\nThe final model is then "
             "trained on all years before the test year. 25h purge / 24h embargo at each "
             "boundary; episodes straddling a boundary are dropped.",
             fontsize=8, color=MUTED, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)

