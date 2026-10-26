"""
Group depeg episodes into market-wide events (definition only, no model output).

Several coin-episodes can be one market shock (SVB in March 2023 hit USDC,
DAI, PAX and TUSD), so counting episodes overstates how much independent
evidence we have. Events are for reporting only -- labels and training are
untouched.

EVENT RULE
Episodes are sorted by start. An episode joins the current event if it starts
within --window-hours (default 48) of the event's FIRST episode and its coin
is not already in the event; otherwise it opens a new event. Anchoring on the
first episode (not chaining) stops a slow run of dips snowballing into one
event. An event's year is the year of its first episode.

INPUT
  data/processed/depeg_episodes.parquet

OUTPUT (reports/event_tables/)
  episodes.csv  one row per episode with its event_id
  events.csv    one row per event: start, coins, episode IDs
  by_year.csv   number of episodes vs events per year (and per coin)
  figures/      (skip with --no-plots; prices read from master.parquet)
    1_timeline.png        every episode per coin, multi-coin events shaded
    2_event_zooms.png     price of each coin around every multi-coin event
    3_onset_gaps.png      hours to nearest other-coin onset (why 48h)
    4_episodes_vs_events.png  per-year episode vs event counts
    5_coin_year_heatmap.png   episodes per coin per year, with totals

Usage (from the repo root):
    python modelling/reporting/event_table.py
    python modelling/reporting/event_table.py --window-hours 72
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "data_processing"))

from config import DEPEG_BAND, FIAT_COINS, OOS_START_YEAR, PROCESSED_DATA_DIR  # noqa: E402

# fixed coin -> colour (categorical order, validated for CVD separation);
# coins are also always labelled, so colour never carries identity alone
COIN_COLOURS = dict(zip(FIAT_COINS, ["#2a78d6", "#eb6834", "#1baf7a",
                                     "#eda100", "#e87ba4", "#008300"]))
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a64", "#e4e3de", "#fcfcfb"
EVENT_SHADE = "#9ec5f4"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
    "ytick.color": MUTED, "text.color": INK, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
})


def group_events(episodes: pd.DataFrame, window_hours: float) -> pd.DataFrame:
    """Add event_id / event_start to each episode (anchored on the first episode)."""
    eps = episodes.sort_values(["start", "coin"]).reset_index(drop=True).copy()
    window = pd.Timedelta(hours=window_hours)
    event_ids, event_starts = [], []
    current_start, current_coins, current_id = None, set(), None
    used_ids: dict[str, int] = {}
    for row in eps.itertuples(index=False):
        if (current_start is None or row.start - current_start > window
                or row.coin in current_coins):
            current_start, current_coins = row.start, set()
            base = f"EVT_{row.start:%Y%m%d}"
            used_ids[base] = used_ids.get(base, 0) + 1
            current_id = base if used_ids[base] == 1 else f"{base}_{used_ids[base]}"
        current_coins.add(row.coin)
        event_ids.append(current_id)
        event_starts.append(current_start)
    eps["event_id"] = event_ids
    eps["event_start"] = event_starts
    eps["year"] = eps["event_start"].dt.year
    return eps


def plot_timeline(eps: pd.DataFrame, ev_tab: pd.DataFrame, path: Path) -> None:
    """One lane per coin, one dot per episode; multi-coin events shaded and numbered."""
    fig, ax = plt.subplots(figsize=(13, 3.8))
    lanes = {c: i for i, c in enumerate(FIAT_COINS[::-1])}
    multi = ev_tab.loc[ev_tab["n_episodes"] > 1].reset_index(drop=True)
    last_x, level = None, 0
    for i, ev in multi.iterrows():
        lo, hi = ev["event_start"], ev["last_episode_start"] + pd.Timedelta(days=4)
        ax.axvspan(lo - pd.Timedelta(days=4), hi, color=EVENT_SHADE, alpha=0.45, lw=0)
        # stagger labels of events close in time so the numbers don't collide
        level = level + 1 if last_x is not None and lo - last_x < pd.Timedelta(days=40) else 0
        last_x = lo
        ax.text(lo, len(lanes) - 0.45 + 0.28 * level, str(i + 1), ha="center",
                va="bottom", fontsize=7.5, color=INK)
    for coin, g in eps.groupby("coin"):
        size = 18 + 4 * np.sqrt(g["n_hours"])
        ax.scatter(g["start"], [lanes[coin]] * len(g), s=size, color=COIN_COLOURS[coin],
                   edgecolor=SURFACE, linewidth=1, zorder=3)
    ax.set_yticks(list(lanes.values()), list(lanes.keys()))
    ax.set_ylim(-0.6, len(lanes) + 0.5)
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_title(f"{len(eps)} depeg episodes form {len(ev_tab)} events; "
                 f"shaded = the {len(multi)} events spanning several coins", loc="left")
    ax.text(0, -0.17, "Dot = one episode (size ~ hours below $0.99). Numbers match "
            "the panels in 2_event_zooms.png.", transform=ax.transAxes, color=MUTED, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_event_zooms(eps: pd.DataFrame, ev_tab: pd.DataFrame, prices: pd.DataFrame,
                     window_hours: float, path: Path) -> None:
    """Small multiples: price of every coin around each multi-coin event.

    Member coins are drawn in colour; non-member coins in light grey for context.
    Shaded spans are each member's episode; the dashed box is the grouping window."""
    multi = ev_tab.loc[ev_tab["n_episodes"] > 1].reset_index(drop=True)
    ncol = 4
    nrow = math.ceil(len(multi) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(14, 3.1 * nrow), squeeze=False)
    floor = 1 - DEPEG_BAND
    for i, ev in multi.iterrows():
        ax = axes[i // ncol][i % ncol]
        t0 = ev["event_start"]
        lo, hi = t0 - pd.Timedelta(hours=36), t0 + pd.Timedelta(hours=window_hours + 72)
        members = eps.loc[eps["event_id"] == ev["event_id"]]
        win = prices.loc[(prices["hour"] >= lo) & (prices["hour"] <= hi)]
        ax.axvspan(t0, t0 + pd.Timedelta(hours=window_hours), facecolor="none",
                   edgecolor=MUTED, ls="--", lw=0.8, zorder=1)
        for coin, g in win.groupby("coin"):
            if coin in set(members["coin"]):
                continue
            ax.plot(g["hour"], g["price"], color=GRID, lw=1, zorder=2)
        for m in members.itertuples(index=False):
            c = COIN_COLOURS[m.coin]
            g = win.loc[win["coin"] == m.coin]
            ax.axvspan(m.start, m.end + pd.Timedelta(hours=1), color=c, alpha=0.12, lw=0)
            lag = (m.start - t0) / pd.Timedelta(hours=1)
            ax.plot(g["hour"], g["price"], color=c, lw=1.6, zorder=3,
                    label=f"{m.coin} (+{lag:.0f}h)")
        ax.axhline(floor, color=INK, lw=0.8, zorder=2)
        ymin = min(members["min_price"].min(), floor) - 0.004
        ax.set_ylim(max(ymin, 0.85), 1.012)
        ax.set_xlim(lo, hi)
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1 if window_hours <= 48 else 2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        ax.tick_params(axis="x", labelrotation=0, labelsize=7)
        ax.set_title(f"{i + 1}. {t0:%d %b %Y} - {ev['n_episodes']} coins", loc="left")
        ax.legend(loc="lower right", fontsize=7, frameon=True, framealpha=0.9,
                  edgecolor=GRID, handlelength=1.2, borderpad=0.4, labelspacing=0.25)
    for j in range(len(multi), nrow * ncol):
        axes[j // ncol][j % ncol].set_visible(False)
    fig.suptitle(f"Multi-coin events: hourly price of member coins (colour; legend = "
                 f"hours after the first onset), other coins grey\nshaded = each "
                 f"episode, dashed box = {window_hours:g}h grouping window, "
                 f"black line = ${floor:.2f}", x=0.01, ha="left", fontsize=10.5,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_onset_gaps(eps: pd.DataFrame, window_hours: float, path: Path) -> None:
    """For each episode, hours to the nearest onset on a DIFFERENT coin."""
    starts = eps[["coin", "start"]].reset_index(drop=True)
    gaps = []
    for r in starts.itertuples(index=False):
        other = starts.loc[starts["coin"] != r.coin, "start"]
        gaps.append(((other - r.start).abs().min()) / pd.Timedelta(hours=1))
    gaps = np.array(gaps)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.set_axisbelow(True)
    bins = np.logspace(0, np.log10(max(gaps.max(), 10) + 1), 40)
    ax.hist(np.clip(gaps, 1, None), bins=bins, color="#2a78d6", edgecolor=SURFACE, lw=1)
    ax.set_xscale("log")
    ax.axvline(window_hours, color=INK, lw=1, ls="--")
    within = int((gaps <= window_hours).sum())
    ax.text(window_hours * 1.08, ax.get_ylim()[1] * 0.92,
            f"{window_hours:g}h: {within} of {len(gaps)} episodes\nhave another coin's onset this close",
            fontsize=8, color=INK, va="top")
    ax.set_xlabel("hours to nearest onset on another coin (log scale; 0h shown at 1h)")
    ax.set_ylabel("episodes")
    ax.set_title("Same-shock onsets cluster within hours; unrelated ones are weeks apart",
                 loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_episodes_vs_events(by_year: pd.DataFrame, path: Path) -> None:
    yrs = by_year.loc[by_year["year"] != "total"]
    x = np.arange(len(yrs))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.set_axisbelow(True)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    b1 = ax.bar(x - w / 2 - 0.01, yrs["episodes"], w, color="#2a78d6", label="episodes")
    b2 = ax.bar(x + w / 2 + 0.01, yrs["events"], w, color="#eb6834", label="events")
    for bars in (b1, b2):
        ax.bar_label(bars, fontsize=7.5, color=INK, padding=2)
    ax.set_xticks(x, yrs["year"].astype(str))
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper right")
    ax.set_ylabel("count")
    ax.set_title("Independent evidence per year: coin-episodes vs grouped events", loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_coin_year_heatmap(by_year: pd.DataFrame, test_start: int, path: Path) -> None:
    """Episodes per coin per year (one-hue blue ramp), totals as plain columns."""
    yrs = by_year.loc[by_year["year"] != "total"].copy()
    coins = [c for c in ["USDT", "USDC", "BUSD", "DAI", "TUSD", "PAX"]
             if f"episodes_{c}" in yrs.columns]
    grid = yrs[[f"episodes_{c}" for c in coins]].to_numpy(dtype=float)
    totals = by_year.loc[by_year["year"] == "total"]
    ramp = matplotlib.colors.LinearSegmentedColormap.from_list(
        "blue", ["#f1f6fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                 "#256abf", "#184f95", "#0d366b"])
    n_rows, n_cols = grid.shape
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.imshow(grid, cmap=ramp, vmin=0, vmax=max(grid.max(), 1), aspect="auto",
              extent=(-0.5, n_cols - 0.5, n_rows - 0.5, -0.5))
    # 2px-style surface gaps between cells
    for x in np.arange(-0.5, n_cols, 1):
        ax.axvline(x, color=SURFACE, lw=2.5)
    for y in np.arange(-0.5, n_rows, 1):
        ax.axhline(y, color=SURFACE, lw=2.5)
    hi = grid.max()
    for r in range(n_rows):
        for c in range(n_cols):
            v = int(grid[r, c])
            ax.text(c, r, "–" if v == 0 else str(v), ha="center", va="center",
                    fontsize=11, fontweight="bold" if v else "normal",
                    color=(SURFACE if v > hi * 0.5 else (MUTED if v == 0 else INK)))
    # totals columns (plain text, not coloured: different measure)
    for j, (col, label) in enumerate([("episodes", "Episodes"), ("events", "Events")]):
        x = n_cols + 0.35 + j * 1.05
        ax.text(x, -0.85, label, ha="center", va="bottom", fontsize=9.5,
                fontweight="bold", color=INK)
        for r, v in enumerate(yrs[col]):
            ax.text(x, r, str(int(v)), ha="center", va="center", fontsize=11, color=INK)
        ax.text(x, n_rows + 0.15, str(int(totals[col].iloc[0])), ha="center",
                va="center", fontsize=11, fontweight="bold", color=INK)
    for c, coin in enumerate(coins):
        ax.text(c, n_rows + 0.15, str(int(totals[f"episodes_{coin}"].iloc[0])),
                ha="center", va="center", fontsize=11, fontweight="bold", color=INK)
    ax.text(-0.6, n_rows + 0.15, "Total", ha="right", va="center", fontsize=10,
            fontweight="bold", color=INK)
    ax.axhline(n_rows - 0.5 + 0.17, xmin=0, xmax=1, color=GRID, lw=1)
    # bracket the test years
    test_rows = [i for i, y in enumerate(yrs["year"]) if int(y) >= test_start]
    if test_rows:
        top, bot = test_rows[0] - 0.45, test_rows[-1] + 0.45
        xb = n_cols + 2.25
        ax.plot([xb - 0.12, xb, xb, xb - 0.12], [top, top, bot, bot], color=MUTED,
                lw=1, clip_on=False)
        ep = int(yrs.loc[yrs["year"].astype(int) >= test_start, "episodes"].sum())
        ev = int(yrs.loc[yrs["year"].astype(int) >= test_start, "events"].sum())
        ax.text(xb + 0.12, (top + bot) / 2, f"test years\n{ep} episodes\n{ev} events",
                ha="left", va="center", fontsize=8.5, color=MUTED, clip_on=False)
    for c, coin in enumerate(coins):  # same baseline as the totals headers
        ax.text(c, -0.85, coin, ha="center", va="bottom", fontsize=10, color=INK)
    ax.set_xticks([])
    ax.tick_params(length=0, labelsize=10, colors=INK)
    ax.set_yticks(range(n_rows), yrs["year"].astype(str))
    ax.set_xlim(-0.5, n_cols + 2.9)
    ax.set_ylim(n_rows + 0.6, -0.95)
    ax.set_title("Depeg episodes per coin per year", loc="left", pad=26, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--window-hours", type=float, default=48,
                        help="episodes on different coins starting within this many hours "
                             "of an event's first episode join that event (default 48)")
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "reports/event_tables"))
    parser.add_argument("--no-plots", action="store_true", help="skip the figures")
    args = parser.parse_args(argv)

    episodes = pd.read_parquet(PROCESSED_DATA_DIR / "depeg_episodes.parquet")
    for col in ("start", "end"):
        episodes[col] = pd.to_datetime(episodes[col], utc=True)
    eps = group_events(episodes, args.window_hours)

    ep_tab = eps[["event_id", "event_start", "year", "episode_id", "coin",
                  "start", "end", "n_hours", "min_price"]]
    ev_tab = (eps.groupby("event_id", sort=False)
              .agg(event_start=("event_start", "first"),
                   last_episode_start=("start", "max"),
                   year=("year", "first"),
                   n_episodes=("episode_id", "size"),
                   coins=("coin", ",".join),
                   episode_ids=("episode_id", ",".join),
                   min_price=("min_price", "min"))
              .reset_index())

    by_year = (eps.groupby("year")
               .agg(episodes=("episode_id", "size"), events=("event_id", "nunique")))
    by_year = by_year.join(eps.pivot_table(index="year", columns="coin",
                                           values="episode_id", aggfunc="size",
                                           fill_value=0).add_prefix("episodes_"))
    by_year.loc["total"] = by_year.sum()
    by_year["episodes_per_event"] = (by_year["episodes"] / by_year["events"]).round(2)
    by_year = by_year.reset_index()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in {"episodes": ep_tab, "events": ev_tab, "by_year": by_year}.items():
        path = out_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        print(f"Saved {name}: {path}")

    print(f"\n{len(eps)} episodes -> {len(ev_tab)} events "
          f"(window {args.window_hours:g}h)\n")
    print(by_year.to_string(index=False))
    print("\nMulti-coin events:")
    multi = ev_tab.loc[ev_tab["n_episodes"] > 1]
    print(multi[["event_id", "event_start", "n_episodes", "coins"]].to_string(index=False))

    if not args.no_plots:
        fig_dir = out_dir / "figures"
        fig_dir.mkdir(exist_ok=True)
        prices = pd.read_parquet(PROCESSED_DATA_DIR / "master.parquet",
                                 columns=["coin", "hour", "price"])
        prices["hour"] = pd.to_datetime(prices["hour"], utc=True)
        plot_timeline(eps, ev_tab, fig_dir / "1_timeline.png")
        plot_event_zooms(eps, ev_tab, prices, args.window_hours, fig_dir / "2_event_zooms.png")
        plot_onset_gaps(eps, args.window_hours, fig_dir / "3_onset_gaps.png")
        plot_episodes_vs_events(by_year, fig_dir / "4_episodes_vs_events.png")
        plot_coin_year_heatmap(by_year, OOS_START_YEAR, fig_dir / "5_coin_year_heatmap.png")
        print(f"\nSaved figures: {fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
