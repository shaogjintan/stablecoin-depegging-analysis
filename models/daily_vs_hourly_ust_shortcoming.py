"""
Workstream 1, deliverable 2a: "Show that daily models react too late to
intraday crashes (plot the daily vs. hourly view of the May 9 UST collapse)."

WHAT THIS SCRIPT DOES
----------------------
Lee et al. (2025) -- and most of the depeg-prediction literature we're
replicating/critiquing -- operates on **daily** data: one row per coin per
day, predicting the next day's depeg from today's features. This script
asks a simple question their design can't answer: what did the UST
collapse actually look like *within* the days their model treats as single
observations, and how much later does a daily view notice it?

We build the same price series at two frequencies (hourly and daily, both
derived from one source so they're directly comparable) and mark the first
point at which each frequency's own close price crosses a depeg threshold.
The literature doesn't agree on one threshold -- Cintra and Holloway (2023),
cited by Lee et al. as prior work, explicitly recommend using a *tighter*
band for finer-grained data to control noise: they suggest 5% for hourly
data and 1% for daily data. We use exactly those two numbers here, so the
comparison isn't "same threshold, different frequency" (which would be an
unfair test -- a coarser threshold at daily granularity is a modeling
choice, not just data), it's "each frequency's own literature-recommended
threshold, applied honestly."

DATA SOURCE
-----------
Binance's public klines API, symbol USTUSDT, 1h interval, May 5-14 2022.
Neither Yahoo Finance nor CoinGecko's free tier can serve this: Yahoo only
keeps ~2 years of *hourly* history from today (2022 is far outside that
window by now), and CoinGecko's free API refuses anything older than 365
days (see lee_et_al_replication.py's docstring for the same wall). Binance
has no such restriction and still serves USTUSDT's full trading history up
to its mid-2022 delisting, so it's the one source that actually reaches
back to the event itself. Cached locally at
data/hourly/ustusdt_hourly_may2022.csv (gitignored; re-run to regenerate).

We deliberately build BOTH the "hourly" and "daily" series from this one
hourly pull (resampling it down to daily OHLC ourselves) rather than pairing
it with, say, our existing Yahoo-sourced daily USTC series -- two different
vendors can disagree slightly on intraday high/low (we saw this with USDC
in lee_et_al_replication.py), which would confound "different frequency"
with "different data source". One series, two resamplings, is a cleaner
comparison.

OUTPUT
------
models/figures/daily_vs_hourly_ust.png -- two-panel chart for the slide.
"""
import os
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import requests

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOURLY_DIR = os.path.join(REPO_ROOT, "data", "hourly")
FIGURES_DIR = os.path.join(REPO_ROOT, "models", "figures")
CACHE_PATH = os.path.join(HOURLY_DIR, "ustusdt_hourly_may2022.csv")

SYMBOL = "USTUSDT"
FETCH_START = "2022-05-05"   # a few calm days before the Curve 3pool imbalance (May 7-8)
FETCH_END = "2022-05-14"     # collapse is essentially complete by here (close ~$0.25)

# Cintra & Holloway (2023) -- cited in Lee et al. Section 2 -- recommend a
# tighter band for finer-grained data to control noise: 5% for hourly data,
# 1% for daily data. We adopt both thresholds as-is (downward only, since
# UST's collapse was one-directional).
HOURLY_THRESHOLD = 0.95   # -5%
DAILY_THRESHOLD = 0.99    # -1%

# Lee et al.'s own design predicts day t+1's depeg from day t's features
# (see lee_et_al_replication.py) -- so even a same-day daily detection isn't
# when their *model* would flag risk; that happens one full day later still.
NEXT_DAY_LAG_HOURS = 24

# Palette (dataviz skill reference palette): categorical slot 1 (blue) for
# the actual price series, the fixed status "critical" red for the depeg
# threshold/breach marker -- a status color, not a second series, so it's
# never asked to double as a legend entry.
COLOR_PRICE = "#2a78d6"
COLOR_CRITICAL = "#d03b3b"
COLOR_MUTED = "#898781"
COLOR_INK = "#0b0b0b"
COLOR_INK_SECONDARY = "#52514e"
COLOR_GRID = "#e1e0d9"
COLOR_SURFACE = "#fcfcfb"


def fetch_hourly_klines() -> pd.DataFrame:
    """Pull USTUSDT 1h klines from Binance (or load the cached copy)."""
    if os.path.exists(CACHE_PATH):
        df = pd.read_csv(CACHE_PATH)
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
        return df

    def to_ms(d: str) -> int:
        return int(datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)

    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": SYMBOL, "interval": "1h",
        "startTime": to_ms(FETCH_START), "endTime": to_ms(FETCH_END),
        "limit": 1000,
    }
    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    raw = resp.json()

    df = pd.DataFrame(raw, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "n_trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df = df[["open_time", "open", "high", "low", "close", "volume"]]

    os.makedirs(HOURLY_DIR, exist_ok=True)
    df.to_csv(CACHE_PATH, index=False)
    return df


def resample_to_daily(hourly: pd.DataFrame) -> pd.DataFrame:
    """Collapse the hourly series to one row per UTC day -- proper OHLC
    aggregation (first open, max high, min low, last close), not just a
    'take every 24th row' sample.
    """
    daily = hourly.set_index("open_time").resample("1D").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
    })
    return daily.dropna().reset_index().rename(columns={"open_time": "date"})


def first_breach(df: pd.DataFrame, time_col: str, threshold: float) -> pd.Series | None:
    """First row where close <= threshold, or None if it never breaches."""
    breached = df[df["close"] <= threshold]
    if breached.empty:
        return None
    return breached.iloc[0]


def main():
    hourly = fetch_hourly_klines()
    daily = resample_to_daily(hourly)

    hourly_breach = first_breach(hourly, "open_time", HOURLY_THRESHOLD)
    daily_breach = first_breach(daily, "date", DAILY_THRESHOLD)

    if hourly_breach is None or daily_breach is None:
        raise RuntimeError("Threshold never breached in the fetched window -- widen FETCH_START/FETCH_END.")

    hourly_breach_time = hourly_breach["open_time"]
    daily_breach_time = daily_breach["date"]
    # daily_breach_time is midnight UTC of the breach day; the day's actual
    # close (hence its "detection") isn't knowable until that day ends --
    # i.e. the true detection instant is the *next* midnight.
    daily_detection_time = daily_breach_time + pd.Timedelta(days=1)
    model_flag_time = daily_detection_time + pd.Timedelta(hours=NEXT_DAY_LAG_HOURS)

    lag_hourly_to_daily = (daily_detection_time - hourly_breach_time).total_seconds() / 3600
    lag_hourly_to_model = (model_flag_time - hourly_breach_time).total_seconds() / 3600

    print(f"Hourly breach (close <= {HOURLY_THRESHOLD}): {hourly_breach_time} UTC, close={hourly_breach['close']:.4f}")
    print(f"Daily breach  (close <= {DAILY_THRESHOLD}): day of {daily_breach_time.date()}, "
          f"close={daily_breach['close']:.4f} -- not knowable until day-end, {daily_detection_time} UTC")
    print(f"Lag, true onset -> daily-close detection: {lag_hourly_to_daily:.1f} hours")
    print(f"Lag, true onset -> Lee-et-al.-style next-day model flag: {lag_hourly_to_model:.1f} hours")

    # ---- Figure: two stacked panels, one shared time axis, one price unit ----
    fig, (ax_hourly, ax_daily) = plt.subplots(
        2, 1, figsize=(11, 7.2), sharex=True, sharey=True,
        gridspec_kw={"height_ratios": [1, 1]},
    )
    fig.patch.set_facecolor(COLOR_SURFACE)
    # Manual layout instead of tight_layout: tight_layout's auto title-space
    # estimate (suptitle + two axis titles) reserves far more headroom than
    # this figure actually needs, leaving a large dead band at the top.
    fig.subplots_adjust(top=0.82, bottom=0.08, left=0.07, right=0.97, hspace=0.4)

    for ax in (ax_hourly, ax_daily):
        ax.set_facecolor(COLOR_SURFACE)
        ax.grid(True, color=COLOR_GRID, linewidth=0.8, zorder=0)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color(COLOR_MUTED)
        ax.tick_params(colors=COLOR_INK_SECONDARY, labelsize=9)
        ax.set_ylabel("Price (USD)", color=COLOR_INK_SECONDARY, fontsize=10)
        ax.margins(y=0.12)  # sensible autoscale off the real $0.24-$1.00 data range,
        # not a hand-picked ceiling -- annotations use axes-fraction placement
        # (see below) instead of inflating this range to make room for text.

    # --- Top: hourly ---
    ax_hourly.plot(hourly["open_time"], hourly["close"], color=COLOR_PRICE, linewidth=2, zorder=3)
    ax_hourly.axhline(HOURLY_THRESHOLD, color=COLOR_CRITICAL, linewidth=1.2, linestyle="--", zorder=2)
    ax_hourly.text(
        hourly["open_time"].iloc[-1], HOURLY_THRESHOLD, f"hourly threshold −5% (${HOURLY_THRESHOLD:.2f})  ",
        color=COLOR_CRITICAL, fontsize=8.5, va="bottom", ha="right",
    )
    ax_hourly.axvline(hourly_breach_time, color=COLOR_CRITICAL, linewidth=1.2, zorder=2)
    ax_hourly.scatter([hourly_breach_time], [hourly_breach["close"]], color=COLOR_CRITICAL, s=45,
                       zorder=4, edgecolors=COLOR_SURFACE, linewidths=1)
    ax_hourly.annotate(
        f"Actual onset: {hourly_breach_time.strftime('%b %d, %H:%M')} UTC (${hourly_breach['close']:.3f})",
        xy=(hourly_breach_time, hourly_breach["close"]),
        xytext=(12, 30), textcoords="offset points",
        fontsize=8.5, color=COLOR_INK,
        arrowprops=dict(arrowstyle="-", color=COLOR_CRITICAL, linewidth=1),
    )
    ax_hourly.set_title("Hourly view — when the collapse actually happened", loc="left",
                         fontsize=11.5, color=COLOR_INK, fontweight="bold", pad=8)

    # --- Bottom: daily ---
    ax_daily.plot(daily["date"], daily["close"], color=COLOR_PRICE, linewidth=2,
                  marker="o", markersize=8, zorder=3)
    ax_daily.axhline(DAILY_THRESHOLD, color=COLOR_CRITICAL, linewidth=1.2, linestyle="--", zorder=2)
    ax_daily.text(
        daily["date"].iloc[-1], DAILY_THRESHOLD, f"daily threshold −1% (${DAILY_THRESHOLD:.2f})  ",
        color=COLOR_CRITICAL, fontsize=8.5, va="top", ha="right",
    )
    ax_daily.axvline(daily_detection_time, color=COLOR_CRITICAL, linewidth=1.2, zorder=2)
    ax_daily.scatter([daily_breach_time], [daily_breach["close"]], color=COLOR_CRITICAL, s=65,
                      zorder=4, edgecolors=COLOR_SURFACE, linewidths=1)
    ax_daily.annotate(
        f"Close for {daily_breach_time.strftime('%b %d')}: ${daily_breach['close']:.3f}",
        xy=(daily_breach_time, daily_breach["close"]),
        xytext=(-100, -35), textcoords="offset points",
        fontsize=8.5, color=COLOR_INK,
        arrowprops=dict(arrowstyle="-", color=COLOR_CRITICAL, linewidth=1),
    )
    # Placed with a blended transform (x in data units, y in axes-fraction) so
    # it sits near the top of the panel regardless of the data's actual price
    # range -- no need to artificially stretch the y-axis to make room for it.
    ax_daily.text(
        daily_detection_time, 0.95,
        f"Day-end confirms it: {daily_detection_time.strftime('%b %d')} 00:00 UTC "
        f"(+{lag_hourly_to_daily:.0f}h vs. actual onset)",
        transform=ax_daily.get_xaxis_transform(),
        fontsize=8.5, color=COLOR_INK, va="bottom", ha="left",
    )
    ax_daily.set_title("Daily view — what a daily-frequency model sees", loc="left",
                        fontsize=11.5, color=COLOR_INK, fontweight="bold", pad=8)

    # Both panels get their own explicit date ticks -- with sharex=True the
    # x-*limits* are linked automatically, but each Axes still needs its own
    # locator/formatter set. (An earlier version relied on fig.autofmt_xdate()
    # for this, which -- by design -- hides tick labels on every row except
    # the bottom one; that's what made the hourly panel's x-axis unreadable.)
    for ax in (ax_hourly, ax_daily):
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.xaxis.set_major_locator(mdates.DayLocator())
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center", visible=True)
    ax_daily.set_xlabel("Date (UTC)", color=COLOR_INK_SECONDARY, fontsize=10)

    fig.text(
        0.07, 0.965,
        "Daily models react too late: the May 9, 2022 UST collapse",
        fontsize=15,
        color=COLOR_INK, fontweight="bold", ha="left", va="top",
    )
    fig.text(
        0.07, 0.925,
        "Same USTUSDT data (Binance), resampled to two frequencies. A next-day-ahead\n"
        f"model like Lee et al. (2025) would only flag this ~{lag_hourly_to_model:.0f}h after the actual onset.",
        fontsize=9.5, color=COLOR_INK_SECONDARY, ha="left", va="top",
    )

    os.makedirs(FIGURES_DIR, exist_ok=True)
    out_path = os.path.join(FIGURES_DIR, "daily_vs_hourly_ust.png")
    fig.savefig(out_path, dpi=200, facecolor=COLOR_SURFACE)
    print(f"Saved figure -> {out_path}")


if __name__ == "__main__":
    main()
